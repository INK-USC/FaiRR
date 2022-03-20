'''
This is the inference script that performs the iterative inference and also computes the proof graph for the given statement and theory.
'''

from helper import *

from basemodel import BaseModel
from proofwriter_classes import PWReasonerInstance, PWQRuleInstance, PWQFactInstance
from fairr_ruleselector_model import FaiRRRuleSelector
from fairr_factselector_model import FaiRRFactSelector
from fairr_reasoner_model import FaiRRReasoner


class FaiRRInference(BaseModel):

	# counter to count the # times proof generation fails (mainly due to cycles in proof graph)
	count_error_graphs = 0

	# local accounting of proof accuracy
	local_proof_accuracy = []
	local_step = 0

	def __init__(self, ruleselector_ckpt, factselector_ckpt, reasoner_ckpt, arch='', train_batch_size=1, eval_batch_size=1, accumulate_grad_batches=1, learning_rate=1e-5, \
					max_epochs=1, optimizer='adamw', adam_epsilon=1e-8, weight_decay=0.0, lr_scheduler='fixed', warmup_updates=0.0, freeze_epochs=-1, gpus=1):
		super().__init__(train_batch_size=train_batch_size, max_epochs=max_epochs, gpus=gpus)
		self.save_hyperparameters()

		self.p                         = types.SimpleNamespace()
		self.p.arch                    = arch
		self.p.train_batch_size        = train_batch_size
		self.p.eval_batch_size         = eval_batch_size
		self.p.accumulate_grad_batches = accumulate_grad_batches
		self.p.learning_rate           = learning_rate
		self.p.max_epochs              = max_epochs
		self.p.optimizer               = optimizer
		self.p.adam_epsilon            = adam_epsilon
		self.p.weight_decay            = weight_decay
		self.p.lr_scheduler            = lr_scheduler
		self.p.warmup_updates          = warmup_updates
		self.p.freeze_epochs           = freeze_epochs
		self.p.gpus                    = gpus

		self.rule_selector             = FaiRRRuleSelector().load_from_checkpoint(ruleselector_ckpt)
		self.rule_tokenizer            = self.rule_selector.tokenizer

		self.fact_selector             = FaiRRFactSelector().load_from_checkpoint(factselector_ckpt)
		self.fact_tokenizer            = self.fact_selector.tokenizer

		self.reasoner                  = FaiRRReasoner().load_from_checkpoint(reasoner_ckpt)
		self.reasoner_tokenizer        = self.reasoner.tokenizer

	def forward(self, batch):
		facts         = batch['all_facts']
		rules         = batch['all_rules']
		ques          = batch['all_ques']
		batch_size    = len(facts)
		device        = self.reasoner.device
		count         = 0
		stop          = False
		output_dict   = [dict() for _ in range(batch_size)]
		proof_dict    = [ddict(list) for _ in range(batch_size)]

		# prefill the proof_dict with single triples
		for idx in range(batch_size):
			for fact in facts[idx]:
				proof_dict[idx][fact].append(([fact], ''))		# value format: ([facts], rule)

		try:
			while not stop:
				# process data for rule selector and select rule
				input_ids, attn_mask, token_mask = PWQRuleInstance.tokenize_batch(self.rule_tokenizer, rules, facts, ques)
				rule_ids, rule_mask = self.rule_selector.predict(input_ids.to(device), token_mask.to(device), attn_mask.to(device))

				# loop break condition
				if rule_mask.sum().item() == 0:
					stop = True
					break

				for idx in range(rule_ids.shape[1]):
					selected_rules = [rules[x][y] for x,y in zip(range(batch_size), rule_ids[:, idx])]

					# this will be used to determine which inferences to keep and which ones to reject (batching trick)
					valid_mask     = rule_mask[:, idx]

					# process data for fact selector and select facts for the selected rule
					input_ids, attn_mask, token_mask = PWQFactInstance.tokenize_batch(self.fact_tokenizer, selected_rules, facts, ques)
					fact_ids, fact_mask = self.fact_selector.predict(input_ids.to(device), token_mask.to(device), attn_mask.to(device))

					# update valid_mask to account for cases when no facts are selected (batching trick)
					valid_mask     = valid_mask * fact_mask

					# if nothing is valid then stop
					if valid_mask.sum() == 0:
						stop = True
						break

					selected_facts = [[facts[x][y] for y in fact_ids[x] if y != -1] for x in range(batch_size)]

					# generate intermediate conclusion
					input_ids   = PWReasonerInstance.tokenize_batch(self.reasoner_tokenizer, selected_rules, selected_facts)
					conclusions = self.reasoner.predict_and_decode(torch.LongTensor(input_ids).to(device))

					new_conc = False	# This flag checks if any new intermediate conclusion was generated in this round for any of the instance in the batch
					for batch_idx in range(batch_size):
						if valid_mask[batch_idx]:
							# add proof to output_dict and increase count
							out_key   = ' '.join(selected_facts[batch_idx]) + '::' + selected_rules[batch_idx] + '::' + conclusions[batch_idx].lower()
							proof_key = conclusions[batch_idx].lower()

							if out_key not in output_dict[batch_idx]:
								new_conc = True
								output_dict[batch_idx][out_key] = 1
								facts[batch_idx].append(conclusions[batch_idx].lower())

								if len(selected_facts[batch_idx]) == 0:
									sys.stdout = sys.__stdout__; import pdb; pdb.set_trace()

								# update proof_dict
								proof_dict[batch_idx][proof_key].append((selected_facts[batch_idx], selected_rules[batch_idx]))
							else:
								output_dict[batch_idx][out_key] += 1

					facts = [list(set(x)) for x in facts]

					# if there are no new conclusions in the batch and all selected rules have been tried, then stop
					if not new_conc and (idx + 1 == rule_ids.shape[1]):
						stop = True

					# fail-safe to check for infinite loops cases, if any
					count += 1
					if count == 1000:
						print('Stop hit!')
						sys.stdout = sys.__stdout__; import pdb; pdb.set_trace()

		except Exception as e:
			print('Exception Cause: {}'.format(e.args[0]))
			print(traceback.format_exc())

		# solve each instance in batch
		results = []
		for idx in range(batch_size):
			ans, prf = self.solver(facts[idx], ques[idx], dict(proof_dict[idx]))
			results.append((ans, prf))

		return results

	def solver(self, facts, ques, proof_dict, gold_proof=None, gold_ans=None):
		try:
			# check if question is already in facts
			if ques in facts:
				proofs = generate_proof(ques, proof_dict)
				return (1, proofs)
			else:
				# try to negate the ques and see if its present
				ques_neg = negate(ques)
				if ques_neg in facts:
					proofs = generate_proof(ques_neg, proof_dict)
					return (-1, proofs)
				else:
					# no proof exists.
					return (0, [['None']])
		except Exception as e:
			self.count_error_graphs += 1
			return (0, [['None']])

	def calc_acc(self, preds, targets):
		matched = np.array(preds) == np.array(targets)
		return 100 * np.mean(matched), matched

	def match_proof(self, all_proofs, all_gold_proofs, ans_match):
		res = []
		for idx in range(len(all_proofs)):
			proofs = all_proofs[idx]
			gold_proofs = all_gold_proofs[idx]

			gold_proofs_counter = [Counter(x) for x in gold_proofs]
			gold_proofs_counter = [Counter({y:1 for y in x}) for x in gold_proofs_counter]

			found = False
			for prf in proofs:
				if Counter({y:1 for y in Counter(prf)}) in gold_proofs_counter:
					found = True
					break

			res.append(found)

		final_res = res * ans_match

		return 100 * np.mean(final_res), final_res

	def run_step(self, batch, split):
		out         = self(batch)
		targets     = batch['all_answer']
		gold_proofs = batch['all_proof']

		# calculate question entailment accuracy
		preds              = [x[0] for x in out]
		ans_acc, ans_match = self.calc_acc(preds, targets)
		ans_acc            = torch.FloatTensor([ans_acc]).to(self.reasoner.device)

		# calculate proof match accuracy
		proofs             = [x[1] for x in out]
		prf_acc, prf_match = self.match_proof(proofs, gold_proofs, ans_match)
		self.local_proof_accuracy.append(prf_acc)
		prf_acc = torch.FloatTensor([prf_acc]).to(self.reasoner.device)

		self.local_step += 1
		if self.local_step % 20 == 0:
			print(f'\nProof Accuracy: {np.mean(self.local_proof_accuracy)}\n')

		if split == 'train':
			self.log(f'train_ans_acc_step', ans_acc, prog_bar=True)
			self.log(f'train_prf_acc_step', prf_acc, prog_bar=True)
		else:
			self.log(f'{split}_ans_acc_step', ans_acc, prog_bar=True, sync_dist=True)
			self.log(f'{split}_prf_acc_step', prf_acc, prog_bar=True, sync_dist=True)

		return {'ans_acc': ans_acc, 'prf_acc': prf_acc, 'loss': torch.FloatTensor([0]).to(self.reasoner.device)}

	def aggregate_epoch(self, outputs, split):
		ans_acc = torch.stack([x['ans_acc'] for x in outputs]).mean()
		prf_acc = torch.stack([x['prf_acc'] for x in outputs]).mean()

		if split == 'train':
			self.log(f'train_ans_acc_epoch', ans_acc.item())
			self.log(f'train_prf_acc_epoch', prf_acc.item())
		else:
			self.log(f'{split}_ans_acc_epoch', ans_acc.item(), sync_dist=True)
			self.log(f'{split}_prf_acc_epoch', prf_acc.item(), sync_dist=True)
			self.log(f'Graph Cycle Errors: ', self.count_error_graphs, sync_dist=True)

	def configure_optimizers(self):
		no_decay = ['bias', 'LayerNorm.weight']
		optimizer_grouped_parameters = [
			{
				'params'      : [p for n, p in self.rule_selector.named_parameters() if not any(nd in n for nd in no_decay)],
				'weight_decay': self.p.weight_decay,
			},
			{
				'params'      : [p for n, p in self.rule_selector.named_parameters() if any(nd in n for nd in no_decay)],
				'weight_decay': 0.0,
			}
		]

		optimizer_grouped_parameters += [
			{
				'params'      : [p for n, p in self.fact_selector.named_parameters() if not any(nd in n for nd in no_decay)],
				'weight_decay': self.p.weight_decay,
			},
			{
				'params'      : [p for n, p in self.fact_selector.named_parameters() if any(nd in n for nd in no_decay)],
				'weight_decay': 0.0,
			}
		]

		optimizer_grouped_parameters += [
			{
				'params'      : [p for n, p in self.reasoner.named_parameters() if not any(nd in n for nd in no_decay)],
				'weight_decay': self.p.weight_decay,
			},
			{
				'params'      : [p for n, p in self.reasoner.named_parameters() if any(nd in n for nd in no_decay)],
				'weight_decay': 0.0,
			}
		]

		if self.p.optimizer == 'adamw':
			optimizer = AdamW(optimizer_grouped_parameters, lr=self.p.learning_rate)
		else:
			raise NotImplementedError

		if self.p.lr_scheduler == 'linear_with_warmup':
			if self.p.warmup_updates > 1.0:
				warmup_steps = int(self.p.warmup_updates)
			else:
				warmup_steps = int(self.total_steps * self.p.warmup_updates)
			print(f'\nTotal steps: {self.total_steps} with warmup steps: {warmup_steps}\n')

			scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=self.total_steps)
			scheduler = {
				'scheduler': scheduler,
				'interval': 'step',
				'frequency': 1
			}
		elif self.p.lr_scheduler == 'fixed':
			return [optimizer]
		else:
			raise NotImplementedError

		return [optimizer], [scheduler]


class InfiniteRecursionError(OverflowError):
    '''raise this when there's an infinite recursion possibility in proof generation'''

def get_verb(sent):
	if ' visits ' in sent:
		return 'visits'
	elif ' sees ' in sent:
		return 'sees'
	elif ' likes ' in sent:
		return 'likes'
	elif ' eats ' in sent:
		return 'eats'
	elif ' chases ' in sent:
		return 'chases'
	elif ' needs ' in sent:
		return 'needs'
	elif ' wants ' in sent:
		return 'wants'
	elif ' forgets ' in sent:
		return 'forgets'
	elif ' humiliates ' in sent:
		return 'humiliates'
	elif ' treats ' in sent:
		return 'treats'
	elif ' serves ' in sent:
		return 'serves'
	elif ' abandons ' in sent:
		return 'abandons'
	elif ' hates ' in sent:
		return 'hates'
	elif ' loves ' in sent:
		return 'loves'
	elif ' kills ' in sent:
		return 'kills'
	elif ' doubts ' in sent:
		return 'doubts'
	elif ' runs ' in sent:
		return 'runs'

def negate(sent):
	'''Generate the negation of a sentence using simple regex'''
	if ' is not ' in sent:
		# is not --> is
		sent = sent.replace('is not', 'is')
	elif ' is ' in sent:
		# is --> is not
		sent = sent.replace('is', 'is not')
	elif ' does not ' in sent:
		# does not visit --> visits
		# find the next word in the sentence after not, i.e., "... does not X ..."
		all_words = sent.split()
		next_word = all_words[all_words.index('not') + 1]
		new_word  = next_word + 's'
		sent      = sent.replace(f'does not {next_word}', new_word)
	else:
		# visits --> does not visit
		verb = get_verb(sent)
		new_verb = verb[:-1] # removes the s in the last place
		sent = sent.replace(verb, f'does not {new_verb}')

	return sent

def generate_proof(last_fact, proof_dict, last_rule=None):
	all_proofs = []
	for idx in range(len(proof_dict[last_fact])):
		facts, rule = proof_dict[last_fact][idx]

		# hack to handle an infinite recursion issue - this can happen if the last_fact equals one of the facts in the proof
		if last_fact in facts and rule != '':
			# If rule is equal to '' then it's expected to contain last_fact by design
			raise InfiniteRecursionError('Cycle in proof graph!')

		if rule == '':
			assert len(facts) == 1
			if last_rule is None:
				return [[(facts[0])]]
			else:
				return [[(facts[0], last_rule)]]

		else:
			if len(facts) == 1:
				proofs = generate_proof(facts[0], proof_dict, rule)
				if last_rule is not None:
					_ = [x.append((rule, last_rule)) for x in proofs]
				all_proofs.extend(proofs)

			elif len(facts) >= 2:
				intermediate_proofs = [generate_proof(facts[fact_idx], proof_dict, rule) for fact_idx in range(len(facts))]
				permuted = list(itertools.product(*intermediate_proofs))
				permuted = [list(itertools.chain.from_iterable(x)) for x in permuted]
				if last_rule is not None:
					_ = [x.append((rule, last_rule)) for x in permuted]
				all_proofs.extend(permuted)

	return all_proofs
