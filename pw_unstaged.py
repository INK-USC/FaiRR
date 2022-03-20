from helper import *
from pw_helper import *


class PWInstance:

	def __init__(self, rules, facts, ques, answer, proofs, strategy, qdep, equiv_id):
		self.rules           = rules
		self.facts           = facts
		self.ques            = ques
		self.answer          = answer
		self.proofs          = proofs
		self.strategy        = strategy
		self.qdep            = qdep
		self.equiv_id        = equiv_id

	@classmethod
	def from_json(cls, json_dict, dep=None, lowercase=True):
		'''
		parses the question, answer, proofs, and the theory.
		'''
		if lowercase:
			facts               = [x['text'].lower() for x in json_dict['triples'].values()]
			rules               = [x['text'].lower() for x in json_dict['rules'].values()]
			all_questions       = parse_all_questions(json_dict, inference_data=True, lowercase=True)
		else:
			facts               = [x['text'] for x in json_dict['triples'].values()]
			rules               = [x['text'] for x in json_dict['rules'].values()]
			all_questions       = parse_all_questions(json_dict, inference_data=True, lowercase=False)

		equiv_id = json_dict.get('equiv_id', 'None')

		instances = []
		for question in all_questions:
			ques, answer, strategy, proofs, qdep = question

			if dep is None:
				instances.append(PWInstance(rules, facts, ques, answer, proofs, strategy, qdep, equiv_id))

			elif (qdep == dep) and proofs != [['None']]:
				instances.append(PWInstance(rules, facts, ques, answer, proofs, strategy, qdep, equiv_id))

			elif dep == 100 and proofs == [['None']]: # dep 100 is for N/A
				instances.append(PWInstance(rules, facts, ques, answer, proofs, strategy, dep, equiv_id))

		return instances

class PWQRuleInstance:

	def __init__(self, rule_list, facts_para, ques, labels, strategy):
		self.rule_list  = rule_list
		self.facts_para = facts_para
		self.ques       = ques
		self.labels     = labels # labels for rules [0,1,0 ....] of length = no. of rules
		self.strategy   = strategy

	@classmethod
	def from_json(cls, non_stage_json_dict, stage_json_dicts):
		'''
		makes datapoints such as: "theory(rules+facts) + derived_facts(if any) + rule"
		json_dict = non staged json dict
		stage_json_dicts = list of json dicts taken from the corresponding staged files
		'''
		instances = []
		# each non_stage_json_dict ahs many questions to be answered True/False
		all_questions = parse_all_questions(non_stage_json_dict, inference_data=False) # list of the form [[q1, ans1, strategy1, proof1], []]
		for question in all_questions:
			instances_ques = [] # instances for this question
			ques, answer, strategy, proofs, qdep = question
			facts_para, fact_list, _ = get_facts(non_stage_json_dict)
			rules_para, rule_list, _, _, _ = get_rules(non_stage_json_dict)

			if strategy in ['proof', 'inv-proof']:
				# for these strategies, the proofs exist,
				# ie we can either prove the question to for proving it to be True, or, prove the negative of the question for proving it False
				# for eg "cow eats the lion" is the question and strategy is 'proof' then we have a direct proof. eg "cow does not eat the lion" is the question and strategy is 'inv-proof' then we have a direct proof for 'cow eats the lion',  which makes the question False

				found_proof = False # will keep track if we found a proof or not
				for proof in proofs: # Why are we doing this? -> see the similar code in fact selector and read the comments above that

					# get the intermediates and rules in the proof using regex
					proof_inters = re.findall(r'int[0-9]+', proof["representation"]) # [int1, int2, int1]
					proof_rules = re.findall(r'rule[0-9]+', proof["representation"]) # [rule1, rule4, rule1]
					# rule corresponding to an int should be the same, however many times the int is repeated in the proof, because each intermediate is made using 1 and only 1 rule. it shouldn't be like proof_inters = [int1, int2, int1], proof)rules = [rule1, rule4, rule3]
					# also even if a intermediate is repeated, same should happen with the corresponding rule

					proof_inters_text = [proof['intermediates'][proof_inter]['text'].lower() for proof_inter in proof_inters]

					try:
						for i in range(len(stage_json_dicts)):
							json_dict1 = stage_json_dicts[i]
							inferences1 = parse_all_inferences(json_dict1, return_text=True) # inferences is of the form {conclusion_text:(facts, fact_ids, rule, rule_id), .....]

							if(i<len(stage_json_dicts)-1):
								json_dict2 = stage_json_dicts[i+1]
								inferences2 = parse_all_inferences(json_dict2, return_text=True)

								# print(set(inferences1), inferences1)
								assert(len(inferences1) == len(set(inferences1)))
								assert(len(inferences2) == len(set(inferences2)))

								inference_differ = set(inferences1).difference(set(inferences2)) # difference of the sets inference1 and inference2
								# print(inference_differ)
								assert(len(inference_differ) == 1)
								inference_added = inference_differ.pop()

								if(inference_added in proof_inters_text):
									labels = np.zeros(len(rule_list))
									rule_id_for_inference = inferences1[inference_added][3]

									facts_for_inference = inferences1[inference_added][0] # list of fact texts for making the inference
									for f in facts_for_inference:
										assert f in fact_list # assert would hit and take this to the except state where some values will be reset and we will do the loop again over stage files
										labels[int(rule_id_for_inference[4:])-1] = 1 # ie if proof_rules[i] = rule18 then labels[17] = 1. note: rules start from 1 ie rule1, rule2, ...

									instances_ques.append(PWQRuleInstance(rule_list, facts_para, ques, labels.tolist(), strategy))
									facts_para = facts_para + ' ' + inference_added # update facts para
									assert inference_added not in fact_list # before adding it to the fact list, check if its not present already
									fact_list.append(inference_added) # update fact list

							else:
								assert (len(inferences1) == 0) #no. of infrences is 0 for the last json dict
								# make a datapoint where no rule is selected
								labels = np.zeros(len(rule_list))
								instances_ques.append(PWQRuleInstance(rule_list, facts_para, ques, labels.tolist(), strategy))

						found_proof = True
						instances.extend(instances_ques)
						break
						# if this part of code is successful, break it, ie we are working with the current proof
						# (its the first proof out of the list of proofs that works for us)

					except Exception as e:
						# reset the values of the following
						facts_para, fact_list, _ = get_facts(non_stage_json_dict)
						instances_ques = []
						continue # to the next proof since this didnot work

			else:
				# all the other proof strategies are based on whether the question (or its negated form) can be generated or not.
				# the proof for all these is None.
				# make a datapoint where no rule is selected
				labels = np.zeros(len(rule_list))
				instances.append(PWQRuleInstance(rule_list, facts_para, ques, labels.tolist(), strategy))


		return instances

	def tokenize_ptlm(self, tokenizer):
		# convert the data in the format expected by the PTLM
		# format: [CLS]question[SEP]factspara[SEP]rule1text[SEP]rule2text[SEP].....[SEP]
		input_tokens = tokenizer.cls_token + self.ques + tokenizer.sep_token + self.facts_para + tokenizer.sep_token
		for ruletext in self.rule_list:
			input_tokens = input_tokens + ruletext + tokenizer.sep_token

		input_tokens_tokenized = tokenizer.tokenize(input_tokens)
		input_ids  = tokenizer.convert_tokens_to_ids(input_tokens_tokenized)
		token_mask = [1 if token == tokenizer.sep_token else 0 for token in input_tokens_tokenized] # list of 0s and 1s with 1s at positions of all sep tokens.

		sep_token_indices = [i for i in range(len(token_mask)) if token_mask[i] == 1]
		token_mask[sep_token_indices[-1]], token_mask[sep_token_indices[0]] = 0, 0  # since the first and last sep token doesnot correspond to any rule
		sep_token_indices = sep_token_indices[1:-1]
		token_labels = np.zeros(len(token_mask))
		assert len(self.labels) == len(sep_token_indices)

		token_labels[sep_token_indices] = self.labels
		return input_ids, token_labels.tolist(), token_mask

	def tokenize(self, tokenizer, arch, split):
		return self.tokenize_ptlm(tokenizer)

	@classmethod
	def tokenize_batch(cls, tokenizer, batched_rules, batched_facts, batched_ques):
		new_rules        = [map(str.lower, rules) for rules in batched_rules]
		new_facts        = [map(str.lower, facts) for facts in batched_facts]
		new_ques         = [ques.lower() for ques in batched_ques]
		input_tokens     = [ques + tokenizer.sep_token + ' '.join(facts) + tokenizer.sep_token + tokenizer.sep_token.join(rules) for ques,facts,rules in zip(new_ques, new_facts, new_rules)]
		tokenized        = tokenizer(input_tokens, add_special_tokens=True, padding=True, truncation=True, return_tensors='pt', return_special_tokens_mask=True)
		input_ids        = tokenized['input_ids']
		attn_mask        = tokenized['attention_mask']

		# create dummy input tokens to quickly identify the first occurrence of tokenizer.sep_token that's present after ques
		dummy_input_tokens     = [ques + tokenizer.sep_token for ques in new_ques]
		dummy_tokenized        = tokenizer(dummy_input_tokens, add_special_tokens=True, padding='max_length', max_length=input_ids.shape[1], truncation=True, return_tensors='pt', return_special_tokens_mask=True)
		dummy_input_ids        = dummy_tokenized['input_ids']
		first_sep_token_mask   = (dummy_input_ids == tokenizer.sep_token_id) * (~dummy_tokenized['special_tokens_mask'].bool())

		token_mask       = (input_ids == tokenizer.sep_token_id)
		sep_mask         = first_sep_token_mask		# add the first sep token - we want that to be False as well in token_mask

		sep_mask += (tokenized['special_tokens_mask'] * token_mask).bool()
		token_mask[sep_mask.bool()] = False
		token_mask[:, 0] = 1

		return input_ids, attn_mask, token_mask


class PWQFactInstance:

	def __init__(self, rule, fact_list, ques, labels):
		self.rule      = rule
		self.fact_list = fact_list
		self.ques      = ques
		self.labels    = labels # labels for facts [0,1,0 ....] of length = no. of rules

	@classmethod
	def from_json(cls, non_stage_json_dict, stage_json_dicts):
		'''
		makes datapoints such as: "theory(rules+facts) + derived_facts(if any) + rule"
		json_dict = non staged json dict
		stage_json_dicts = list of json dicts taken from the corresponding staged files
		'''
		instances = []
		# each non_stage_json_dict ahs many questions to be answered True/False
		all_questions = parse_all_questions(non_stage_json_dict, inference_data=False) # list of the form [[q1, ans1, strategy1, proof1], []]
		for question in all_questions:
			instances_ques = [] # instances for this question
			ques, answer, strategy, proofs, qdep = question
			_, fact_list, _ = get_facts(non_stage_json_dict)

			fact2num = {fact_list[k]: k+1 for k in range(len(fact_list))} # {fact1_text:1, fact2_text:2, ......}
			num_facts = len(fact_list)

			if strategy in ['proof', 'inv-proof']:
				# for these strategies, the proofs exist,
				# ie we can either prove the question to for proving it to be True, or, prove the negative of the question for proving it False
				# for eg "cow eats the lion" is the question and strategy is 'proof' then we have a direct proof. eg "cow does not eat the lion" is the question and strategy is 'inv-proof' then we have a direct proof for 'cow eats the lion',  which makes the question False


				# iterating over the proofs because, we are selecting facts one by one from as per the staged files
				# this means that if there are 2 proofs like 1.]((((triple2) -> (rule6 % A-int2)) triple2) -> (rule1 % int1))
				# 											2.]((((triple7) -> (rule6 % B-int2)) triple7) -> (rule1 % int1))
				# ie the conclusions are same (int1) but the intermediates are different (A-int2 and B-int2)
				# Now, the stage -add0 file will have 1 step conclusions A-int2 and B-int2. But only one of them is sent to -add1 file
				# so if B-int2 is sent to the add 1 file, then we have to choose triple7 and NOT triple2
				# if we select the first proof (having A-int2) but the stage file sends B-int2 to the theory, we would not send it to the theory, since it does not belong to the proof 1 (having A-int2)
				# but then in the next step (int1) would require (B-int2) to be proved, which would not be present in the facts due to the above reason
				# Hence we iterate over the proofs and choose the one which works for us

				found_proof = False # will keep track if we found a proof or not
				for proof in proofs:
					# get the intermediates and rules in the proof using regex
					proof_inters = re.findall(r'int[0-9]{1,2}', proof["representation"]) # [int1, int2, int1]
					proof_facts = re.findall(r'triple[0-9]{1,2}', proof["representation"])
					# proof_facts should never be empty! atleast one triple exists for the proof
					assert len(proof_facts) >=1
					proof_inters_text = [proof['intermediates'][proof_inter]['text'].lower() for proof_inter in proof_inters]

					try:
						for i in range(len(stage_json_dicts)):
							json_dict1 = stage_json_dicts[i]
							inferences1 = parse_all_inferences(json_dict1, return_text=True) # inferences is of the form {conclusion_text:(facts, fact_ids, rule, rule_id), .....]

							if(i<len(stage_json_dicts)-1):
								json_dict2 = stage_json_dicts[i+1]
								inferences2 = parse_all_inferences(json_dict2, return_text=True)

								# print(set(inferences1), inferences1)
								assert(len(inferences1) == len(set(inferences1)))
								assert(len(inferences2) == len(set(inferences2)))

								inference_differ = set(inferences1).difference(set(inferences2)) # difference of the sets inference1 and inference2.
								# NOTEL set of a dictionary in python only has keys, hence the instance_differ will only have the key (conclusion_text) corresponding to the inference added in the theory
								assert(len(inference_differ) == 1)
								inference_added = inference_differ.pop()


								if(inference_added in proof_inters_text):
									labels = np.zeros(num_facts)
									rule_text_for_inference = inferences1[inference_added][2]
									facts_for_inference = inferences1[inference_added][0] # list of fact texts for making the inference
									for f in facts_for_inference:
										assert f in fact_list # assert would hit and take this to the except state where some values will be reset and we will do the loop again over stage files
										labels[fact2num[f]-1] = 1 # ie if proof_facts[i] = triple18 then labels[17] = 1. note: triples start from 1 ie triple1, triple2, ...

									instances_ques.append(PWQFactInstance(rule_text_for_inference, list(fact_list), ques, labels.tolist()))
									# add the fact to the fact_list, fact2num and increment the number of facts
									fact_list.append(inference_added)

									assert inference_added not in fact2num.keys()
									fact2num[inference_added] = num_facts+1
									num_facts+=1
									assert len(fact_list) == num_facts

						found_proof = True
						instances.extend(instances_ques)
						break
						# if this part of code is successful, break it, ie we are working with the current proof
						# (its the first proof out of the list of proofs that works for us)

					except Exception as e:
						# reset the values of the following
						_, fact_list, _ = get_facts(non_stage_json_dict)
						fact2num = {fact_list[k]: k+1 for k in range(len(fact_list))} # {fact1_text:1, fact2_text:2, ......}
						num_facts = len(fact_list)
						instances_ques = []
						continue # to the next proof since this didnot work

				assert (found_proof == True)

		return instances

	def tokenize_ptlm(self, tokenizer):
		# convert the data in the format expected by the PTLM
		# format: [CLS]question[SEP]rule[SEP]fact1text[SEP]fact2text[SEP].....[SEP]

		input_tokens = tokenizer.cls_token + self.ques + tokenizer.sep_token + self.rule + tokenizer.sep_token
		for facttext in self.fact_list:
			input_tokens = input_tokens + facttext + tokenizer.sep_token
		input_tokens_tokenized = tokenizer.tokenize(input_tokens)
		input_ids    = tokenizer.convert_tokens_to_ids(input_tokens_tokenized)
		token_mask = [1 if token == tokenizer.sep_token else 0 for token in input_tokens_tokenized] # list of 0s and 1s with 1s at positions of all sep tokens.

		sep_token_indices = [i for i in range(len(token_mask)) if token_mask[i] == 1]
		token_mask[sep_token_indices[-1]], token_mask[sep_token_indices[0]] = 0, 0  # since the first and last sep token doesnot correspond to any fact
		sep_token_indices = sep_token_indices[1:-1] # we don't want the first and last sep token which donot correspond to any fact
		token_labels = np.zeros(len(token_mask))

		assert len(self.labels) == len(sep_token_indices)

		token_labels[sep_token_indices] = self.labels
		return input_ids, token_labels.tolist(), token_mask

	def tokenize(self, tokenizer, arch, split):
		return self.tokenize_ptlm(tokenizer)

	@classmethod
	def tokenize_batch(cls, tokenizer, batched_rules, batched_facts, batched_ques):
		new_rules        = [rule.lower() for rule in batched_rules]
		new_facts        = [map(str.lower, facts) for facts in batched_facts]
		new_ques         = [ques.lower() for ques in batched_ques]
		input_tokens     = [ques + tokenizer.sep_token + rule + tokenizer.sep_token + tokenizer.sep_token.join(facts) for ques,facts,rule in zip(new_ques, new_facts, new_rules)]

		tokenized        = tokenizer(input_tokens, add_special_tokens=True, padding=True, truncation=True, return_tensors='pt', return_special_tokens_mask=True)
		input_ids        = tokenized['input_ids']
		attn_mask        = tokenized['attention_mask']

		# create dummy input tokens to quickly identify the first occurrence of tokenizer.sep_token that's present after ques
		dummy_input_tokens     = [ques + tokenizer.sep_token for ques in new_ques]
		dummy_tokenized        = tokenizer(dummy_input_tokens, add_special_tokens=True, padding='max_length', max_length=input_ids.shape[1], truncation=True, return_tensors='pt', return_special_tokens_mask=True)
		dummy_input_ids        = dummy_tokenized['input_ids']
		first_sep_token_mask   = (dummy_input_ids == tokenizer.sep_token_id) * (~dummy_tokenized['special_tokens_mask'].bool())

		token_mask       = (input_ids == tokenizer.sep_token_id)
		sep_mask         = tokenized['special_tokens_mask'] * token_mask
		sep_mask         = sep_mask + first_sep_token_mask		# add the first sep token - we want that to be False as well in token_mask
		token_mask[sep_mask.bool()] = False

		return input_ids, attn_mask, token_mask
