'''
This script is the rule selector component of FaiRR. It selects a rule from a given theory and statement.
'''

from helper import *
from basemodel import BaseModel


class FaiRRRuleSelector(BaseModel):
	def __init__(self, arch='roberta_large', train_batch_size=16, eval_batch_size=16, accumulate_grad_batches=1, learning_rate=1e-5, max_epochs=5,\
					optimizer='adamw', adam_epsilon=1e-8, weight_decay=0.0, lr_scheduler='linear_with_warmup', warmup_updates=0.0, freeze_epochs=-1, gpus=1,\
					hf_name='roberta-large', cls_dropout=0.1):
		super().__init__(train_batch_size=train_batch_size, max_epochs=max_epochs, gpus=gpus)
		self.save_hyperparameters()
		assert arch == 'roberta_large'

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
		self.p.cls_dropout             = cls_dropout

		self.text_encoder    = AutoModel.from_pretrained(hf_name)
		self.tokenizer       = AutoTokenizer.from_pretrained(hf_name)
		out_dim              = self.text_encoder.config.hidden_size
		self.out_dim         = out_dim
		self.classifier      = nn.Linear(out_dim, 1)
		self.dropout         = torch.nn.Dropout(self.p.cls_dropout)

		self.initialize()

	def initialize(self):
		xavier_normal_(self.classifier.weight)
		self.classifier.bias.data.zero_()

	def forward(self, input_ids, attn_mask):
		last_hidden_state = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)['last_hidden_state'] #shape (batchsize, seqlen, hiddensize)
		last_hidden_state = self.dropout(last_hidden_state)
		logits            = self.classifier(last_hidden_state).squeeze()

		return logits

	def predict(self, input_ids, token_mask, attn_mask):
		device  = input_ids.device
		outputs = self(input_ids, attn_mask)
		logits  = outputs

		# First filter out the logits corresponding to the valid tokens
		mask_len          = token_mask.sum(1) # (batchsize) eg [8,3,2,1]
		mask_nonzero      = torch.nonzero(token_mask) # (z, 2) size tensor, having x, y coordinates of non zero elements. z = no. of non zero elements
		y_indices         = torch.cat([torch.arange(x) for x in mask_len]).to(device)
		x_indices         = mask_nonzero[:, 0]
		filtered_logits   = torch.full((input_ids.shape[0], mask_len.max()), -1000.0).to(device)
		filtered_logits[x_indices, y_indices] = torch.masked_select(logits, token_mask.bool())

		# Then compute the predictions for each of the logit
		argmax_filtered_logits	= torch.argmax(filtered_logits, dim=1)
		preds 					= (F.one_hot(argmax_filtered_logits, num_classes=filtered_logits.shape[1])).int()

		# truncating preds to remove the cls token predictions (prioritize rule selection)
		preds = preds[:, 1:]

		# Finally, save a padded rule matrix with indices of the rules and the corresponding mask
		pred_mask_lengths = preds.sum(1)
		pred_mask_nonzero = torch.nonzero(preds)
		y_indices         = torch.cat([torch.arange(x) for x in pred_mask_lengths]).to(device)
		x_indices         = pred_mask_nonzero[:, 0]
		filtered_rule_ids = torch.full((input_ids.shape[0], pred_mask_lengths.max()), -1).to(device)
		filtered_rule_ids[x_indices, y_indices] = pred_mask_nonzero[:, 1]
		filtered_mask     = (filtered_rule_ids != -1)

		# Make the -1's -> 0 so that we can select some rule. Given the mask we can always prune this later
		# This step is non-intuitive here. To understand this, we need to consider the for loop in the main decoding logic where the rule_ids are used.
		filtered_rule_ids[~filtered_mask] = 0

		# filtered_rule_ids of size (b*maxrule_ids)
		return filtered_rule_ids, filtered_mask

	def calc_loss(self, outputs, targets, token_mask):
		# all rows of target are one hot, i.e., there is only 1 rule that needs to be selected
		assert torch.all(torch.sum(targets * token_mask, dim=1) == torch.ones(targets.shape[0]).to(targets.device))

		exp_logits = torch.exp(outputs)
		assert exp_logits.shape == token_mask.shape

		masked_exp_logits = exp_logits * token_mask
		norm_masked_exp_logits = masked_exp_logits/torch.sum(masked_exp_logits, dim=1).unsqueeze(-1)

		# convert the 0's to 1 in norm_masked_exp_logits so that log makes it 0
		# can be done by setting those indexes in norm_masked_exp_logits to 1., where token_mask = 0
		zeros_mask = (1 - token_mask).bool() # zeros_mask is 0 for SEP/CLS token and 1 everywhere else
		norm_masked_exp_logits[zeros_mask] = 1. # setting those indices of norm_mask = 1. where zeros_mask = 1

		# handling log(0) --> log(small_value) for places where token_mask is 1 and norm_masked_exp_logits is 0
		zeros_mask_ = (norm_masked_exp_logits == 0)
		norm_masked_exp_logits[zeros_mask_] = 1e-8

		logvals = torch.log(norm_masked_exp_logits)
		loss_reduced = F.nll_loss(logvals, torch.nonzero(targets)[:, 1], reduction='mean')

		return loss_reduced

	def calc_acc(self, preds, targets, token_mask):
		acc_not_reduced = (preds == targets).float()
		acc_masked      = torch.mul(acc_not_reduced, token_mask)
		acc_reduced     = acc_masked.sum()/token_mask.sum()
		acc             = 100 * acc_reduced

		return acc

	def calc_F1(self, preds, targets, token_mask):
		'''calculates the binary F1 score between preds and targets, with positive class being 1'''
		assert preds.shape == targets.shape
		assert preds.shape == token_mask.shape

		# get only the relevant indices of preds and targets, ie those which are non zero in token_mask
		mask           = (token_mask == 1)
		preds_masked   = torch.masked_select(preds, mask).cpu()
		targets_masked = torch.masked_select(targets, mask).cpu()

		binary_f1_class1 = f1_score(y_true=targets_masked, y_pred=preds_masked, pos_label=1, average='binary')
		binary_f1_class0 = f1_score(y_true=targets_masked, y_pred=preds_masked, pos_label=0, average='binary')
		macro_f1         = f1_score(y_true=targets_masked, y_pred=preds_masked, average='macro')
		micro_f1         = f1_score(y_true=targets_masked, y_pred=preds_masked, average='micro')

		return {'f1_class1':binary_f1_class1, 'f1_class0':binary_f1_class0, 'macro_f1':macro_f1, 'micro_f1':micro_f1}

	def calc_perf_metrics(self, preds, targets, token_mask):
		acc       = self.calc_acc(preds, targets, token_mask)
		F1_scores = self.calc_F1(preds, targets, token_mask)

		return {'acc':acc, 'f1_class1':F1_scores['f1_class1'], 'f1_class0':F1_scores['f1_class0'], 'macro_f1':F1_scores['macro_f1'], 'micro_f1':F1_scores['micro_f1']}

	def run_step(self, batch, split):
		outputs    = self(batch['all_sents'], batch['attn_mask'])
		token_mask = batch['all_token_mask']
		targets    = batch['all_token_labels']

		relevant_outputs        = outputs * token_mask
		argmax_relevant_outputs = torch.argmax(relevant_outputs, dim=1)
		loss                    = self.calc_loss(outputs.squeeze(), targets.squeeze(), token_mask.squeeze())
		preds                   = (F.one_hot(argmax_relevant_outputs, num_classes=outputs.shape[1])).int()
		perf_metrics            = self.calc_perf_metrics(preds.squeeze(), targets.squeeze(), token_mask.squeeze())

		if split == 'train':
			self.log(f'train_loss_step', loss.item(), prog_bar=True)
			for metric in perf_metrics.keys():
				self.log(f'train_{metric}_step', perf_metrics[metric], prog_bar=True)
		else:
			self.log(f'{split}_loss_step', loss.item(), prog_bar=True, sync_dist=True)
			for metric in perf_metrics.keys():
				self.log(f'{split}_{metric}_step', perf_metrics[metric], prog_bar=True)

		return {'loss': loss, 'preds': preds, 'targets': targets, 'token_mask': token_mask}

	def aggregate_epoch(self, outputs, split):
		preds        = torch.cat([x['preds'].reshape(-1) for x in outputs])
		targets      = torch.cat([x['targets'].reshape(-1) for x in outputs])
		token_mask   = torch.cat([x['token_mask'].reshape(-1) for x in outputs])
		loss         = torch.stack([x['loss'] for x in outputs]).mean()
		perf_metrics = self.calc_perf_metrics(preds.squeeze(), targets.squeeze(), token_mask.squeeze())

		if split == 'train':
			self.log(f'train_loss_epoch', loss.item())
			for metric in perf_metrics.keys():
				self.log(f'train_{metric}_epoch', perf_metrics[metric], prog_bar=True)
		elif split == 'valid':
			self.log(f'valid_loss_epoch', loss.item(), sync_dist=True)
			for metric in perf_metrics.keys():
				self.log(f'valid_{metric}_epoch', perf_metrics[metric], prog_bar=True)
		elif split == 'test':
			self.log(f'test_loss_epoch', loss.item(), sync_dist=True)
			for metric in perf_metrics.keys():
				self.log(f'test_{metric}_epoch', perf_metrics[metric], prog_bar=True)
			self.predictions = torch.stack((preds, targets), dim=1)
			print('predictions tensor in ruletaker class, shape = {}'.format(self.predictions.shape))

	def configure_optimizers(self):
		no_decay = ['bias', 'LayerNorm.weight']
		optimizer_grouped_parameters = [
			{
				'params'      : [p for n, p in self.text_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
				'weight_decay': self.p.weight_decay,
			},
			{
				'params'      : [p for n, p in self.text_encoder.named_parameters() if any(nd in n for nd in no_decay)],
				'weight_decay': 0.0,
			}
		]

		optimizer_grouped_parameters += [
			{
				'params'      : [p for n, p in self.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
				'weight_decay': self.p.weight_decay,
			},
			{
				'params'      : [p for n, p in self.classifier.named_parameters() if any(nd in n for nd in no_decay)],
				'weight_decay': 0.0,
			}
		]

		if self.p.optimizer == 'adamw':
			optimizer = AdamW(optimizer_grouped_parameters, lr=self.p.learning_rate, eps=self.p.adam_epsilon, betas=[0.9, 0.98])
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
