'''
This script is the knowledge composer component of FaiRR. It takes the selected rule and facts and generates a new conclusion.
'''

from helper import *
from basemodel import BaseModel


class FaiRRReasoner(BaseModel):
	def __init__(self, arch='t5_base', train_batch_size=16, eval_batch_size=16, accumulate_grad_batches=1, learning_rate=1e-5, max_epochs=5,\
					optimizer='adamw', adam_epsilon=1e-8, weight_decay=0.0, lr_scheduler='linear_with_warmup', warmup_updates=0.0, freeze_epochs=-1, gpus=1,\
					hf_name='t5-base'):
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

		self.reasoner  = T5ForConditionalGeneration.from_pretrained(hf_name)
		self.tokenizer = AutoTokenizer.from_pretrained(hf_name)

	def forward(self, batch):
		outputs = self.reasoner(input_ids=batch['all_inps'], attention_mask=batch['attn_mask'], decoder_input_ids=batch['y_ids'], labels=batch['labels'])

		return outputs

	def predict(self, batch):
		max_length = batch['all_outs'].size(1)
		out        = self.reasoner.generate(batch['all_inps'], num_beams=1, min_length=1, max_length=max_length)
		preds      = torch.zeros_like(batch['all_outs'])
		preds[:, :out.shape[1]] = out

		return preds

	def decode(self, preds):
		return [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in preds]

	def predict_and_decode(self, input_ids):
		pred    = self.reasoner.generate(input_ids, max_length=128, num_beams=1, min_length=1)
		decoded = self.decode(pred)

		return decoded

	def calc_acc(self, preds, targets):
		return 100 * (preds == targets).float().mean()

	def run_step(self, batch, split):
		out     = self(batch)
		loss    = out.loss
		targets = batch['all_outs']
		preds   = self.predict(batch)
		acc     = self.calc_acc(preds, targets)

		if split == 'train':
			self.log(f'train_loss_step', loss.item(), prog_bar=True)
			self.log(f'train_acc_step', acc.item(), prog_bar=True)
		else:
			self.log(f'{split}_loss_step', loss.item(), prog_bar=True, sync_dist=True)
			self.log(f'{split}_acc_step', acc.item(), prog_bar=True, sync_dist=True)

		return {'loss': loss, 'acc': acc}

	def aggregate_epoch(self, outputs, split):
		loss = torch.stack([x['loss'] for x in outputs]).mean()
		acc  = torch.stack([x['acc'] for x in outputs]).mean()

		if split == 'train':
			self.log(f'train_loss_epoch', loss.item())
			self.log(f'train_acc_epoch', acc.item())
		elif split == 'valid':
			self.log(f'valid_loss_epoch', loss.item(), sync_dist=True)
			self.log(f'valid_acc_epoch', acc.item(), sync_dist=True)
		elif split == 'test':
			self.log(f'test_loss_epoch', loss.item(), sync_dist=True)
			self.log(f'test_acc_epoch', acc.item(), sync_dist=True)

	def configure_optimizers(self):
		no_decay = ['bias', 'LayerNorm.weight']
		optimizer_grouped_parameters = [
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
		elif self.p.optimizer == 'adafactor':
			optimizer = Adafactor(optimizer_grouped_parameters, scale_parameter=False, relative_step=False, warmup_init=False, clip_threshold=1.0, lr=self.p.learning_rate)
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
