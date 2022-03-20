from helper import *


class BaseModel(pl.LightningModule):
	'''
	The main runner class that integrates Pytorch Lightning with the framework.
	'''

	def __init__(self, train_batch_size=16, max_epochs=5, gpus=1):
		super().__init__()

		self.p                  = types.SimpleNamespace()
		self.p.train_batch_size = train_batch_size
		self.p.max_epochs       = max_epochs
		self.p.gpus             = gpus

	def forward(self, batch):
		raise NotImplementedError

	def calc_loss(self, preds, targets):
		raise NotImplementedError

	def calc_acc(self, preds, targets):
		raise NotImplementedError

	def run_step(self, batch, split):
		outputs = self(batch)
		preds   = (outputs > 0).float().squeeze()
		targets = batch['all_lbls']
		loss    = self.calc_loss(outputs.squeeze(), targets)
		acc     = self.calc_acc(preds, targets)

		if split == 'train':
			self.log(f'train_loss_step', loss.item(), prog_bar=True)
			self.log(f'train_acc_step', acc.item(), prog_bar=True)
		else:
			self.log(f'{split}_loss_step', loss.item(), prog_bar=True, sync_dist=True)
			self.log(f'{split}_acc_step', acc.item(), prog_bar=True, sync_dist=True)

		return {'loss': loss, 'preds': preds, 'targets': targets}

	def aggregate_epoch(self, outputs, split):
		preds   = torch.cat([x['preds'] for x in outputs])
		targets = torch.cat([x['targets'] for x in outputs])
		loss    = torch.stack([x['loss'] for x in outputs]).mean()
		acc     = self.calc_acc(preds, targets)

		if split == 'train':
			self.log(f'train_loss_epoch', loss.item())
			self.log(f'train_acc_epoch', acc.item())
		else:
			self.log(f'{split}_loss_epoch', loss.item(), sync_dist=True)
			self.log(f'{split}_acc_epoch', acc.item(), sync_dist=True)

	def training_step(self, batch, batch_idx):

		return self.run_step(batch, 'train')

	def training_epoch_end(self, outputs):
		self.aggregate_epoch(outputs, 'train')

	def validation_step(self, batch, batch_idx):
		return self.run_step(batch, 'valid')

	def validation_epoch_end(self, outputs):
		self.aggregate_epoch(outputs, 'valid')

	def test_step(self, batch, batch_idx):
		return self.run_step(batch, 'test')

	def test_epoch_end(self, outputs):
		self.aggregate_epoch(outputs, 'test')

	def setup(self, stage):
		if stage == 'fit':
			# Get train dataloader
			train_loader = self.trainer.datamodule.train_dataloader()

			# Calculate total steps
			effective_batch_size = (self.p.train_batch_size * max(1, self.p.gpus) * self.p.accumulate_grad_batches)
			self.total_steps     = int((len(train_loader.dataset) // effective_batch_size) * float(self.p.max_epochs))

			print('Total steps: ', self.total_steps)

	def configure_optimizers(self):
		raise NotImplementedError

	@staticmethod
	def add_model_specific_args(parent_parser):
		parser = ArgumentParser(parents=[parent_parser], add_help=False)
		parser.add_argument('--optimizer', 					default='adamw', 					type=str)
		parser.add_argument('--model', 						default='', 						type=str)
		parser.add_argument('--arch', 						default='',							type=str)
		parser.add_argument('--hf_name', 					default='',							type=str)
		parser.add_argument('--padding', 					default=0,							type=int)

		parser.add_argument('--learning_rate', 				default=1e-5, 						type=float)
		parser.add_argument('--adam_epsilon', 				default=1e-8, 						type=float)
		parser.add_argument('--warmup_updates', 			default=0.0, 						type=float)
		parser.add_argument('--weight_decay', 				default=0.0, 						type=float)
		parser.add_argument('--lr_scheduler',				default='linear_with_warmup',		type=str)
		parser.add_argument('--freeze_epochs',				default=-1,							type=int)
		parser.add_argument("--train_batch_size",			default=16,							type=int)
		parser.add_argument("--eval_batch_size", 			default=16, 						type=int)

		# FaiRR inference specific params
		parser.add_argument('--ruleselector_ckpt', 			default='', 						type=str,)
		parser.add_argument('--factselector_ckpt', 			default='', 						type=str,)
		parser.add_argument('--reasoner_ckpt', 				default='', 						type=str,)

		# FaiRR ruleselector specific params
		parser.add_argument('--cls_dropout', 				default=0.1, 						type=float,)

		return parser
