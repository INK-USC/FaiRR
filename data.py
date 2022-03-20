from helper import *


class DataModule(pl.LightningDataModule):

	def __init__(self, dataset, train_dataset, dev_dataset, test_dataset, arch, train_batch_size=32, eval_batch_size=32, num_workers=10, pad_idx=0):
		super().__init__()
		self.p                  = types.SimpleNamespace()
		self.p.dataset          = dataset
		self.p.train_dataset    = train_dataset		# used in load_dataset()
		self.p.dev_dataset      = dev_dataset		# used in load_dataset()
		self.p.test_dataset     = test_dataset		# used in load_dataset()
		self.p.arch             = arch
		self.p.train_batch_size = train_batch_size
		self.p.eval_batch_size  = eval_batch_size
		self.p.num_workers      = num_workers
		self.p.pad_idx          = pad_idx

	def load_dataset(self, split):
		if self.p.dataset.startswith('pw_'):
			if self.p.dataset.startswith('pw_leq'):
				world_assump = self.p.dataset.split('_')[3]
				maindir      = "_".join(self.p.dataset.split('_')[:3])
				subdir       = 'fairr_' + self.p.dataset.split('_', 4)[-1]  # can be fairr_rule, fairr_fact ...

			folder = f'../data/processed/{maindir}/{world_assump}/{subdir}/{self.p.arch}/{split}/'
			print(f'data being used from the folder = {folder}')
			dataset = ddict(list)
			if 'reasoner' in self.p.dataset:
				keys = ['input_ids', 'output_ids']
			else:
				keys = ['input_ids', 'token_labels', 'token_mask']
			for key in keys:
				with open(folder + f'{key}.pkl', 'rb') as f:
					dataset[key] = pickle.load(f)

		elif self.p.dataset.startswith('pwq_'):
			if self.p.dataset.startswith('pwq_leq'):
				world_assump = self.p.dataset.split('_')[-2]
				maindir      = "_".join(self.p.dataset.split('_')[:-2])
				subdir       = 'fairr_' + self.p.dataset.split('_')[-1]  # can be fairr_rule, fairr_fact ...

			folder = f'../data/processed/{maindir}/{world_assump}/{subdir}/{self.p.arch}/{split}/'
			print(f'data being used from the folder = {folder}')
			dataset = ddict(list)
			if 'reasoner' in self.p.dataset:
				keys = ['input_ids', 'output_ids']
			else:
				keys = ['input_ids', 'token_labels', 'token_mask']
			for key in keys:
				with open(folder + f'{key}.pkl', 'rb') as f:
					print(folder + f'{key}.pkl')
					dataset[key] = pickle.load(f)

		elif self.p.dataset.startswith('pwu_') or self.p.dataset.startswith('pwur_'):
			if self.p.dataset.startswith('pwu_leq_') or self.p.dataset.startswith('pwur_leq_'):
				if self.p.dataset.startswith('pwur_'):
					if '_eq_' in self.p.dataset:
						maindir      = "_".join(self.p.dataset.split('_')[:6])
						world_assump = self.p.dataset.split('_')[6]
					else:
						maindir      = "_".join(self.p.dataset.split('_')[:4])
						world_assump = self.p.dataset.split('_')[4]
				elif self.p.dataset.startswith('pwu_'):
					if '_eq_' in self.p.dataset:
						maindir      = "_".join(self.p.dataset.split('_')[:5])
						world_assump = self.p.dataset.split('_')[5]
					else:
						maindir      = "_".join(self.p.dataset.split('_')[:3])
						world_assump = self.p.dataset.split('_')[3]
				else:
					import pdb; pdb.set_trace()

				folder       = f'../data/processed/{maindir}/{world_assump}/{split}/'
				print(f'data being used from the folder = {folder}')
				dataset = ddict(list)
				keys = ['facts', 'rules', 'ques', 'answer', 'proof', 'equiv_id', 'qdep']
				for key in keys:
					with open(folder + f'{key}.pkl', 'rb') as f:
						dataset[key] = pickle.load(f)

		return dataset

	def setup(self, splits='all'):
		self.data = ddict(list)
		if splits == 'all':
			splits = ['train', 'dev', 'test']

		for split in splits:
			if self.p.dataset.startswith('pw_') and self.p.dataset.endswith('reasoner'):
				self.data[split] = FaiRRReasonerDataset(self.load_dataset(split), self.p.pad_idx)

			elif self.p.dataset.startswith('pwu_') or self.p.dataset.startswith('pwur_'):
				self.data[split] = FaiRRInferenceDataset(self.load_dataset(split))

			elif self.p.dataset.startswith('pwq_') and self.p.dataset.endswith('rule'):
				self.data[split] = FaiRRRuleSelectorDataset(self.load_dataset(split), self.p.pad_idx)

			elif self.p.dataset.startswith('pwq_') and self.p.dataset.endswith('fact'):
				self.data[split] = FaiRRFactSelectorDataset(self.load_dataset(split), self.p.pad_idx)

	def train_dataloader(self, shuffle=True):
		return DataLoader(
					self.data['train'],
					batch_size=self.p.train_batch_size,
					num_workers=self.p.num_workers,
					collate_fn=self.data['train'].collater,
					shuffle=shuffle,
				)

	def val_dataloader(self):
		return DataLoader(
					self.data['dev'],
					batch_size=self.p.eval_batch_size,
					num_workers=self.p.num_workers,
					collate_fn=self.data['dev'].collater,
				)

	def test_dataloader(self):
		return DataLoader(
					self.data['test'],
					batch_size=self.p.eval_batch_size,
					num_workers=self.p.num_workers,
					collate_fn=self.data['test'].collater,
				)

	@staticmethod
	def add_data_specific_args(parent_parser):
		parser = ArgumentParser(parents=[parent_parser], add_help=False)
		parser.add_argument("--dataset", 		 				type=str)
		parser.add_argument("--train_dataset",	default='', 	type=str)
		parser.add_argument("--dev_dataset",	default='', 	type=str)
		parser.add_argument("--test_dataset",	default='', 	type=str)
		parser.add_argument("--num_workers", 	default=10, 	type=int)
		return parser


class FaiRRRuleSelectorDataset(Dataset):

	def __init__(self, dataset, pad_idx):
		self.data    = dataset
		self.pad_idx = pad_idx

	def __len__(self):
		return len(self.data['token_labels']) # total number of datapoints

	def __getitem__(self, idx):
		item = {
			'sent'        : torch.LongTensor(self.data['input_ids'][idx]),
			'token_labels': torch.FloatTensor([self.data['token_labels'][idx]]),
			'token_mask'  : torch.FloatTensor([self.data['token_mask'][idx]]),
		}

		item['token_mask'][0][0] = 1.
		if(item['token_labels'].sum()==0):
			item['token_labels'][0][0] = 1.

		return item

	def collater(self, items):
		# note for sent, the pad value can be 1/0 based on roberta/bert
		# for token labels and token masks, the pad value must be 0
		all_sents = pad_sequence([x['sent'] for x in items], batch_first=True, padding_value=self.pad_idx)
		batch = {
			'all_sents'       : all_sents,
			'all_token_labels': pad_sequence([x['token_labels'].squeeze() for x in items], batch_first=True, padding_value=0),
			'all_token_mask'  : pad_sequence([x['token_mask'].squeeze() for x in items], batch_first=True, padding_value=0),
			'attn_mask'       : (all_sents != self.pad_idx).long(),
		}

		return batch

class FaiRRFactSelectorDataset(Dataset):

	def __init__(self, dataset, pad_idx):
		self.data    = dataset
		self.pad_idx = pad_idx

	def __len__(self):
		return len(self.data['token_labels'])

	def __getitem__(self, idx):
		item = {
			'sent'    : torch.LongTensor(self.data['input_ids'][idx]),
			'token_labels'     : torch.FloatTensor([self.data['token_labels'][idx]]),
			'token_mask' : torch.FloatTensor([self.data['token_mask'][idx]]),
		}
		return item

	def collater(self, items):
		all_sents = pad_sequence([x['sent'] for x in items], batch_first=True, padding_value=self.pad_idx)
		batch = {
			'all_sents'       : all_sents,
			'all_token_labels': pad_sequence([x['token_labels'].squeeze() for x in items], batch_first=True, padding_value=0),
			'all_token_mask'  : pad_sequence([x['token_mask'].squeeze() for x in items], batch_first=True, padding_value=0),
			'attn_mask'       : (all_sents != self.pad_idx).long(),
		}

		return batch

class FaiRRReasonerDataset(Dataset):

	def __init__(self, dataset, pad_idx):
		self.data    = dataset
		self.pad_idx = pad_idx

	def __len__(self):
		return len(self.data['input_ids'])

	def __getitem__(self, idx):
		item = {
			'input'   : torch.LongTensor(self.data['input_ids'][idx]),
			'output'  : torch.LongTensor(self.data['output_ids'][idx]),
		}

		return item

	def collater(self, items):
		all_inps        = pad_sequence([x['input'] for x in items], batch_first=True, padding_value=self.pad_idx)
		all_outs        = pad_sequence([x['output'] for x in items], batch_first=True, padding_value=self.pad_idx)
		y_ids           = all_outs[:, :-1].contiguous()
		labels          = all_outs[:, 1:].clone()
		labels[all_outs[:, 1:] == self.pad_idx] = -100

		batch = {
			'all_inps' : all_inps,
			'all_outs' : all_outs,
			'attn_mask': (all_inps != self.pad_idx).long(),
			'y_ids'    : y_ids,
			'labels'   : labels,
		}

		return batch

class FaiRRInferenceDataset(Dataset):

	def __init__(self, dataset):
		self.data = dataset

	def __len__(self):
		return len(self.data['answer'])

	def __getitem__(self, idx):
		item = {
			'facts'          : self.data['facts'][idx],
			'rules'          : self.data['rules'][idx],
			'ques'           : self.data['ques'][idx],
			'answer'         : self.data['answer'][idx],
			'proof'          : self.data['proof'][idx],
			'equiv_id'       : self.data['equiv_id'][idx],
			'qdep'           : self.data['qdep'][idx],
		}

		return item

	def collater(self, items):
		all_facts       = [x['facts'] for x in items]
		all_rules       = [x['rules'] for x in items]
		all_ques        = [x['ques'] for x in items]
		all_answer      = [x['answer'] for x in items]
		all_proof       = [x['proof'] for x in items]
		all_equiv_id    = [x['equiv_id'] for x in items]
		all_qdep        = [x['qdep'] for x in items]

		batch = {
			'all_facts'      : all_facts,
			'all_rules'      : all_rules,
			'all_ques'       : all_ques,
			'all_answer'     : all_answer,
			'all_proof'      : all_proof,
			'all_equiv_id'   : all_equiv_id,
			'all_qdep'       : all_qdep,
		}

		return batch
