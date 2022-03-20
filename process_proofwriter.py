'''
This is the main data processing script. For all datasets, this script tokenizes every input/output and saves as pickle files.
'''

from helper import *
from proofwriter_classes import *
import random
random.seed(10)

def get_pw_fname(args, subdir, split, is_staged=True, equivalence=False):
	if is_staged:
		fname = f'../data/raw/proofwriter/{args.world_assump}/{subdir}/meta-stage-{split}.jsonl'
	else:
		fname = f'../data/raw/proofwriter/{args.world_assump}/{subdir}/meta-{split}.jsonl'

	# change input filename for pararules
	if subdir == 'NatLang':
		fname = fname.replace('.jsonl', '-processed.jsonl')

	if equivalence:
		equiv_name = args.dataset.split('_')[-1]

		if equiv_name in ['name', 'attr', 'rel', 'name$attr']:
			if equiv_name == 'name':
				equiv_name = 'name$comm'
			if equiv_name == 'attr':
				equiv_name = 'attrs'
			fname = fname.replace('.jsonl', f'-equiv_{equiv_name}.jsonl')
		else:
			raise NotImplementedError

	return fname

def get_out_fname(args, split, key, is_staged=True, return_folder=False):
	if return_folder:
		if is_staged:
			return f'../data/processed/{args.dataset}/{args.world_assump}/{args.fairr_model}/{args.arch}/{split}'
		else:
			return f'../data/processed/{args.dataset}/{args.world_assump}/{split}'
	else:
		if is_staged:
			return f'../data/processed/{args.dataset}/{args.world_assump}/{args.fairr_model}/{args.arch}/{split}/{key}.pkl'
		else:
			return f'../data/processed/{args.dataset}/{args.world_assump}/{split}/{key}.pkl'

def get_pw_subdir(dataset):
	if ('_leq_' in dataset or '_eq_' in dataset): # handles, 'pw_leq_', 'pwu_leq_', 'pwq_leq_', 'pwur_leq_',
		subdir = f'depth-{dataset.split("_")[2]}'
	elif 'pararules' in dataset: # handles, 'pw_pararules', 'pwu_pararules', 'pwq_pararules'
		subdir = 'NatLang'

	return subdir

def get_keys(fairr_model):
	if fairr_model == 'fairr_rule' or fairr_model == 'fairr_fact':
		return ['input_ids', 'token_labels', 'token_mask']
	else:
		return ['input_ids', 'output_ids']

def is_valid_row(args, row_id):
	return (args.world_assump == 'OWA')

def make_data_from_instance(data, output, keys):
	for i, key in enumerate(keys):
		data[key].append(output[i])

def pickle_dump_file(keys, data, split, args, is_staged=True):
	print(args)
	for key in keys:
		print(f'Dumping {len(data[key])} lines for dataset: {args.dataset} split: {split} key: {key}')
		print(f'file writing to = {get_out_fname(args, split, key, is_staged=is_staged)}')
		with open(get_out_fname(args, split, key, is_staged=is_staged), 'wb') as f:
			pickle.dump(data[key], f)

def get_row_chunks(stagefile, non_stagefile, use_equiv_id=False):
	# returns a list of the form [[non-stagefile1, stagefile1-add0, stagefile1-add1 ...], [], []]
	print(f'stagefile = {stagefile}, non_stagefile = {non_stagefile}')

	if use_equiv_id:
		id_to_use = 'equiv_id'
	else:
		id_to_use = 'id'

	stagefile_lines = []
	non_stagefile_lines = []
	with jsonlines.open(stagefile) as f:
		for i, row in enumerate(tqdm(f)):
			stagefile_lines.append(row)
	with jsonlines.open(non_stagefile) as f:
		for i, row in enumerate(tqdm(f)):
			non_stagefile_lines.append(row)
	i, j = 0, 0

	if not use_equiv_id:
		row_chunks = []
		while(i<len(non_stagefile_lines)):
			chunk = []
			chunk.append(non_stagefile_lines[i])
			while(j<len(stagefile_lines) and stagefile_lines[j][id_to_use].rsplit('-', 1)[0] == non_stagefile_lines[i][id_to_use]): # ie the dicts should correspond to the same id except the -add0/1/2 part
				chunk.append(stagefile_lines[j])
				j+=1
			row_chunks.append(chunk)
			i+=1
	else:
		row_chunks = []
		while(i<len(non_stagefile_lines)):
			chunk = []
			chunk.append(non_stagefile_lines[i])
			while(j<len(stagefile_lines) and (stagefile_lines[j][id_to_use].rsplit('-', 1)[0] + '_' + stagefile_lines[j][id_to_use].split('_', 1)[1]) == non_stagefile_lines[i][id_to_use]): # ie the dicts should correspond to the same id except the -add0/1/2 part
				chunk.append(stagefile_lines[j])
				j+=1
			row_chunks.append(chunk)
			i+=1

	return row_chunks

def main(args):

	# load tokenizer
	if args.arch == 't5_base':
		tokenizer = AutoTokenizer.from_pretrained("t5-base")
	elif args.arch == 't5_large':
		tokenizer = AutoTokenizer.from_pretrained("t5-large")
	elif args.arch == 'roberta_large':
		tokenizer = AutoTokenizer.from_pretrained("roberta-large")
	else:
		print('Token type ids not implemented in tokenize call, will not work for bert models')
		raise NotImplementedError

	# check if the dataset is staged dataset or unstaged dataset
	if args.dataset.startswith('pwu_') or args.dataset.startswith('pwur_'):
		is_staged = False
	else:
		is_staged = True

	# load data
	for split in ['train', 'dev', 'test']:
		# pos -> some rule is selected, good_neg -> stop examples, bad_neg -> no rule possible to select
		pos_count, good_neg_count, bad_neg_count = 0, 0, 0

		print(f'Processing {split} split...')

		# make folder if not exists
		pathlib.Path(get_out_fname(args, split, None, is_staged=is_staged, return_folder=True)).mkdir(exist_ok=True, parents=True)

		data = ddict(list)

		if args.dataset.startswith('pw_leq'):
			# load the relevant original file and select all the data

			# we are using row chunks because we may have to sample 20 percent from D0 to 2, and for that we can't sample randomly from the rows of the stage json files
			# but rather we need to sample from the rows of the non - stage files, which is equivalent to saying that we can sample from row_chunks, which is a list of tuples
			# and each tuple corresponds to just one row (id) in the non - stage file
			if args.dataset == 'pw_leq_0to3':
				# combine depth 0, 1, 2 and sample 20%. add depth 3 to this to make the train/valid/test data
				subdirs          = [get_pw_subdir(f'pw_leq_{i}') for i in range(4)]
				stagefiles       = [get_pw_fname(args, subdir, split, is_staged=True) for subdir in subdirs]
				non_stagefiles   = [get_pw_fname(args, subdir, split, is_staged=False) for subdir in subdirs]
				row_chunks       = [get_row_chunks(stagefiles[i], non_stagefiles[i]) for i in range(len(subdirs))]
				row_chunks012    = []
				for i in range(3):
					row_chunks012.extend(row_chunks[i])
				row_chunks012 = random.sample(row_chunks012, len(row_chunks012)//5)
				print(f'Length of row_chunks for dataset row_chunks012 is = {len(row_chunks012)}')
				row_chunks012.extend(row_chunks[3])
				row_chunks = row_chunks012 # Has all elemennts of depth 3 stage file, along with 20 percent from depth 0 to 2. Format islist of lists [[non-stagefile1, stagefile1-add0, stagefile1-add1 ...], [], []]
				print(f'Length of row_chunks for dataset pw_leq_0to3 is = {len(row_chunks)}')

			else:
				subdir    	  = get_pw_subdir(args.dataset)
				stagefile     = get_pw_fname(args, subdir, split, is_staged=True)
				non_stagefile = get_pw_fname(args, subdir, split, is_staged=False)
				row_chunks    = get_row_chunks(stagefile, non_stagefile) # gives back a list of lists [[non-stagefile1, stagefile1-add0, stagefile1-add1 ...], [], []]

			if args.fairr_model == 'fairr_reasoner':
				for row_chunk in tqdm(row_chunks):
					non_stage_row = row_chunk[0] # row chunk is of the form [non-stagefile1, stagefile1-add0, stagefile1-add1 ...]
					stage_rows    = row_chunk[1:]
					if is_valid_row(args, non_stage_row['id']):
						# for i, row in enumerate(rows):
						for i, row in enumerate(stage_rows):
							instances = PWReasonerInstance.from_json(row)
							for instance in instances:
								output   = instance.tokenize(tokenizer, args.arch, split)
								make_data_from_instance(data, output, get_keys(args.fairr_model))

#########################################

		elif args.dataset.startswith('pwu_leq'):
			subdir = get_pw_subdir(args.dataset)
			if '_eq_' in args.dataset:
				qdep_required = int(args.dataset.split("_")[-1]) # the depth which we want
			else:
				qdep_required = None
			with jsonlines.open(get_pw_fname(args, subdir, split, is_staged=False)) as f:
				for i, row in enumerate(tqdm(f)):
					instances = PWInstance.from_json(row, qdep_required, lowercase=True)
					for instance in instances:
						data['rules'].append(instance.rules)
						data['facts'].append(instance.facts)
						data['ques'].append(instance.ques)
						data['answer'].append(instance.answer)
						data['proof'].append(instance.proofs)
						data['qdep'].append(instance.qdep)
						data['equiv_id'].append(instance.equiv_id)

		elif (args.dataset.startswith('pwu_pararules') and (not '_eq_' in args.dataset)):
			if split == 'train':
				all_subdirs = ['NatLang', 'depth-3']
			else:
				all_subdirs = ['NatLang']
			for subdir in all_subdirs:
				with jsonlines.open(get_pw_fname(args, subdir, split, is_staged=False)) as f:
					for i, row in enumerate(tqdm(f)):
						instances = PWInstance.from_json(row, lowercase=True)
						for instance in instances:
							data['rules'].append(instance.rules)
							data['facts'].append(instance.facts)
							data['ques'].append(instance.ques)
							data['answer'].append(instance.answer)
							data['proof'].append(instance.proofs)
							data['qdep'].append(instance.qdep)
							data['equiv_id'].append(instance.equiv_id)

#########################################

		elif args.dataset.startswith('pwur_'):
			if split == 'test':
				subdir = get_pw_subdir(args.dataset)
				if '_eq_' in args.dataset:
					qdep_required = int(args.dataset.split("_")[-2]) # the depth which we want (NOTE This is -2 only for this dataset)
				else:
					qdep_required = None
				with jsonlines.open(get_pw_fname(args, subdir, split, is_staged=False, equivalence=True)) as f:
					for i, row in enumerate(tqdm(f)):
						instances = PWInstance.from_json(row, qdep_required, lowercase=True)
						for instance in instances:
							data['rules'].append(instance.rules)
							data['facts'].append(instance.facts)
							data['ques'].append(instance.ques)
							data['answer'].append(instance.answer)
							data['proof'].append(instance.proofs)
							data['qdep'].append(instance.qdep)
							data['equiv_id'].append(instance.equiv_id)

#########################################

		elif args.dataset.startswith('pwq_leq'):
			# load the relevant original file and select all the data
			if args.dataset == 'pwq_leq_0to3':
				# COMBINE DEPTH 0, 1,2 AND SAMPLE 20%. ADD DEPTH 3 TO THIS TO MAKE THE TRAIN/VALID/TEST DATA
				subdirs        = [get_pw_subdir(f'pwq_leq_{i}') for i in range(4)] # subdirs are common for both pwq and pwqr datasets
				stagefiles     = [get_pw_fname(args, subdir, split, is_staged=True, equivalence=False) for subdir in subdirs]
				non_stagefiles = [get_pw_fname(args, subdir, split, is_staged=False, equivalence=False) for subdir in subdirs]
				row_chunks     = [get_row_chunks(stagefiles[i], non_stagefiles[i], False) for i in range(len(subdirs))]
				row_chunks012  = []
				for i in range(3):
					row_chunks012.extend(row_chunks[i])
				row_chunks012 = random.sample(row_chunks012, len(row_chunks012)//5)
				# print(f'Length of row_chunks for dataset row_chunks012 is = {len(row_chunks012)}')
				row_chunks012.extend(row_chunks[3])
				row_chunks = row_chunks012 # Has all elemennts of depth 3 stage file, along with 20 percent from depth 0 to 2. Format is list of lists [[non-stagefile1, stagefile1-add0, stagefile1-add1 ...], [], []]
				print(f'Length of row_chunks for dataset {args.dataset} is = {len(row_chunks)}')
			else:
				subdir    	  = get_pw_subdir(args.dataset)
				stagefile     = get_pw_fname(args, subdir, split, is_staged=True, equivalence=False)
				non_stagefile = get_pw_fname(args, subdir, split, is_staged=False, equivalence=False)
				row_chunks    = get_row_chunks(stagefile, non_stagefile, False) # gives back a list of lists [[non-stagefile1, stagefile1-add0, stagefile1-add1 ...], [], []]

			if args.fairr_model == 'fairr_rule':
				for row_chunk in tqdm(row_chunks):
					non_stage_row = row_chunk[0] # row chunk is of the form [non-stagefile1, stagefile1-add0, stagefile1-add1 ...]
					stage_rows    = row_chunk[1:]
					if is_valid_row(args, non_stage_row['id']):
						instances = PWQRuleInstance.from_json(non_stage_row, stage_rows) # returns a list of instances [instance1, instance2, ....]
						for instance in instances:
							output = instance.tokenize(tokenizer, args.arch, split)
							make_data_from_instance(data, output, get_keys(args.fairr_model))

							# Do some accounting and upsampling if required
							if sum(instance.labels) > 0.0:
								pos_count += 1
							else:
								assert sum(instance.labels) == 0.0
								if instance.strategy in ['proof', 'inv-proof']:
									good_neg_count += 1
								else:
									bad_neg_count += 1

			elif args.fairr_model == 'fairr_fact':
				for row_chunk in tqdm(row_chunks):
					non_stage_row = row_chunk[0] # row chunk is of the form [non-stagefile1, stagefile1-add0, stagefile1-add1 ...]
					stage_rows    = row_chunk[1:]
					if is_valid_row(args, non_stage_row['id']):
						instances = PWQFactInstance.from_json(non_stage_row, stage_rows) # returns a list of instances [instance1, instance2, ....]
						for instance in instances:
							output   = instance.tokenize(tokenizer, args.arch, split)
							make_data_from_instance(data, output, get_keys(args.fairr_model))

#########################################

# write the data in pickle format to processed folder

		if args.dataset.startswith('pwu_') or args.dataset.startswith('pwur_'):
			keys = ['rules', 'facts', 'ques', 'answer', 'proof', 'qdep', 'equiv_id']
			pickle_dump_file(keys, data, split, args, is_staged=False)

		elif args.dataset.startswith('pwq_') or args.dataset.startswith('pw_'):
			pickle_dump_file(get_keys(args.fairr_model), data, split, args, is_staged=True)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Preprocess data')

	# pw_ --> staged dataset (used for training reasoner)
	# pwq_ --> question augmented staged dataset (used for training ruleselector and factselector)
	# pwu_ --> unstaged (used for evaluation)
	# pwur_ --> robustness datasets (used for robustness evaluation)
	parser.add_argument('--dataset', choices=[	'pw_leq_0to3', 'pwu_leq_3', 'pwq_leq_0to3', \
												'pwu_leq_3_eq_3', \
												'pwur_leq_3_name', 'pwur_leq_3_attr', 'pwur_leq_3_name$attr',\
												'pwur_leq_3_eq_0_name', 'pwur_leq_3_eq_1_name', 'pwur_leq_3_eq_2_name', 'pwur_leq_3_eq_3_name', 'pwur_leq_3_eq_100_name',\
												'pwur_leq_3_eq_0_attr', 'pwur_leq_3_eq_1_attr', 'pwur_leq_3_eq_2_attr', 'pwur_leq_3_eq_3_attr', 'pwur_leq_3_eq_100_attr',\
												'pwur_leq_3_eq_0_name$attr', 'pwur_leq_3_eq_1_name$attr', 'pwur_leq_3_eq_2_name$attr', 'pwur_leq_3_eq_3_name$attr',\
												'pwur_leq_3_eq_100_name$attr'])
	parser.add_argument('--fairr_model', choices=['fairr_rule', 'fairr_fact', 'fairr_reasoner', 'fairr_iter'])
	parser.add_argument('--world_assump', default='OWA', choices=['OWA'])
	parser.add_argument('--arch', default='roberta_large', choices=['t5_base', 'roberta_large', 't5_large'])
	args = parser.parse_args()

	main(args)
