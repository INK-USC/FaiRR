'''
This script generates the robustness equivalence sets. The basic idea is to read a theory from original dataset
and then replace the names (or attributes) with some randomly sampled similar names (or attributes).
'''

import sys
sys.path.insert(0, "./")
from helper import *
from process_proofwriter import get_row_chunks

random.seed(42)

theory_names = ['Anne', 'Bob', 'Charlie', 'Dave', 'Erin', 'Fiona', 'Gary', 'Harry']
theory_replacement_names = {
	'v1' : ['George', 'Paul', 'Ronald', 'Emma', 'Magnus', 'Timothy', 'Chris', 'Molly', 'Diana', 'Joseph', 'Becky', 'Kurt', 'Ivan', 'Steve', 'Laura', 'Oliver', 'Adam', 'Larry'],
}

attributes = ['red', 'blue', 'green', 'kind', 'nice', 'big', 'cold', 'young', 'round', 'rough', 'white', 'smart', 'quiet', 'furry']
attributes_repl = {
	'v1' : ['maroon', 'brown', 'black', 'orange', 'cordial', 'friendly', 'adorable', 'old', 'soft', 'violent', 'intelligent', 'square', 'warm', 'large', 'cylindrical', 'spherical', 'tiny', 'microscopic', 'brilliant', 'noisy', 'playful', 'tender', 'gracious', 'patient', 'funny', 'hilarious', 'thorny', 'sensitive', 'diplomatic', 'thoughtful'],
}

common_names = ['cat', 'dog', 'bald eagle', 'rabbit', 'mouse', 'tiger', 'lion', 'bear', 'squirrel', 'cow']
common_names_repl = {
	'v1' : ['mother', 'father', 'baby', 'child', 'toddler', 'teenager', 'grandmother', 'student', 'teacher', 'alligator', 'cricket', 'bird', 'wolf', 'giraffe', 'dinosaur', 'thief', 'soldier', 'officer', 'artist', 'shopkeeper', 'caretaker', 'janitor', 'minister', 'salesman', 'saleswoman', 'runner', 'racer', 'painter', 'dresser', 'shoplifter'],
}

def get_equiv_name(args):
	equiv_names = []
	if args.names:
		equiv_names.append('name')

	if args.attr:
		equiv_names.append('attrs')

	if args.comm_names:
		equiv_names.append('comm')

	if args.name_attr:
		equiv_names.append('name$attr')

	return '$'.join(equiv_names)

def get_input_fname(split, world_assump, dataset, staged=False):
	if staged:
		return f'../data/raw/proofwriter/{world_assump}/{dataset}/meta-stage-{split}.jsonl'
	else:
		return f'../data/raw/proofwriter/{world_assump}/{dataset}/meta-{split}.jsonl'

def get_output_fname(args, staged=False):
	version_name = '' if args.version == '' or args.version == 'v1' else args.version

	if staged:
		return f'../data/raw/proofwriter/{args.world_assump}/{args.dataset}/meta-stage-{args.split}-equiv_{get_equiv_name(args)}{version_name}.jsonl'
	else:
		return f'../data/raw/proofwriter/{args.world_assump}/{args.dataset}/meta-{args.split}-equiv_{get_equiv_name(args)}{version_name}.jsonl'

def dump_format(data, staged=True):
	if staged:
		return {
					'id'            : data['id'],
					'maxD'          : data['maxD'],
					'NFact'         : data['NFact'],
					'NRule'         : data['NRule'],
					'triples'       : data['triples'],
					'rules'         : data['rules'],
					'allInferences' : data['allInferences'],
					'equiv_id'      : data.get('equiv_id', f'{data["id"]}_original'),
				}
	else:
		return {
					'id'            : data['id'],
					'maxD'          : data['maxD'],
					'NFact'         : data['NFact'],
					'NRule'         : data['NRule'],
					'triples'       : data['triples'],
					'rules'         : data['rules'],
					'questions'     : data['questions'],
					'proofDetails'	: data['proofDetails'],
					'equiv_id'      : data.get('equiv_id', f'{data["id"]}_original'),
				}

def replace_json(inp, mapping, count_idx, args, staged=True):
	if not staged:
		try:
			for key, val in mapping.items():
				inp['theory'] = inp['theory'].replace(key, val)
				for triple_id, triple_val in inp['triples'].items():
					inp['triples'][triple_id] = {k: v.replace(key, val) for k,v in triple_val.items()}
				for rule_id, rule_val in inp['rules'].items():
					inp['rules'][rule_id] = {k: v.replace(key, val) for k,v in rule_val.items()}
				for ques_id, ques_val in inp['questions'].items():
					for qval_key,qval_val in ques_val.items():
						if type(qval_val) == str:
							inp['questions'][ques_id][qval_key] = qval_val.replace(key, val)
						elif qval_key == 'proofsWithIntermediates':
							assert type(qval_val) == list
							new_pwi = []
							for intermediate in qval_val:
								if len(intermediate['intermediates']):
									for int_id, int_val in intermediate['intermediates'].items():
										intermediate['intermediates'][int_id] = {k: v.replace(key, val) for k,v in int_val.items()}
								new_pwi.append(intermediate)
							inp['questions'][ques_id]['proofsWithIntermediates'] = new_pwi

				for i, conc in enumerate(inp['proofDetails']):
					# conc is a dict of the form {test:, QDep:, proofsWithIntermediates}
					for conc_key,conc_val in conc.items():
						if type(conc_val) == str:
							inp['proofDetails'][i][conc_key] = conc_val.replace(key, val)
						elif conc_key == 'proofsWithIntermediates':
							assert type(conc_val) == list
							new_pwi = []
							for intermediate in conc_val:
								if len(intermediate['intermediates']):
									for int_id, int_val in intermediate['intermediates'].items():
										intermediate['intermediates'][int_id] = {k: v.replace(key, val) for k,v in int_val.items()}
								new_pwi.append(intermediate)
							inp['proofDetails'][i]['proofsWithIntermediates'] = new_pwi

		except Exception as e:
			print('Exception Cause: {}'.format(e.args[0]))
			import pdb; pdb.set_trace()
	else:
		try:
			for key, val in mapping.items():
				for triple_id, triple_val in inp['triples'].items():
					inp['triples'][triple_id] = {k: v.replace(key, val) for k,v in triple_val.items()}
				for rule_id, rule_val in inp['rules'].items():
					inp['rules'][rule_id] = {k: v.replace(key, val) for k,v in rule_val.items()}
				if len(inp['allInferences']):
					new_allinferences = []
					for inference in inp['allInferences']:
						inference = {k: v.replace(key, val) for k,v in inference.items()}
						new_allinferences.append(inference)
					inp['allInferences'] = new_allinferences
		except Exception as e:
			print('Exception Cause: {}'.format(e.args[0]))
			import pdb; pdb.set_trace()

	# add the equivalence id
	inp['equiv_id'] = f'{inp["id"]}_{get_equiv_name(args)}_{count_idx}'

	return inp

def sample_and_map(theory, original_list, new_list):
	mapping = {}
	token_bank = deepcopy(new_list)
	for token in original_list:
		if token in theory:
			sampled_token  = random.sample(token_bank, 1)[0]
			mapping[token] = sampled_token
			token_bank.remove(sampled_token)

	return mapping

def sample_equivalence_dicts(theory, args):
	global theory_names, theory_replacement_names, attributes, attributes_repl, common_names, common_names_repl

	if args.version == '':
		version = 'v1'
	else:
		version = args.version

	equivalence_mappings = []
	for idx in range(args.equiv_count):
		if args.names:
			name_map = sample_and_map(theory, theory_names, theory_replacement_names[version])
			if len(name_map) > 0:
				equivalence_mappings.append(name_map)

		if args.attr:
			attr_map = sample_and_map(theory, attributes, attributes_repl[version])
			if len(attr_map) > 0:
				equivalence_mappings.append(attr_map)

		if args.comm_names:
			comm_map = sample_and_map(theory, common_names, common_names_repl[version])
			if len(comm_map) > 0:
				equivalence_mappings.append(comm_map)

		if args.name_attr:
			name_map = sample_and_map(theory, theory_names, theory_replacement_names[version])
			attr_map = sample_and_map(theory, attributes, attributes_repl[version])
			name_attr_map = {**name_map, **attr_map}
			if len(name_attr_map) > 0:
				equivalence_mappings.append(name_attr_map)

	return equivalence_mappings

def main(args):

	if args.names:
		args.comm_names = True
	else:
		args.comm_names = False

	row_chunks = get_row_chunks(get_input_fname(args.split, args.world_assump, args.dataset, staged=True),
								get_input_fname(args.split, args.world_assump, args.dataset, staged=False))

	new_staged_data, new_unstaged_data = [], []
	for row in tqdm(row_chunks):
		# parse the original data
		unstaged_data, staged_data = row[0], row[1:]
		new_unstaged_data.append(dump_format(unstaged_data, staged=False))
		_ = [new_staged_data.append(dump_format(x, staged=True)) for x in staged_data]

		# generate multiple equivalence mappings
		equivalence_mappings = sample_equivalence_dicts(unstaged_data['theory'], args)

		# create equivalence data
		for count_idx, mapping in enumerate(equivalence_mappings):
			equiv_unstaged_data = replace_json(deepcopy(unstaged_data), mapping, count_idx, args, staged=False)
			equiv_staged_data   = [replace_json(deepcopy(x), mapping, count_idx, args, staged=True) for x in staged_data]

			# add the data
			new_unstaged_data.append(dump_format(equiv_unstaged_data, staged=False))
			_ = [new_staged_data.append(dump_format(x, staged=True)) for x in equiv_staged_data]

	print(len(new_unstaged_data), len(new_staged_data))

	# write new data in same folder
	print(f'file writing to = {get_output_fname(args, staged=False)}')
	with jsonlines.open(get_output_fname(args, staged=False), mode='w') as writer:
		for row in new_unstaged_data:
			writer.write(row)

	print(f'file writing to = {get_output_fname(args, staged=True)}')
	with jsonlines.open(get_output_fname(args, staged=True), mode='w') as writer:
		for row in new_staged_data:
			writer.write(row)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Fact Augmentation')
	parser.add_argument('--split',  		type=str, 	default='test',		choices=['train', 'dev', 'test'])
	parser.add_argument('--world_assump', 	type=str,	default='OWA', 		choices=['OWA'])
	parser.add_argument('--dataset', 					default='depth-3', 	choices=['depth-0', 'depth-1', 'depth-2', 'depth-3', 'depth-5',])
	parser.add_argument('--equiv_count',	type=int,	default=5)
	parser.add_argument('--version', 		type=str,	default='', 		choices=['', 'v1'])
	parser.add_argument('--names', 			action='store_true')
	parser.add_argument('--comm_names', 	action='store_true')
	parser.add_argument('--attr', 			action='store_true')
	parser.add_argument('--name_attr', 		action='store_true')

	args = parser.parse_args()

	main(args)
