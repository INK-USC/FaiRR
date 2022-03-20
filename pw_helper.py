'''
This script contains generic functions for working with the ProofWriter dataset format.
'''

from helper import *


class Node:
	def __init__(self, head):
		self.head = head

	def __str__(self):
		return str(self.head)


def get_proof_graph(proof_str):
	# The proof parsing function is taken from https://github.com/swarnaHub/PRover/blob/master/proof_utils.py

	stack = []
	last_open = 0
	last_open_index = 0
	pop_list = []
	all_edges = []
	all_nodes = []

	proof_str = proof_str.replace("(", " ( ")
	proof_str = proof_str.replace(")", " ) ")
	proof_str = proof_str.split()

	should_join = False
	for i in range(len(proof_str)):

		_s = proof_str[i]
		x = _s.strip()
		if len(x) == 0:
			continue

		if x == "(":
			stack.append((x, i))
			last_open = len(stack) - 1
			last_open_index = i
		elif x == ")":
			for j in range(last_open + 1, len(stack)):
				if isinstance(stack[j][0], Node):
					pop_list.append((stack[j][1], stack[j][0]))

			stack = stack[:last_open]
			for j in range((len(stack))):
				if stack[j][0] == "(":
					last_open = j
					last_open_index = stack[j][1]

		elif x == '[' or x == ']':
			pass
		elif x == "->":
			should_join = True
		else:
			# terminal
			if x not in all_nodes:
				all_nodes.append(x)

			if should_join:

				new_pop_list = []
				# Choose which ones to add the node to
				for (index, p) in pop_list:
					if index < last_open_index:
						new_pop_list.append((index, p))
					else:
						all_edges.append((p.head, x))
				pop_list = new_pop_list

			stack.append((Node(x), i))

			should_join = False

	return all_nodes, all_edges

def parse_proof_inference_unstaged(inference, json_dict, strategy, inference_data, lowercase=True):
	if 'FAIL' in inference['proofs']:
		assert 'proofsWithIntermediates' not in inference
		assert strategy not in ['proof', 'inv-proof']

	# some proofs might not have any intermediates if the statements are not provable
	if 'proofsWithIntermediates' not in inference:
		if inference_data:
			proofs = [['None']]
		else:
			proofs = ['None']

		if not ('Birds' in json_dict['id'] or 'Electricity' in json_dict['id']):
			assert 'FAIL' in inference['proofs']
		assert strategy not in ['proof', 'inv-proof']

	else:
		assert strategy in ['proof', 'inv-proof']
		assert 'FAIL' not in inference['proofs']

		proofs = []
		if inference_data:
			# get all proof strings
			all_proof_texts = inference['proofs'][2:-2].split(' OR ')

			for proof_text in all_proof_texts:
				# get the edge list from the proof string
				nodelist, edgelist = get_proof_graph(proof_text)

				if len(edgelist):
					# replace triple id with triple text
					fact_replacer = json_dict['triples'].get
					# replace rule id with rule text
					rule_replacer = json_dict['rules'].get

					def replacer(string):
						if lowercase:
							return fact_replacer(string)['text'].lower() if 'triple' in string else rule_replacer(string)['text'].lower()
						else:
							return fact_replacer(string)['text'] if 'triple' in string else rule_replacer(string)['text']

					edgelist   = [(replacer(x[0]), replacer(x[1])) for x in edgelist]
					proof_text = edgelist

				else:
					# proof only has a triple
					assert len(nodelist) == 1
					if lowercase:
						proof_text = [(json_dict['triples'][str(nodelist[0])]['text'].lower())]
					else:
						proof_text = [(json_dict['triples'][str(nodelist[0])]['text'])]

				proofs.append(proof_text)
		else:
			for item in inference['proofsWithIntermediates']:
				proofs.append(item)

	return proofs

def parse_all_questions(json_dict, inference_data=True, lowercase=True):
	outputs = []
	for key, metadata in json_dict['questions'].items():
		if metadata['answer'] == True:
			answer = 1
		elif metadata['answer'] == False:
			answer = -1
		elif metadata['answer'] == 'Unknown':
			answer = 0

		ques     = metadata['question'].lower() if lowercase else metadata['question']
		strategy = metadata['strategy']
		qdep     = int(metadata['QDep'])
		outputs.append([ques, answer, strategy, parse_proof_inference_unstaged(metadata, json_dict, strategy, inference_data, lowercase=lowercase), qdep])

	return outputs

def parse_proof_ids(text):
	splits   = text.split(' -> ')
	fact_ids = splits[0].strip('(').strip(')').split(' ')
	rule_id  = splits[1].strip('(').strip(')')

	return fact_ids, rule_id


def parse_proof_inference(inference, return_text=False, json_dict=None, take_first_proof=True, lowercase=True):
	'''
	parameters:
	take_first_proof - if true, takes only the first proof into account otherwise returns all proofs (for a particular conclusion)
	returns:
	proof - of the form (facts, fact_ids, rule, rule_id)
	conclusion - text for the conclusion made using the facts and rule in the above proof
	'''
	assert return_text and json_dict is not None
	conclusion  = inference['text'].lower() if lowercase else inference['text']
	proofs_list = inference['proofs'][2:-2].split(' OR ') # gets [((triple4) -> rule8),((triple6) -> rule7)] from [(((triple4) -> rule8) OR ((triple6) -> rule7))]

	if take_first_proof:
		# NOTE: taking the first proof of the inference only, for the question augmented case
		proof = parse_proof_text(proofs_list[0], json_dict, lowercase=lowercase)
		return proof, conclusion
	else:
		proofs = []
		for proof_text in proofs_list:
			proofs.append(parse_proof_text(proof_text, json_dict, lowercase=lowercase))
		return proofs, conclusion


def parse_all_inferences(json_dict, return_text=False, pwq=True, take_first_proof=True, lowercase=True):
	'''
	   parameters: pwq  = True means, its used for the question augmented case, in the unstaged file
	   this is for json_dicts of the staged files
	   returns: output: having the form {conclusion_text:(facts, fact_ids, rule, rule_id), .....]
	'''
	if pwq == True:
		output = {}
	else:
		output = []
	for inference in json_dict['allInferences']:
		try:
			# this is only applied because I found that , int the depth-5 stage dataset of proofwriter, there are 3 incorrect json lines,
			# with triple 91 being used in the proof, but triple 91 doesnot exist in the theory
			if pwq == True:
				proof, conclusion = parse_proof_inference(inference, return_text=return_text, json_dict=json_dict, take_first_proof=take_first_proof, lowercase=lowercase)
				assert conclusion not in output.keys()
				output[conclusion] = proof # in case take_first_proof is false, than this will be multiple proofs and not jsut one proof
			else:
				output.append(parse_proof_inference(inference, return_text=return_text, json_dict=json_dict, take_first_proof=False, lowercase=lowercase))
		except Exception as e:
			print(f'skipped because of an error, possibly, the fact in the proof doesnot exist in the theory eg the problem of triple91')
			print('Exception Cause: {}'.format(e.args[0]))
			continue # ie if there is an exception then continue

	return output

def parse_proof_details(json_dict, lowercase=False):
	# return all the conclusions possible for a specific json row (unstaged file)
	try:
		proof_details = json_dict['proofDetails']
		all_conclusions = ddict(list)
		for row in proof_details:
			conc = row['text'].lower() if lowercase else row['text']
			for proof in row['proofsWithIntermediates']:
				if len(proof['intermediates']):
					all_conclusions[conc].append([v['text'].lower() if lowercase else v['text'] for v in proof['intermediates'].values()])
				else:
					all_conclusions[conc].append([conc.lower() if lowercase else conc])
	except Exception as e:
		print('Proof Details absent in file. Skipping!!')
		all_conclusions = dict()

	return dict(all_conclusions)

def format_facts(facts):
	return ' '.join(facts)

def parse_proof_text(proof_text, json_dict, lowercase=True):
	fact_ids, rule_id = parse_proof_ids(proof_text)
	facts             = [json_dict['triples'][x]['text'].lower() if lowercase else json_dict['triples'][x]['text'] for x in fact_ids]
	rule              = json_dict['rules'][rule_id]['text'].lower() if lowercase else json_dict['rules'][rule_id]['text']

	return facts, fact_ids, rule, rule_id

def get_facts(json_dict, lowercase=True):
	numfacts      = len(json_dict['triples'])
	fact_list     = [x['text'].lower() if lowercase else x['text'] for x in json_dict['triples'].values()]
	factsnum_list = [f'triple{i+1}' for i in range(numfacts)]
	fact_para     = " ".join(fact_list) # facts with spaces in between

	return fact_para, fact_list, factsnum_list

def get_rules(json_dict, lowercase=True):
	numrules         = len(json_dict['rules'])
	rule_list        = [x['text'].lower() if lowercase else x['text'] for x in json_dict['rules'].values()]
	rulesnum_list    = [f'rule{i+1}' for i in range(numrules)]
	rule_para        = " ".join(rule_list) # rules with spaces in between
	rules_dict       = {}
	rules_tuple_list = []
	for i, rulenum in enumerate(json_dict['rules'].keys()):
		ruletext            = json_dict['rules'][rulenum]['text'].lower() if lowercase else json_dict['rules'][rulenum]['text']
		rules_dict[rulenum] = ruletext
		rules_tuple_list.append((rulenum, ruletext))

	return rule_para, rule_list, rulesnum_list, rules_dict, rules_tuple_list
