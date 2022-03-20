from helper import *
from pw_helper import *


class PWReasonerInstance:

	def __init__(self, rule, facts, conclusion):
		self.rule  		= rule # selected from rule selector
		self.facts  	= facts # selected from fact selector
		self.conclusion = conclusion

	@classmethod
	def from_json(cls, json_dict):
		'''
		creates a input output pair, where input contains facts + rules and output contains the generated conclusion
		'''
		all_inferences = parse_all_inferences(json_dict, return_text=True, pwq=False)

		instances = []
		for inference in all_inferences:
			proofs, conclusion = inference
			for proof in proofs:
				facts, _, rule, _ = proof
				instances.append(PWReasonerInstance(rule, facts, conclusion))

		return instances

	def tokenize_ptlm(self, tokenizer):
		# convert the data in the format expected by the PTLM
		# input format: facts rule </s>
		# output format: <pad> conclusion </s>

		input_tokens  = format_facts(self.facts) + self.rule + tokenizer.eos_token
		input_ids     = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_tokens))
		output_tokens = tokenizer.pad_token + self.conclusion + tokenizer.eos_token
		output_ids    = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output_tokens))

		return input_ids, output_ids

	def tokenize(self, tokenizer, arch, split):
		if arch == 't5_base' or arch == 't5_large':
			return self.tokenize_ptlm(tokenizer)
		else:
			raise NotImplementedError

	@classmethod
	def tokenize_instance(cls, tokenizer, rule, facts):
		input_tokens  = ' '.join(facts) + rule + tokenizer.eos_token
		input_ids     = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_tokens, truncation=True))

		return input_ids

	@classmethod
	def tokenize_batch(cls, tokenizer, batched_rules, batched_facts):
		new_rules        = [rule.lower() for rule in batched_rules]
		new_facts        = [list(map(str.lower, facts)) if len(facts) > 0 else [] for facts in batched_facts]
		input_tokens     = [(' '.join(facts) if len(facts) > 0 else '') + rule + tokenizer.eos_token for facts,rule in zip(new_facts, new_rules)]
		tokenized        = tokenizer(input_tokens, add_special_tokens=False, padding=True, truncation=True, return_tensors='pt')
		input_ids        = tokenized['input_ids']

		return input_ids
