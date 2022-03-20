from helper import *

def main(args):
	preds = ddict(dict)

	with open(f'../saved/{args.exp_id}/pred.csv') as f:
		reader = csv.reader(f)
		count = 0
		block = []
		consistency = []
		prev_row_id, prev_equiv_count = None, 'original'
		for row in reader:
			ques = row[5]
			row_id, equiv_count = row[1].split('_', 1)
			if prev_row_id is None or row_id == prev_row_id:
				if equiv_count == 'original':
					block.append([(ques, row[7], row[2], row[9], row[10])])				# [question, gold answer, predicted ans, ans match, proof match]
				else:
					if prev_equiv_count != equiv_count:
						count = 0
						block[count].append((ques, row[7], row[2], row[9], row[10]))
					else:
						count += 1
						block[count].append((ques, row[7], row[2], row[9], row[10]))

				prev_row_id = row_id
				prev_equiv_count = equiv_count
			else:
				assert equiv_count == 'original'

				# process the accumulated block
				for item in block:
					if len(item) == 1:
						consistency.append(1)
					else:
						orig_pred = item[0][2]
						rest_pred = [item[x][2] for x in range(1, len(item))]
						res = (np.array(rest_pred) == orig_pred)
						consistency.append(np.mean(res))

				# reset and process current row
				block = []
				prev_row_id = row_id
				prev_equiv_count = 'original'
				block.append([(ques, row[7], row[2], row[9], row[10])])

		print('Answer consistency: ', np.mean(consistency))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Robustness Consistency evaluation')
	parser.add_argument('--exp_id')
	args = parser.parse_args()

	main(args)
