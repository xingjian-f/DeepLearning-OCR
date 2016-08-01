import editdistance
import random

def get_label_set(file_dir):
	file_path = file_dir + 'label.txt'
	ret = set()
	with open(file_path) as f:
		for raw in f:
			raw = raw.decode('utf-8').strip('\n\r')
			if len(raw) > 0:
				ret.add(raw)
	return list(ret)


def edit_dis(a, b):
	return editdistance.eval(a, b)


def correction(preds, label_set):
	ret = []
	for i in range(len(preds)):
		dis_vector = [edit_dis(preds[i], j) for j in label_set]
		min_dis = min(dis_vector)
		ans_set = []
		for j in range(len(label_set)):
			if dis_vector[j] == min_dis:
				ans_set.append(label_set[j])
		# print preds[i]
		# print len(ans_set), min_dis
		# for i in ans_set:
		# 	print i
		ret.append(random.choice(ans_set))
	return ret