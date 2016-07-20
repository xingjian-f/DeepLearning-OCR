#coding:utf-8
import time
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import keras.backend as K


@profile
def one_hot_encoder(data, whole_set, char2idx):
	"""
	 对整个list做encoder，而不是单个record
	"""
	ret = []
	for i in data:
		idx = char2idx[i]
		tmp = np.zeros(len(whole_set))
		tmp[idx] = 1
		ret.append(tmp)
	return ret


def one_hot_decoder(data, whole_set):
	ret = []
	for probs in data:
		idx = np.argmax(probs)
		if whole_set[idx] != 'empty':
			ret.append(whole_set[idx])
		else:
			break
	ret = ''.join(ret)
	return ret


def plot_loss_figure(history, save_path):
	train_loss = history.history['loss']
	val_loss = history.history['val_loss']
	plt.plot(train_loss, 'b', val_loss, 'r')
	plt.xlabel('train_loss: blue   val_loss: red      epoch')
	plt.ylabel('loss')
	plt.title('loss figure')
	plt.savefig(save_path)


@profile
def load_data(input_dir, max_nb_cha, width, height, channels, char_set, char2idx):
	"""
	文件夹的规范：
	所有图片文件，命名方式为id.jpg，id从1开始
	标签文件label.txt，第1行对应1.jpg的标签，文件必须用utf-8编码
	# y[0][0],y[0][1],y[0][2],y[0][3] 分别对应第1,2,3,4个字符类推
	"""
	print 'Loading data...'
	tag = time.time()
	x = []
	y = []

	for dirpath, dirnames, filenames in os.walk(input_dir):
		nb_pic = len(filenames)-1
		if nb_pic <= 0:
			continue
		for i in range(1, nb_pic+1):
			filename = str(i) + '.jpg'
			filepath = dirpath + os.sep + filename
			pixels = load_img(filepath, width, height, channels)
			x.append(pixels)
			# print sys.getsizeof(x), i
		
		label_path = dirpath + os.sep + 'label.txt'
		with open(label_path) as f:
			for raw in f:
				raw = raw.decode('utf-8').strip('\n\r')
				if len(raw) == 0:
					break
				y.append([])
				for i in range(max_nb_cha):
					if i < len(raw):
						y[-1].append(raw[i])
					else:
						y[-1].append('empty')

	# 转成keras能接受的数据形式，以及做one hot 编码
	x = np.array(x)
	x /= 255 # normalized
	y = [one_hot_encoder(i, char_set, char2idx) for i in y]
	y = np.array(y)

	print 'Data loaded, spend time(m) :', (time.time()-tag)/60
	return [x, y]


def load_img(path, width, height, channels):
	img = Image.open(path)
	img = img.resize((width, height)) # resize is necessary if not using FCN
	img = np.asarray(img, dtype='float32')
	if channels > 1:
		img = np.rollaxis(img, 2, 0)
	else:
		img = [[[img[k*width+i] for k in range(height)] for i in range(width)]] # TODO
	return img


def get_char_set(file_dir):
	file_path = file_dir+'label.txt'
	ret = set(['empty'])
	with open(file_path) as f:
		for raw in f:
			raw = raw.decode('utf-8').strip('\r\n')
			for i in raw:
				ret.add(i)
	char_set = list(ret)
	char2idx = dict(zip(char_set, range(len(char_set))))
	return char_set, char2idx


def get_maxnb_char(file_dir):
	file_path = file_dir+'label.txt'
	ret = 1
	with open(file_path) as f:
		for raw in f:
			raw = raw.decode('utf-8').strip('\r\n')
			ret = max(ret, len(raw))
	return ret


def categorical_accuracy_per_sequence(y_true, y_pred):
	return K.mean(K.min(K.equal(K.argmax(y_true, axis=-1),
				  K.argmax(y_pred, axis=-1)), axis=-1))


def get_sample_weight(label, whole_set):
	ret = []
	for i in label:
		ret.append([])
		tag = False
		for j in i:
			cha = whole_set[np.argmax(j)]
			weight = 0
			if cha == 'empty' and tag == False:
				weight = 1
				tag = True 
			if cha != 'empty':
				weight = 1
			ret[-1].append(weight)
	return np.array(ret)