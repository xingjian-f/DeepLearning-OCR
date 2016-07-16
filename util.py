#coding:utf-8
import time
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import keras.backend as K

def one_hot_encoder(data, whole_set):
	"""
	 对整个list做encoder，而不是单个record
	"""
	ret = []
	for i in data:
		idx = whole_set.index(i)
		ret.append([1 if j==idx else 0 for j in range(len(whole_set))])
	return ret


def one_hot_decoder(data, whole_set):
	ret = []
	for probs in data:
		idx = np.argmax(probs)
		if whole_set[idx] != 'empty':
			ret.append(whole_set[idx])
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


def load_data(input_dir, max_nb_cha, width, height, channels, cha_set):
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
		for i in range(1, nb_pic+1):
			filename = str(i) + '.jpg'
			filepath = dirpath + os.sep + filename
			pixels = load_img(filepath, width, height, channels)
			x.append(pixels)
		
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
	x = x.astype('float32') # gpu只接受32位浮点运算
	x /= 255 # normalized
	y = [one_hot_encoder(i, cha_set) for i in y]
	y = np.array(y)

	print 'Data loaded, spend time(m) :', (time.time()-tag)/60
	return [x, y]


def load_img(path, width, height, channels):
	img = Image.open(path)
	im = img.resize((width, height)) # resize is necessary if not using FCN
	pixels = list(im.getdata())
	if channels > 1:
		x = [[[pixels[k*width+i][j] for k in range(height)] for i in range(width)] for j in range(channels)] # 转成（channel，width，height）shape
	else:
		x = [[[pixels[k*width+i] for k in range(height)] for i in range(width)]]
	return x


def get_char_set(file_dir):
	file_path = file_dir+'label.txt'
	ret = set(['empty'])
	with open(file_path) as f:
		for raw in f:
			raw = raw.decode('utf-8').strip('\r\n')
			for i in raw:
				ret.add(i)
	return ret


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