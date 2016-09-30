import time
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import keras.backend as K
import glob


def one_hot_encoder(data, whole_set, char2idx):
	"""
	Encode the whole list, not a single record
	"""
	ret = []
	for i in data:
		idx = char2idx[i]
		tmp = np.zeros(len(whole_set), dtype=np.int8)
		tmp[idx] = 1
		ret.append(tmp)
	return np.asarray(ret)


def one_hot_decoder(data, whole_set):
	ret = []
	if data.ndim == 1: # keras bug ?
		data = np.expand_dims(data, 0)
	for probs in data:
		idx = np.argmax(probs)
		# print idx, whole_set[idx], probs[idx]
		ret.append(whole_set[idx])
	return ret


def top_one_prob(data):
	ret = []
	if data.ndim == 1: # keras bug ?
		data = np.expand_dims(data, 0)
	for probs in data:
		idx = np.argmax(probs)
		ret.append(probs[idx])
	return ret


def list2str(data):
	return ''.join([i if i != 'empty' else '' for i in data])


def plot_loss_figure(history, save_path):
	train_loss = history.history['loss']
	val_loss = history.history['val_loss']
	plt.plot(train_loss, 'b', val_loss, 'r')
	plt.xlabel('train_loss: blue   val_loss: red      epoch')
	plt.ylabel('loss')
	plt.title('loss figure')
	plt.savefig(save_path)


def load_data(input_dir, max_nb_cha, width, height, channels, char_set, char2idx):
	"""
	The format of the file folder
	All image file, named as 'id.jpg', id starts from 1
	label.txt, the first row correspond to 1.jpg's label, the file should be encoded in utf-8
	"""
	print 'Loading data...'
	tag = time.time()
	x = []
	y = []

	for dirpath, dirnames, filenames in os.walk(input_dir):
		nb_pic = len(glob.glob(dirpath + os.sep + '*.jpg'))
		if nb_pic <= 0:
			continue
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
					if i < len(raw): # make sure there isn't empty label
						y[-1].append(raw[i])
					else:
						y[-1].append('empty')

	# transform to keras format, and do one hot encoding
	x = np.asarray(x)
	x /= 255 # normalized
	y = [one_hot_encoder(i, char_set, char2idx) for i in y]
	y = np.asarray(y)
	if y.shape[1] == 1: # keras bug ?
		y = y[:,0,:] 
	print 'Data loaded, spend time(m) :', (time.time()-tag)/60
	return [x, y]


def load_img(path, width, height, channels):
	img = Image.open(path)
	img = img.resize((width, height))
	if channels==1: # convert the image to gray scale image if it's RGB
		img = img.convert('L')
	img = np.asarray(img, dtype='float32')
	if channels > 1:
		img = np.rollaxis(img, 2, 0)
	else:
		img = np.expand_dims(img, 0)
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
	if label.ndim < 3: # in case output_size==1
		return None
	ret = []
	for i in label:
		ret.append([])
		tag = False
		for j in i:
			cha = whole_set[np.argmax(j)]
			weight = 0
			if cha == 'empty' and tag == False:
				weight = 1 # TODO
				tag = True 
			if cha != 'empty':
				weight = 1
			ret[-1].append(weight)
	ret = np.asarray(ret)
	return ret