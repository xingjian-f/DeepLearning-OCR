import os
import time
from datetime import datetime
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from util import one_hot_decoder, plot_loss_figure, load_data, get_char_set, get_maxnb_char
from util import get_sample_weight, list2str, sin2mul
from post_correction import get_label_set, correction
# from architecture.CNN_LSTM import build_CNN_LSTM
from architecture.cv_vgg import build_vgg

# @profile
def pred(model, X, char_set, label_set, multiple, post_correction):
	pred_res = model.predict(X)
	pred_res = [one_hot_decoder(i, char_set) for i in pred_res]
	# multiple output format
	if multiple:
		pred_res = sin2mul(pred_res)
	pred_res = [list2str(i) for i in pred_res]
	# post correction
	if post_correction:
		pred_res = correction(pred_res, label_set)
	return pred_res

# @profile
def test(model, test_data, char_set, label_set, multiple, post_correction):
	test_X, test_y = test_data[0], test_data[1]
	test_y = [one_hot_decoder(i, char_set) for i in test_y]
	# multiple output format
	if multiple:
		test_y = sin2mul(test_y)
	test_y = [list2str(i) for i in test_y]
	pred_res = pred(model, test_X, char_set, label_set, multiple, post_correction)
	# for i in pred_res:
	# 	print i
	nb_correct = sum(pred_res[i]==test_y[i] for i in range(len(pred_res)))
	for i in range(len(pred_res)):
		if test_y[i] != pred_res[i]:
			print 'test:', test_y[i]
			print 'pred:', pred_res[i]
		pass
	print 'nb_correct: ', nb_correct
	print 'Acurracy: ', float(nb_correct) / len(pred_res)


def train(model, batch_size, nb_epoch, save_dir, train_data, val_data, char_set, multiple):
	X_train, y_train = train_data[0], train_data[1]
	sample_weight = get_sample_weight(y_train, char_set, multiple)
	print type(sample_weight)
	print 'X_train shape:', X_train.shape
	print X_train.shape[0], 'train samples'
	if os.path.exists(save_dir) == False:
		os.mkdir(save_dir)

	start_time = time.time()
	save_path = save_dir + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
	check_pointer = ModelCheckpoint(save_path, 
		save_best_only=True)
	history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, 
		validation_data=val_data,
		validation_split=0.1, 
		callbacks=[check_pointer],
		# sample_weight=sample_weight
		)

	plot_loss_figure(history, save_dir + str(datetime.now()).split('.')[0].split()[1]+'.jpg')
	print 'Training time(h):', (time.time()-start_time) / 3600


def main():
	img_width, img_height = 250, 50
	img_channels = 1 
	batch_size = 32
	nb_epoch = 500
	multiple = True
	post_correction = True

	save_dir = 'save_model/' + str(datetime.now()).split('.')[0].split()[0] + '/' # model is saved corresponding to the datetime
	train_data_dir = 'train_data/zhejiang/'
	val_data_dir = 'test_data/nacao_5/'
	test_data_dir = 'test_data/cv/'
	weights_file_path = 'save_model/2016-07-31/weights.212-0.16.hdf5'
	char_set, char2idx = get_char_set(train_data_dir)
	nb_classes = len(char_set)
	max_nb_char = get_maxnb_char(train_data_dir)
	label_set = get_label_set(train_data_dir)
	# print 'char_set:', char_set
	print 'nb_classes:', nb_classes
	print 'max_nb_char:', max_nb_char
	model = build_vgg(img_channels, img_width, img_height, max_nb_char, nb_classes) # build CNN architecture
	model.load_weights(weights_file_path) # load trained model

	# val_data = load_data(val_data_dir, max_nb_char, img_width, img_height, img_channels, char_set, char2idx)
	val_data = None
	train_data = load_data(train_data_dir, max_nb_char, img_width, img_height, img_channels, char_set, char2idx, multiple) 
	# train(model, batch_size, nb_epoch, save_dir, train_data, val_data, char_set)

	# train_data = load_data(train_data_dir, max_nb_char, img_width, img_height, img_channels, char_set, char2idx, multiple)
	test(model, train_data, char_set, label_set, multiple, post_correction)
	# val_data = load_data(val_data_dir, max_nb_char, img_width, img_height, img_channels, char_set, char2idx, multiple)
	# test(model, val_data, char_set, label_set, multiple, post_correction)
	# test_data = load_data(test_data_dir, max_nb_char, img_width, img_height, img_channels, char_set, char2idx, multiple)
	# test(model, test_data, char_set, label_set, multiple, post_correction)


if __name__ == '__main__':
	main()