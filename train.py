#coding:utf-8
import os
import time
from datetime import datetime
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from util import one_hot_decoder, plot_loss_figure, load_data, get_char_set, get_maxnb_char
from architecture.CNN_LSTM import build_CNN_LSTM


def pred(model, X, char_set):
	pred_res = model.predict(X)
	pred_res = [one_hot_decoder(i, char_set) for i in pred_res]
	return pred_res


def test(model, test_data, char_set):
	test_X, test_y = test_data[0], test_data[1]
	test_y = [one_hot_decoder(i, char_set) for i in test_y]
	pred_res = pred(model, test_X, char_set)
	nb_correct = sum(pred_res[i]==test_y[i] for i in range(len(pred_res)))
	for i in range(len(pred_res)):
		print test_y[i], pred_res[i]
	print 'Acurracy: ', float(nb_correct) / len(test_y)


def train(model, batch_size, nb_epoch, save_dir, train_data, val_data):
	X_train, y_train = train_data[0], train_data[1]
	print 'X_train shape:', X_train.shape
	print X_train.shape[0], 'train samples'
	if os.path.exists(save_dir) == False:
		os.mkdir(save_dir)

	start_time = time.time()
	save_path = save_dir + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
	check_pointer = ModelCheckpoint(save_path, 
		save_best_only=False)
	history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, 
		validation_data=val_data,
		validation_split=0.1, 
		callbacks=[check_pointer])

	plot_loss_figure(history, save_dir + str(datetime.now()).split('.')[0].split()[1]+'.jpg')
	print 'Training time(h):', (time.time()-start_time) / 3600

	
if __name__ == '__main__':
	img_width, img_height = 223, 50
	img_channels = 3 
	batch_size = 32
	nb_epoch = 500

	save_dir = 'save_model/' + str(datetime.now()).split('.')[0].split()[0] + '/' # 模型保存在当天对应的目录中
	train_data_dir = 'train_data/guangdong/'
	val_data_dir = 'test_data/nacao_5/'
	test_data_dir = 'test_data/nacao_5/'
	weights_file_path = 'save_model/2016-07-15/weights.499-0.07.hdf5'
	char_set = list(get_char_set(train_data_dir))
	nb_classes = len(char_set)
	max_nb_char = get_maxnb_char(train_data_dir)
	# print 'char_set:', char_set
	print 'nb_classes:', nb_classes
	print 'max_nb_char:', max_nb_char
	model = build_CNN_LSTM(img_channels, img_width, img_height, max_nb_char, nb_classes) # 生成CNN的架构
	model.load_weights(weights_file_path) # 读取训练好的模型

	# 先读取整个数据集，然后训练    
	# val_data = load_data(val_data_dir, max_nb_char, img_width, img_height, img_channels, char_set)
	val_data = None
	# train_data = load_data(train_data_dir, max_nb_char, img_width, img_height, img_channels, char_set) 
	# train(model, batch_size, nb_epoch, save_dir, train_data, val_data)

	# 测试	
	train_data = load_data(train_data_dir, max_nb_char, img_width, img_height, img_channels, char_set)
	test(model, train_data, char_set)
	# val_data = load_data(val_data_dir, max_nb_char, img_width, img_height, img_channels, char_set)
	# test(model, val_data, char_set)
	# test_data = load_data(test_data_dir, max_nb_char, img_width, img_height, img_channels, char_set)
	# test(model, test_data, char_set)	