# coding:utf-8
import sys
sys.path.append('/home/feixingjian/DeepLearning-OCR/')
from architecture.CNN_LSTM import build_CNN_LSTM
from architecture.cv_vgg import build_vgg
from util import get_char_set, get_maxnb_char, one_hot_decoder, list2str
from post_correction import get_label_set, correction


class model(object):
    def __init__(self):
        self.char_set = get_char_set(self.train_data_dir)[0]
        self.nb_classes = len(self.char_set)
        self.max_nb_char = get_maxnb_char(self.train_data_dir)
        self.label_set = get_label_set(self.train_data_dir)


    def pred(self, X):
        pred_res = self.model.predict(X)
        pred_res = [one_hot_decoder(i, self.char_set) for i in pred_res]
        pred_res = [list2str(i) for i in pred_res]
        # post correction
        if self.post_correction:
            pred_res = correction(pred_res, self.label_set)
        return pred_res


class vgg(model):
    def __init__(self):
        model.__init__(self)
        self.multiple = True
        self.model = build_vgg(self.img_channels, self.img_width, self.img_height, self.max_nb_char, self.nb_classes) # 生成CNN的架构
        self.model.load_weights(self.weights_file_path) # 读取训练好的模型


class cnn_lstm(model):
    def __init__(self):
        model.__init__(self)
        self.multiple = False
        self.model = build_CNN_LSTM(self.img_channels, self.img_width, self.img_height, self.max_nb_char, self.nb_classes) # 生成CNN的架构
        self.model.load_weights(self.weights_file_path) # 读取训练好的模型


class chi_single(vgg):
    def __init__(self):
        self.img_width = 48
        self.img_height = 48
        self.img_channels = 1
        self.post_correction = False
        self.train_data_dir = '/home/feixingjian/DeepLearning-OCR/train_data/chinese_200000/'
        self.weights_file_path = '/home/feixingjian/DeepLearning-OCR/save_model/2016-08-15/weights.35-0.28.hdf5'
        vgg.__init__(self)


class jiangsu(cnn_lstm):
    def __init__(self):
        self.img_width = 150
        self.img_height = 60
        self.img_channels = 1
        self.post_correction = False
        self.train_data_dir = '/home/feixingjian/DeepLearning-OCR/train_data/jiangsu/'
        self.weights_file_path = '/home/feixingjian/DeepLearning-OCR/save_model/2016-08-01/weights.491-0.47.hdf5'
        cnn_lstm.__init__(self)        


class beijing(cnn_lstm):
    def __init__(self):
        self.img_width = 150
        self.img_height = 50
        self.img_channels = 3
        self.post_correction = False
        self.train_data_dir = '/home/feixingjian/DeepLearning-OCR/train_data/beijing/'
        self.weights_file_path = '/home/feixingjian/DeepLearning-OCR/save_model/2016-07-22/weights.66-0.00.hdf5'
        cnn_lstm.__init__(self)


class guangdong(cnn_lstm):
    def __init__(self):
        self.img_width = 180
        self.img_height = 40
        self.img_channels = 1
        self.post_correction = True
        self.train_data_dir = '/home/feixingjian/DeepLearning-OCR/train_data/guangdong/'
        self.weights_file_path = '/home/feixingjian/DeepLearning-OCR/save_model/2016-08-01/weights.09-0.03.hdf5'
        cnn_lstm.__init__(self)


class hubei(cnn_lstm):
    def __init__(self):
        self.img_width = 150
        self.img_height = 40
        self.img_channels = 1
        self.post_correction = False
        self.train_data_dir = '/home/feixingjian/DeepLearning-OCR/train_data/hubei/'
        self.weights_file_path = '/home/feixingjian/DeepLearning-OCR/save_model/2016-08-11/weights.32-0.00.hdf5'
        cnn_lstm.__init__(self)