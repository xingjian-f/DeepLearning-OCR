# coding:utf-8
import sys
sys.path.append('../')
from architecture.CNN_LSTM import build_CNN_LSTM
from architecture.vgg_merge import build_vgg_merge
from architecture.shallow import build_shallow
from util import get_char_set, get_maxnb_char, one_hot_decoder, list2str, top_one_prob
from post_correction import get_label_set, correction


class model(object):
    def __init__(self):
        self.char_set = get_char_set(self.train_data_dir)[0]
        self.nb_classes = len(self.char_set)
        self.max_nb_char = get_maxnb_char(self.train_data_dir)
        self.label_set = get_label_set(self.train_data_dir)
        self.pred_probs = None


    def pred(self, X):
        pred_res = self.model.predict(X, batch_size=256)
        self.pred_probs = pred_res
        pred_res = [one_hot_decoder(i, self.char_set) for i in pred_res]
        pred_res = [list2str(i) for i in pred_res]
        # post correction
        if self.post_correction:
            pred_res = correction(pred_res, self.label_set)
        return pred_res


    def get_prob(self):
        probs = [top_one_prob(i) for i in self.pred_probs]
        probs = [i[0] for i in probs] #  TODO  
        return probs



class vgg_merge(model):
    def __init__(self):
        model.__init__(self)
        self.model = build_vgg_merge(self.img_channels, self.img_width, self.img_height, self.max_nb_char, self.nb_classes) # 生成CNN的架构
        self.model.load_weights(self.weights_file_path) # 读取训练好的模型


class cnn_lstm(model):
    def __init__(self):
        model.__init__(self)
        self.model = build_CNN_LSTM(self.img_channels, self.img_width, self.img_height, self.max_nb_char, self.nb_classes) # 生成CNN的架构
        self.model.load_weights(self.weights_file_path) # 读取训练好的模型


class shallow(model):
    def __init__(self):
        model.__init__(self)
        self.model = build_shallow(self.img_channels, self.img_width, self.img_height, self.max_nb_char, self.nb_classes) # 生成CNN的架构
        self.model.load_weights(self.weights_file_path) # 读取训练好的模型   


class single_cha(shallow):
    def __init__(self):
        self.img_width = 48
        self.img_height = 48
        self.img_channels = 1
        self.post_correction = False
        self.train_data_dir = '../train_data/single_1000000/'
        self.weights_file_path = '../save_model/2016-09-13/weights.05-0.09.hdf5'
        shallow.__init__(self)


class chi_rotate(shallow):
    def __init__(self):
        self.img_width = 32
        self.img_height = 32
        self.img_channels = 1
        self.post_correction = False
        self.train_data_dir = '../train_data/chi_rotate_1000000/'
        self.weights_file_path = '../save_model/2016-10-27/weights.981-0.32.hdf5'
        shallow.__init__(self)


class nacao(shallow):
    def __init__(self):
        self.img_width = 223
        self.img_height = 50
        self.img_channels = 1
        self.post_correction = False
        self.train_data_dir = '../train_data/nacao_captcha1/'
        self.weights_file_path = '../save_model/2016-11-28/weights.58-0.01.hdf5'
        shallow.__init__(self)


class ip(shallow):
    def __init__(self):
        self.img_width = 200
        self.img_height = 60
        self.img_channels = 1
        self.post_correction = False
        self.train_data_dir = '../train_data/ip_train/'
        self.weights_file_path = '../save_model/2017-01-06/weights.151-0.04.hdf5'
        shallow.__init__(self)


class jiangxi(shallow):
    def __init__(self):
        self.img_width = 150
        self.img_height = 32
        self.img_channels = 1
        self.post_correction = False
        self.train_data_dir = '../train_data/jiangxi_train/'
        self.weights_file_path = '../save_model/2016-10-12/weights.01-0.00.hdf5'
        shallow.__init__(self)


class kanzhun(shallow):
    def __init__(self):
        self.img_width = 100
        self.img_height = 50
        self.img_channels = 1
        self.post_correction = False
        self.train_data_dir = '../train_data/kanzhun_1000/'
        self.weights_file_path = '../save_model/2016-10-24/weights.44-0.91.hdf5'
        shallow.__init__(self)


class lagou(shallow):
    def __init__(self):
        self.img_width = 113
        self.img_height = 42
        self.img_channels = 1
        self.post_correction = False
        self.train_data_dir = '../train_data/lagou_train/'
        self.weights_file_path = '../save_model/2016-10-27/weights.25-0.27.hdf5'
        shallow.__init__(self)


class shanghai(shallow):
    def __init__(self):
        self.img_width = 160
        self.img_height = 53
        self.img_channels = 1
        self.post_correction = False
        self.train_data_dir = '../train_data/shanghai/'
        self.weights_file_path = '../save_model/2016-10-12/weights.33-0.01.hdf5'
        shallow.__init__(self)


class hunan(shallow):
    def __init__(self):
        self.img_width = 160
        self.img_height = 53
        self.img_channels = 1
        self.post_correction = False
        self.train_data_dir = '../train_data/hunan_captcha/'
        self.weights_file_path = '../save_model/2016-11-09/weights.11-0.01.hdf5'
        shallow.__init__(self)


class shandong(shallow):
    def __init__(self):
        self.img_width = 300
        self.img_height = 50
        self.img_channels = 1
        self.post_correction = False
        self.train_data_dir = '../train_data/shandong_train/'
        self.weights_file_path = '../save_model/2016-10-12/weights.29-0.02.hdf5'
        shallow.__init__(self)


class guizhou(vgg_merge):
    def __init__(self):
        self.img_width = 260
        self.img_height = 40
        self.img_channels = 1
        self.post_correction = False
        self.train_data_dir = '../train_data/guizhou_res/'
        self.weights_file_path = '../save_model/2016-08-30/weights.137-0.12.hdf5'
        vgg_merge.__init__(self)


class zhejiang(cnn_lstm):
    def __init__(self):
        self.img_width = 250
        self.img_height = 50
        self.img_channels = 1
        self.post_correction = True
        self.train_data_dir = '../train_data/zhejiang_real/'
        self.weights_file_path = '../save_model/2016-08-24/weights.99-1.44.hdf5'
        cnn_lstm.__init__(self)   


class jiangsu(cnn_lstm):
    def __init__(self):
        self.img_width = 150
        self.img_height = 60
        self.img_channels = 1
        self.post_correction = False
        self.train_data_dir = '../train_data/jiangsu/'
        self.weights_file_path = '../save_model/2016-08-01/weights.491-0.47.hdf5'
        cnn_lstm.__init__(self)        


class beijing(cnn_lstm):
    def __init__(self):
        self.img_width = 150
        self.img_height = 50
        self.img_channels = 3
        self.post_correction = False
        self.train_data_dir = '../train_data/beijing/'
        self.weights_file_path = '../save_model/2016-07-22/weights.66-0.00.hdf5'
        cnn_lstm.__init__(self)


class guangdong(cnn_lstm):
    def __init__(self):
        self.img_width = 180
        self.img_height = 40
        self.img_channels = 1
        self.post_correction = True
        self.train_data_dir = '../train_data/guangdong/'
        self.weights_file_path = '../save_model/2016-08-01/weights.09-0.03.hdf5'
        cnn_lstm.__init__(self)


class hubei(cnn_lstm):
    def __init__(self):
        self.img_width = 150
        self.img_height = 40
        self.img_channels = 1
        self.post_correction = False
        self.train_data_dir = '../train_data/hubei/'
        self.weights_file_path = '../save_model/2016-08-11/weights.32-0.00.hdf5'
        cnn_lstm.__init__(self)


class anhui(cnn_lstm):
    def __init__(self):
        self.img_width = 200
        self.img_height = 50
        self.img_channels = 1
        self.post_correction = True
        self.train_data_dir = '../train_data/anhui/'
        self.weights_file_path = '../save_model/2016-10-09/weights.15-0.08.hdf5'
        cnn_lstm.__init__(self)