# coding:utf-8
import sys
sys.path.append('../')
import json
import numpy as np 
from PIL import Image
from architecture.CNN_LSTM import build_CNN_LSTM
from architecture.cv_vgg import build_vgg
from util import load_img, get_char_set, get_maxnb_char
from post_correction import get_label_set
from train import pred

class zhejiang():
    def __init__(self):
        self.img_width = 250
        self.img_height = 50
        self.img_channels = 1
        self.multiple = True
        self.post_correction = True
        self.train_data_dir = '/home/feixingjian/DeepLearning-OCR/train_data/zhejiang/'
        self.weights_file_path = '/home/feixingjian/DeepLearning-OCR/save_model/2016-07-31/weights.212-0.16.hdf5'
        self.char_set = get_char_set(self.train_data_dir)[0]
        self.nb_classes = len(self.char_set)
        self.max_nb_char = get_maxnb_char(self.train_data_dir)
        self.label_set = get_label_set(self.train_data_dir)
        self.model = build_vgg(self.img_channels, self.img_width, self.img_height, self.max_nb_char, self.nb_classes) # 生成CNN的架构
        self.model.load_weights(self.weights_file_path) # 读取训练好的模型


class beijing():
    def __init__(self):
        self.img_width = 150
        self.img_height = 50
        self.img_channels = 3
        self.multiple = False
        self.post_correction = False
        self.train_data_dir = '/home/feixingjian/DeepLearning-OCR/train_data/beijing/'
        self.weights_file_path = '/home/feixingjian/DeepLearning-OCR/save_model/2016-07-22/weights.66-0.00.hdf5'
        self.char_set = get_char_set(self.train_data_dir)[0]
        self.nb_classes = len(self.char_set)
        self.max_nb_char = get_maxnb_char(self.train_data_dir)
        self.label_set = get_label_set(self.train_data_dir)
        self.model = build_CNN_LSTM(self.img_channels, self.img_width, self.img_height, self.max_nb_char, self.nb_classes) # 生成CNN的架构
        self.model.load_weights(self.weights_file_path) # 读取训练好的模型


def load_data(img_vals, width, height, channels):
    x = []
    for img_val in img_vals:
       x.append(load_img(img_val, width, height, channels))
    x = np.asarray(x)
    x /= 255 # normalized
    return x


def parse_expr(expr):
    if expr[-2:] == u'等于':
        rep_dict = [
            (u'零', '0'),
            (u'壹', '1'),
            (u'贰', '2'),
            (u'叁', '3'),
            (u'肆', '4'),
            (u'伍', '5'),
            (u'陆', '6'),
            (u'柒', '7'),
            (u'捌', '8'),
            (u'玖', '9'),
            (u'加', '+'),
            (u'乘', '*'),
            (u'减', '-'),
            (u'等于', '')]
        for i in rep_dict:
            expr = expr.replace(i[0], i[1])
        try:
            return eval(expr)
        except:
            return False
    else:
        return expr


def predict(predictor, post_vals):
    # load model parameter
    img_width = predictor.img_width
    img_height = predictor.img_height
    img_channels = predictor.img_channels
    multiple = predictor.multiple
    post_correction = predictor.post_correction
    model = predictor.model
    char_set = predictor.char_set
    label_set = predictor.label_set
    # predict result
    keys = post_vals.keys()
    img_vals = post_vals.values()
    X_test = load_data(img_vals, img_width, img_height, img_channels)
    predictions = pred(model, X_test, char_set, label_set, multiple, post_correction)
    # format reply
    res = {}
    for i, expr in enumerate(predictions):
        valid = True
        ans = parse_expr(expr)
        if ans is False:
            valid = False
        form = {'valid':valid, 'answer':ans, 'expr':expr}
        res[keys[i]] = form
    if len(res) == 1: # in case input data is not in batch form 
        res = res['file']
    # print res, len(res)
    return json.dumps(res, ensure_ascii=False) # utf-8 output