# coding:utf-8
import sys
sys.path.append('../')
import json
import numpy as np 
from PIL import Image
from architecture.CNN_LSTM import build_CNN_LSTM
from util import load_img, get_char_set, get_maxnb_char
from train import pred


class guangdong():
    def __init__(self):
        self.img_width = 223
        self.img_height = 50
        self.img_channels = 3
        self.train_data_dir = '/home/feixingjian/DeepLearning-OCR/train_data/guangdong/'
        self.weights_file_path = '/home/feixingjian/DeepLearning-OCR/save_model/2016-07-15/weights.499-0.07.hdf5'
        self.char_set = get_char_set(self.train_data_dir)
        self.nb_classes = len(self.char_set)
        self.max_nb_char = get_maxnb_char(self.train_data_dir)
        self.model = build_CNN_LSTM(self.img_channels, self.img_width, self.img_height, self.max_nb_char, self.nb_classes) # 生成CNN的架构
        self.model.load_weights(self.weights_file_path) # 读取训练好的模型

class jiangsu():
    def __init__(self):
        self.img_width = 150
        self.img_height = 60
        self.img_channels = 3
        self.train_data_dir = '/home/feixingjian/DeepLearning-OCR/train_data/jiangsu/'
        self.weights_file_path = '/home/feixingjian/DeepLearning-OCR/save_model/2016-07-18/weights.55-0.06.hdf5'
        self.char_set = get_char_set(self.train_data_dir)
        self.nb_classes = len(self.char_set)
        self.max_nb_char = get_maxnb_char(self.train_data_dir)
        self.model = build_CNN_LSTM(self.img_channels, self.img_width, self.img_height, self.max_nb_char, self.nb_classes) # 生成CNN的架构
        self.model.load_weights(self.weights_file_path) # 读取训练好的模型


def load_data(img_vals, width, height, channels):
    x = []
    for img_val in img_vals:
       x.append(load_img(img_val, width, height, channels))
    x = np.array(x)
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
    model = predictor.model
    char_set = predictor.char_set
    # predict result
    keys = post_vals.keys()
    img_vals = post_vals.values()
    X_test = load_data(img_vals, img_width, img_height, img_channels)
    predictions = pred(model, X_test, char_set)
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
    return json.dumps(res, ensure_ascii=False)