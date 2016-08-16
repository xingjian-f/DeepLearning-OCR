# coding:utf-8
import sys
sys.path.append('/home/feixingjian/DeepLearning-OCR/')
import json
import numpy as np
from StringIO import StringIO
from util import load_img
from base64 import b64decode

def load_data(img_vals, width, height, channels):
    x = []
    for img_val in img_vals:
        x.append(load_img(img_val, width, height, channels))
    x = np.asarray(x)
    x /= 255 # normalized
    return x


def predict(predictor, post_vals, types):
    width = predictor.img_width
    height = predictor.img_height
    channels = predictor.img_channels
    # predict result
    keys = post_vals.keys()
    img_vals = post_vals.values()    
    if types == 'stringio':
        img_vals = map(b64decode, img_vals)
        img_vals = map(StringIO, img_vals)
    # print types
    # print keys
    # print img_vals
    X = load_data(img_vals, width, height, channels)
    predictions = predictor.pred(X)
    # format reply
    res = {}
    for i, expr in enumerate(predictions):
        res[keys[i]] = expr
    print res, len(res)
    return json.dumps(res, ensure_ascii=False) # utf-8 output