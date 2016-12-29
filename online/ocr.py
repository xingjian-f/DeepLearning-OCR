# coding:utf-8
import sys
sys.path.append('../')
import json
import numpy as np
from StringIO import StringIO
from util import load_img
from base64 import b64decode
import time

def load_data(img_vals, width, height, channels):
    x = []
    for img_val in img_vals:
        x.append(load_img(img_val, width, height, channels))
    x = np.asarray(x)
    x /= 255 # normalized
    return x


def predict(predictor, post_vals, types):
    t1 = time.time()
    width = predictor.img_width
    height = predictor.img_height
    channels = predictor.img_channels
    # predict result
    keys = post_vals.keys()
    img_vals = post_vals.values()
    if len(keys) == 0:
    	return False    
    if types == 'stringio':
        img_vals = map(b64decode, img_vals)
        img_vals = map(StringIO, img_vals)
    # print types
    # print keys
    # print img_vals
    X = load_data(img_vals, width, height, channels)
    predictions = predictor.pred(X)
    probs = predictor.get_prob()
    # format reply
    res = {}
    for i, expr in enumerate(predictions):
        res[keys[i]] = expr # only char
        # res[keys[i]] = '%s(%.2f)' % (expr, probs[i]) # charactor and it's probability
    # print res, len(res)
    print time.time() - t1, len(res)
    return json.dumps(res, ensure_ascii=False) # utf-8 output