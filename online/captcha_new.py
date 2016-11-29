# coding:utf-8
import sys
sys.path.append('/home/feixingjian/DeepLearning-OCR/')
import json
import numpy as np 
from util import load_img


def parse_expr(expr, pinyin):
    if expr[-2:] == u'等于' or expr[-3:] in [u'等于？', u'等于几']:
        rep_dict = [
            (u'零', '0'),
            (u'一', '1'),
            (u'二', '2'),
            (u'三', '3'),
            (u'四', '4'),
            (u'五', '5'),
            (u'六', '6'),
            (u'七', '7'),
            (u'八', '8'),
            (u'九', '9'),
            (u'壹', '1'),
            (u'贰', '2'),
            (u'叁', '3'),
            (u'肆', '4'),
            (u'伍', '5'),
            (u'陆', '6'),
            (u'柒', '7'),
            (u'捌', '8'),
            (u'玖', '9'),
            (u'加上', '+'),
            (u'减去', '-'),
            (u'乘以', '*'),
            (u'除以', '/'),
            (u'加', '+'),
            (u'乘', '*'),
            (u'×', '*'),
            (u'x', '*'),
            (u'减', '-'),
            (u'等于几', ''),
            (u'等于？', ''),
            (u'等于', '')
        ]
        for i in rep_dict:
            expr = expr.replace(i[0], i[1])
        try:
            return eval(expr)
        except:
            return False
    elif expr[-6:] == u'的拼音首字母': # ZheJiang's captcha
        ret = []
        # print expr
        for i in range(len(expr)-6):
            char = expr[i]
            if char not in pinyin:
                return False
            ret.append(pinyin[char][0])
        return ''.join(ret)
    else:
        return expr


def load_data(img_vals, width, height, channels):
    x = []
    for img_val in img_vals:
       x.append(load_img(img_val, width, height, channels))
    x = np.asarray(x)
    x /= 255 # normalized
    return x


def predict(predictor, post_vals, pinyin=None):
    width = predictor.img_width
    height = predictor.img_height
    channels = predictor.img_channels
    # predict result
    keys = post_vals.keys()
    img_vals = post_vals.values()
    X = load_data(img_vals, width, height, channels)
    predictions = predictor.pred(X)
    if pinyin == 'prob': # TODO
        predictions = map(lambda x: ','.join(map(str, x)), predictor.pred_probs)
    # print predictor.pred_probs
    # format reply
    res = {}
    for i, expr in enumerate(predictions):
        valid = True
        ans = parse_expr(expr, pinyin)
        if ans is False:
            valid = False
        form = {'valid':valid, 'answer':ans, 'expr':expr}
        res[keys[i]] = form
    if len(res) == 1 and 'file' in res: # in case input data is not in batch form 
        res = res['file']
    # print res, len(res)
    return json.dumps(res, ensure_ascii=False) # utf-8 output