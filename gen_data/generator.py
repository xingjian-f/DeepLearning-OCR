#coding:utf-8
import random
import os
import string
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np 
from scipy.ndimage.morphology import grey_dilation, grey_erosion
from PIL import ImageFilter
from skimage.util import random_noise, img_as_float


def randRGB():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def captcha_draw(label, fonts, dir_path, pic_id):
    # width, height = 512, 48
    # size_cha = random.randint(24, 48) # 字符大小
    # derx = random.randint(0, 16)
    # im = Image.new(mode='L', size=(width, height), color='white') # color 背景颜色，size 图片大小
    # drawer = ImageDraw.Draw(im)
    # font = ImageFont.truetype(random.choice(fonts), size_cha)
    # drawer.text(xy=(derx, 0), text=label, font=font, fill='black') #text 内容，font 字体（包括大小）
    # # im.show()
    # write2file(dir_path, label, im)

    width, height = 32, 32
    size_cha = random.randint(16, 28) # 字符大小
    derx = random.randint(0, max(width-size_cha-10, 0))
    dery = random.randint(0, max(height-size_cha-10, 0))
    im = Image.new(mode='L', size=(width, height), color='white') # color 背景颜色，size 图片大小
    drawer = ImageDraw.Draw(im)
    font = ImageFont.truetype(random.choice(fonts), size_cha)

    drawer.text(xy=(derx, dery), text=label, font=font, fill='black') #text 内容，font 字体（包括大小）
    # if label != ' ' and (img_as_float(im) == np.ones((48, 48))).all():
    #     # in case the label is not in this font, then the image will be all white
    #     return 0
    im = im.convert('RGBA')
    max_angle = 45 # to be tuned
    angle = random.randint(-max_angle, max_angle)
    im = im.rotate(angle, Image.BILINEAR, expand=0)
    fff = Image.new('RGBA', im.size, (255,)*4)
    im = Image.composite(im, fff, im)
    # if random.random() < 0.5:
    #     im = Image.fromarray(grey_erosion(im, size=(2, 2))) # erosion
    # if random.random() < 0.5:
    #     im = Image.fromarray((random_noise(img_as_float(im), mode='s&p')*255).astype(np.uint8))
    # im = im.filter(ImageFilter.GaussianBlur(radius=random.random()))
    # im.show()
    write2file(dir_path, label, im, pic_id)
    return 1

    
def write2file(dir_path, label, im, pic_id):
    img_name = str(pic_id) + '.jpg'
    img_path = dir_path + img_name
    label_path = dir_path + 'label.txt'
    with open(label_path, 'a') as f:
        f.write((label+'\n').encode('utf-8')) # 在label文件末尾添加新图片的text内容
    print img_path
    im.save(img_path)


if __name__ == "__main__":
    # font_dir = 'fonts/english/'
    # font_paths = map(lambda x: font_dir+x, os.listdir(font_dir))
    # cha_set = string.digits + string.letters + string.punctuation
    # cnt = 50000
    # while cnt > 0:
    #     label = []
    #     for i in range(random.randint(1,18)):
    #         label.append(random.choice(cha_set))
    #     label = ''.join(label)
    #     captcha_draw(label, font_paths, 'asc_seq_1/')
    #     cnt -= 1

    font_dir = 'fonts/chinese/'
    font_paths = map(lambda x: font_dir+x, os.listdir(font_dir))
    chinese_set = open('chinese_6500.txt').readline().decode('utf-8').strip('\n\r')
    eng_set = string.letters + string.digits + string.punctuation
    chi_punctuation = u'。， '
    # cha_set = list(set(chinese_set + eng_set + chi_punctuation))
    cha_set = list(set(chinese_set))
    weight = json.loads(open('weight').readline().decode('utf-8'))
    choice_p = [weight[i] for i in cha_set]
    img_dir = 'chi_rotate_1000000/'
    if os.path.exists(img_dir) == False: # 如果文件夹不存在，则创建对应的文件夹
        os.makedirs(img_dir)
        pic_id = 1
    else:
        pic_id = len(os.listdir(img_dir))


    cnt = 1000000
    for i in range(cnt):
        # label = np.random.choice(cha_set, p=choice_p)
        label = np.random.choice(cha_set)
        if len(label) != 1:
            continue
        pic_id += captcha_draw(label, font_paths, img_dir, pic_id)