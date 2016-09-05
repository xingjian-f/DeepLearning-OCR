#coding:utf-8
import random
import os
import string
from PIL import Image, ImageDraw, ImageFont
import numpy as np 
from scipy.ndimage.morphology import grey_dilation, grey_erosion
from PIL import ImageFilter


def randRGB():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def captcha_draw(label, fonts, dir_path):
    # width, height = 512, 48
    # size_cha = random.randint(24, 48) # 字符大小
    # derx = random.randint(0, 16)
    # im = Image.new(mode='L', size=(width, height), color='white') # color 背景颜色，size 图片大小
    # drawer = ImageDraw.Draw(im)
    # font = ImageFont.truetype(random.choice(fonts), size_cha)
    # drawer.text(xy=(derx, 0), text=label, font=font, fill='black') #text 内容，font 字体（包括大小）
    # # im.show()
    # write2file(dir_path, label, im)

    width, height = 48, 48
    size_cha = random.randint(20, 48) # 字符大小
    derx = random.randint(0, max(width-size_cha-10, 0))
    dery = random.randint(0, max(height-size_cha-10, 0))
    im = Image.new(mode='L', size=(width, height), color='white') # color 背景颜色，size 图片大小
    drawer = ImageDraw.Draw(im)
    font = ImageFont.truetype(random.choice(fonts), size_cha)
    drawer.text(xy=(derx, dery), text=label, font=font, fill='black') #text 内容，font 字体（包括大小）
    if random.random() < 0.5:
        im = Image.fromarray(grey_erosion(im, size=(2, 2))) # erosion
    im = im.filter(ImageFilter.GaussianBlur(radius=random.random()))
    # im.show()
    write2file(dir_path, label, im)

    
def write2file(dir_path, label, im):
    if os.path.exists(dir_path) == False: # 如果文件夹不存在，则创建对应的文件夹
        os.makedirs(dir_path)
        pic_id = 1
    else:
        pic_id = len(os.listdir(dir_path))

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
    chinese_set = open('common3000_chi.txt').readline().decode('utf-8').strip('\n\r')
    eng_set = string.letters + string.digits + string.punctuation
    chi_punctuation = u'，。；'
    cha_set = chinese_set + eng_set + chi_punctuation
    cnt = 500000
    while cnt > 0:
        label = random.choice(cha_set)
        captcha_draw(label, font_paths, 'single_cha_500000/')
        cnt -= 1