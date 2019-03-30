#!/usr/bin/env python3
#coding=utf-8
'''
name : ZhouLiang
email : Brookzhoul@163.com
data : 2019-1-14
company :http://www.dltxsoft.com/
project : Cutout
env : python3.6
'''

import cv2 as cv
import numpy as np
import requests
import sys

class Cutout:

    def __init__(self):
        self.bgdModel = np.zeros((1, 65), np.float64)
        self.fgModel = np.zeros((1, 65), np.float64)
        self.rect = (20, 20, 515, 711)

    # 获取图片
    def ReadImage(self, url):
        resp = requests.get(url)
        image = np.asarray(bytearray(resp.content), dtype="uint8")
        image = cv.imdecode(image, cv.IMREAD_COLOR)
        self.ImgPreprocessing(image)

    # 图片处理
    def ImgPreprocessing(self, image):
        # 获取图片的宽和高
        h, w = image.shape[:2]
        # 将所有照片放大到711*572
        image = cv.resize(image, None, fx=572/w, fy=711/h)
        # 图片打码
        mask = np.zeros((711, 572), np.uint8)
        # 图像分割
        cv.grabCut(image, mask, self.rect, self.bgdModel,
                   self.fgModel, 5, cv.GC_INIT_WITH_RECT)
        # 将图片的背景换位白色
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        image = image * mask2[:, :, None]
        image += 255 * (1 - cv.cvtColor(mask2, cv.COLOR_GRAY2BGR))
        self.CutImage(image)

    # 图片大小调整
    def CutImage(self,image):
        # 图片裁剪
        image = image[0:620, 57:527]
        # 图片缩放
        image = cv.resize(image, None, fx=354/470, fy=472/620)
        self.SaveImage(image)

    # 保存图片
    def SaveImage(self, image):
        img_encode = cv.imencode('.jpg', image)[1]
        # print(img_encode)
        data_encode = np.array(img_encode)
        str_encode = data_encode.tostring()
        # print(str_encode)
        # # 缓存数据保存到本地
        # with open('img_encode.jpg', 'wb') as f:
        #     f.write(str_encode)
        url = 'http://www.mobtop.com.cn/index.php?s=/Business/Pcapi/insertlogoapi'
        params = {"file": ("a.jpg", str_encode, "image/png")}
        res = requests.post(url, files=params)
        print(res.json())


    # 主函数
    def WorkOn(self):
        # if len(sys.argv) != 2:
        #     print('''
        #         argv is error!
        #         run as
        #         python cutout.py Url
        #         ''')
        #     return
        # param = sys.argv[-1]
        # param = "http://qiniuyun.mobtop.com.cn/png1547440855_94885421.png"
        # param = "http://qiniuyun.mobtop.com.cn/png1550886591_72577277.png"
        param = "http://qiniuyun.mobtop.com.cn/png1550887509_71418286.png"
        self.ReadImage(param)

if __name__ == "__main__":
    cutout = Cutout()
    cutout.WorkOn()

