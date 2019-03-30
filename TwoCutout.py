#!/usr/bin/env python3
# coding=utf-8
'''
name : ZhouLiang
email : Brookzhoul@163.com
data : 2019-2-23
company :http://www.dltxsoft.com/
project : TwoCutout
env : python3.6
'''

import cv2 as cv
import numpy as np
import requests
import sys
import os
import json


class TwoCutout:

    def __init__(self):
        self.path = "./EditImage/TwoImageFile.log"
        self.bgdModel = np.zeros((1, 65), np.float64)
        self.fgModel = np.zeros((1, 65), np.float64)

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
        # 图片打码
        mask = np.zeros((h, w), np.uint8)
        # 图像分割
        rect = (10, 10, w, h)
        cv.grabCut(image, mask, rect, self.bgdModel,
                   self.fgModel, 5, cv.GC_INIT_WITH_RECT)
        # 将图片的背景换位白色
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        image = image * mask2[:, :, None]
        image += 255 * (1 - cv.cvtColor(mask2, cv.COLOR_GRAY2BGR))
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
        # 将照片的URL地址存入到本地文件
        with open("EditImage/TwoImageFile.log", "a", encoding="utf8") as f:
            f.write(res.json() + "\n\n")

    # 主函数
    def WorkOn(self):
        if len(sys.argv) != 2:
            print('''
                argv is error!
                run as
                python3 TwoCutout.py 照片URL地址(json格式)!
                ''')
            return
        params_json = sys.argv[1]
        params = json.loads(params_json)
        if not os.path.exists(self.path):
            # 调用系统命令行来创建文件
            os.system(r"touch {}".format(self.path))
        for param in params:
            # 将照片的URL地址存入到本地文件
            with open(self.path, "a", encoding="utf8") as f:
                f.write(param + ",  ")
            self.ReadImage(param)
        # param = "http://qiniuyun.mobtop.com.cn/png1547440855_94885421.png"
        # param = "http://qiniuyun.mobtop.com.cn/png1550886591_72577277.png"
        # param = "http://qiniuyun.mobtop.com.cn/png1550887509_71418286.png"
        # param = "http://qiniuyun.mobtop.com.cn/png1550895478_60576230.png"
        # prams = "http://qiniuyun.mobtop.com.cn/png1550895609_51437528.png"
        # self.ReadImage(param)


if __name__ == "__main__":
    cutout = TwoCutout()
    cutout.WorkOn()
