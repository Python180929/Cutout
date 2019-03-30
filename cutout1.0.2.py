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
        self.ImageCutting(image)

    # 将护照上的照片裁剪下来
    def ImageCutting(self, image):
        h, w = image.shape[:2]
        # print(h, w)
        # 将护照上右面有人体照片的部分裁掉
        img = image[0:h, 0:int(w / 2)]
        # cv2.imshow('img', img)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转换灰色
        # OpenCV人脸识别分类器
        classifier = cv.CascadeClassifier("face.xml")
        color = (0, 255, 0)  # 定义绘制颜色
        # 调用识别人脸
        faceRects = classifier.detectMultiScale(gray, scaleFactor=1.2,minNeighbors=3, minSize=(32, 32))
        # 大于0则检测到人脸
        if len(faceRects):
            # 单独框出每一张人脸
            for x, y, w1, h1 in faceRects:
                # 绘制出人脸，并找到对应的像素点
                # cv2.rectangle(img, (x, y), (x + h1, y + w1), color, 2)
                # print(x, y, w1, h1)
                # img = img[y:y+h1, x:x+w1]
                if (w1 * h1) > (150 * 150):
                    a = y - h1 / 2
                    b = x - w1 / 2
                    if a < 0:
                        a = 0
                    elif b < 0:
                        b = 0
                    img = img[int(a):int(y - h1 / 2 + 2 * h1), int(b):int(x - w1 / 2 + 2 * w1)]
        self.ImgPreprocessing(img)

    # 图片处理
    def ImgPreprocessing(self, image):
        # image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        # print(image)
        # 获取图片的宽和高
        h, w = image.shape[:2]
        # 将所有照片放大到711*572
        image1 = cv.resize(image, None, fx=572/w, fy=711/h)
        # print(image1.shape)
        # 图片打码
        mask = np.zeros((711, 572), np.uint8)
        # 图像分割
        cv.grabCut(image1, mask, self.rect, self.bgdModel,
                   self.fgModel,10, cv.GC_INIT_WITH_RECT)
        # 将图片的背景换位白色
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        image2 = image1 * mask2[:, :, None]
        image2 += 255 * (1 - cv.cvtColor(mask2, cv.COLOR_GRAY2BGR))
        self.CutImage(image2)

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
        if len(sys.argv) != 2:
            print('''
                argv is error!
                run as
                python cutout.py Url
                ''')
            return
        params = sys.argv[-1]
	for param in params:
            self.ReadImage(param)
            # param = "http://qiniuyun.mobtop.com.cn/png1547440855_94885421.png"
            # param = 'http://qiniuyun.mobtop.com.cn/png1548036365_24151224.png'
            # param = "http://www.mobtop.com.cn/Uploads/letter/332/2018-12-04/5c06009f53988.jpg"

if __name__ == "__main__":
    cutout = Cutout()
    cutout.WorkOn()
