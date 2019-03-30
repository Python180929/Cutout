#!/usr/bin/env python3
#coding=utf-8
'''
name : ZhouLiang
email : Brookzhoul@163.com
data : 2019-1-14
company :http://www.dltxsoft.com/
project : Cutout
env : python3.7.1
'''

import cv2 as cv
import numpy as np
import os
import requests
import sys
import json


class Cutout:

    def __init__(self):
        self.bgdModel = np.zeros((1, 65), np.float64)
        self.fgModel = np.zeros((1, 65), np.float64)
        self.rect = (20, 10, 515, 711)

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
        classifier = cv.CascadeClassifier("EditImage/face.xml")
        color = (0, 255, 0)  # 定义绘制颜色
        # 调用识别人脸
        faceRects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))
        # 大于0则检测到人脸
        if len(faceRects):  # 大于0则检测到人脸
            s = sorted([(x, y, w, h) for x, y, w, h in faceRects], key=lambda x: x[2] * x[3], reverse=True)[0]
            x = s[0]
            y = s[1]
            w = s[2]
            h = s[3]
            # cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            '''
            # for x, y, w, h in faceRects: # 单独框出每一张人脸
            #     # 框出人脸
            #     cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)
                # print(x, y, w, h)
                # img = img[y:y+h, x:x+w]
            '''
            a = y - h / 2
            b = x - w / 3
            if a < 0:
                a = 0
            elif b < 0:
                b = 0
            img = img[int(a):int(a + 2 * h), int(b):int(b + 5 * w / 3)]
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
                   self.fgModel, self.num, cv.GC_INIT_WITH_RECT)
        # 将图片的背景换位白色
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        image2 = image1 * mask2[:, :, None]
        image2 += 255 * (1 - cv.cvtColor(mask2, cv.COLOR_GRAY2BGR))
        self.CutImage(image2)

    # 图片大小调整
    def CutImage(self,image):
        # 图片裁剪
        image = image[10:620, 40:527]
        # 图片缩放
        image = cv.resize(image, None, fx=213/470, fy=296/620)
        self.TwoImage(image)

    # 将图片做二次处理
    def TwoImage(self, image):
        h, w = image.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgModel = np.zeros((1, 65), np.float64)
        rect = (32, 32, w, h)
        cv.grabCut(image, mask, rect, bgdModel, fgModel, 5, cv.GC_INIT_WITH_RECT)
        # 将图片的背景换位白色
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        image2 = image * mask2[:, :, None]
        image2 += 255 * (1 - cv.cvtColor(mask2, cv.COLOR_GRAY2BGR))
        # print(image2.shape)
        self.SaveImage(image2)

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
        with open("EditImage/ImageFile.log", "a", encoding="utf8") as f:
            f.write(res.json() + "\n")


    # 主函数
    def WorkOn(self):
        # if len(sys.argv) < 2:
        #     print('''
        #         argv is error!
        #         run as
        #         python3 cutout.py 图片的URL地址(数组)!
        #         ''')
        #     return
        # params_json = sys.argv[1]
        # print(params_json)
        # self.num = int(sys.argv[2]) if len(sys.argv) >= 3 else 5
        # params = json.loads(params_json)
        # print(params)
        # for param in params:
        #     with open("EditImage/ImageFile.log", "a", encoding="utf8") as f:
        #         f.write(param + ", ")
        #     self.ReadImage(param)

        # param = "http://cp.lettours.com/Uploads/letter/154/2018-05-10/5af3b42c1b4e6.jpg"
        # param = "http://qiniuyun.mobtop.com.cn/jpg1548062714_14799978.jpg"
        # param = "http://qiniuyun.mobtop.com.cn/png1548317090_77043396.png"
        # param = "http://qiniuyun.mobtop.com.cn/jpg1548302342_87794741.jpg"
        # param = "http://qiniuyun.mobtop.com.cn/jpg1548302159_53494974.jpg"
        # param = "http://qiniuyun.mobtop.com.cn/png1548144003_92555017.png"
        # param = "http://qiniuyun.mobtop.com.cn/jpg1548379410_73525907.jpg"
        param = 'http://qiniuyun.mobtop.com.cn/jpg1550888044_84912805.jpg'

        self.num = 5
        self.ReadImage(param)
        cv.waitKey()
        cv.destroyAllWindows()

if __name__ == "__main__":
    cutout = Cutout()
    cutout.WorkOn()


# python cutout.py [\"http://qiniuyun.mobtop.com.cn/jpg1550888044_84912805.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1550900113_81493860.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1550901609_32635002.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1550903063_52173019.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1550903270_15689781.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1550904391_53349724.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1550904633_32492791.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1551016531_53163935.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1551016768_19314111.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1551017110_85228245.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1551017285_96715561.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1551017466_50667231.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1551017597_10828535.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1551019090_50107722.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1551022302_63851103.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1551060440_97659614.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1551061334_27534643.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1551061334_27534643.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1551061334_27534643.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1551061610_68236385.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1551062820_45339985.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1551062883_97997261.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1551063172_56432206.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1551071829_76192125.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1551071829_76192125.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1551073821_37059781.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1551075485_53014198.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1551075485_53014198.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1551187889_75962810.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1551189862_38034043.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1551190417_95837405.jpg\",\"http://qiniuyun.mobtop.com.cn/jpg1551193470_58075264.jpg\"]