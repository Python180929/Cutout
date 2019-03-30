#!/usr/bin/env python3
# coding=utf-8
'''
name : ZhouLiang
email : Brookzhoul@163.com
data : 2019-1-14
company :http://www.dltxsoft.com/
project : Cutout
env : python3.7.1
'''

import os
import sys
import json
import base64
import requests
import cv2 as cv
import numpy as np
from random import choice
from PIL import Image
from aip import AipBodyAnalysis


class Cutout:

    def __init__(self):
        """ 你的 APPID AK SK """
        # self.APP_ID = '15883042'
        # self.API_KEY = 'BEzFt5rPE1rGFpQOwS5RD5Vu'
        # self.SECRET_KEY = 'a9UmtZHTQrYhTR85X7iI3ofC4WRGbcVm'
        self.APPID = [
            ['15883042', 'BEzFt5rPE1rGFpQOwS5RD5Vu', 'a9UmtZHTQrYhTR85X7iI3ofC4WRGbcVm'],
            ['15769910', 'ebox3CwFgmny0SqnLYrqdqV4', '88ulxca4dopcCIky5s5xGbcWf9HbjxZR'],
            ['15889978', '1Q8odl1CoCX8tSg9cotXhNxw', 'LaGKXvpspauOG1kvFddZRSDbFEirGaRX'],
            ['15889994', 'UB5FU0I043MLzbU99D3acxiL', 'Wi00zDPstRhWLS6kYqGrb4jMfnElC2Mb'],
            ['15890013', 'r8hzUNs3mtk5hVMEh5XG5LTg', 'M8Lz4YprN3namINXYtcqyigT2F2xjWZW'],
            ['15890027', 'heHy7mSLw33V27sBC5hSmPyD', 'lnshWA0zYhmRwH8eOohD09GGPIwfCjvC'],
            ['15890047', 'Uol0vKDSTlyFwOGMPLq3PmYD', '26Nn4meyqjDNmN2SrnXCqYuFgq6U8fxN'],
            ['15890058', 'iabd0k1GRIMzhPlkwaPA40XI', 'p45WBYOAG3KU0sMV85ncEYE1khbFkC5M'],
            ['15890203', 'vi6tmfHSCdqrKwyoAkkyf4Y1', 'j5tWBBi66a3WopGF9xKB3GlUTrEddfoL'],
            ['15890219', '0qXTxNZzRGphljcpTnPSAKrp', 's4US8xcGMb4GG0drH4HbUShOvTGSigrY'],
            ['15890232', '9Hzgs27SaHndynvfOtd5qDg4', 'WOXwGlo7UWNkGgUxfThi2ILZ2SGNNwr3'],
            ['15890240', 'EjqLH1mvMGI09pnA17RLGkYf', 'o9eflr5ImDtOhonyKfw9OzXsBFoSrngf'],
            ['15890276', '7hds5DnzZe5ghYv7d1HnbKz9', 'vAWtos72r5XpBf3vox3ytnEa6FHummTb']
        ]
        self.bgdModel = np.zeros((1, 65), np.float64)
        self.fgModel = np.zeros((1, 65), np.float64)
        self.rect = (20, 10, 515, 711)

    # 获取图片
    def ReadImage(self, url):
        resp = requests.get(url)
        # 获取图片的二进制文件
        image = resp.content
        self.humanDetection(image)

    # 调用百度接口实现人像分割
    def humanDetection(self, image):
        try:
            APPID = choice(self.APPID)
            client = AipBodyAnalysis(APPID[0], APPID[1], APPID[2])
            res = client.bodySeg(image)
            rsp = res['foreground']
            image = base64.b64decode(rsp)
        except Exception:
            pass
        finally:
            image = np.asarray(bytearray(image), dtype="uint8")
            image = cv.imdecode(image, cv.IMREAD_COLOR)
            # cv.imshow('sdfsa', image)
            self.ImageCutting(image)

    # 将护照上的照片裁剪下来
    def ImageCutting(self, image):
        # OpenCV人脸识别分类器
        classifier = cv.CascadeClassifier("EditImage/face.xml")
        # 定义绘制颜色
        color = (0, 255, 0)
        # 调用识别人脸
        faceRects = classifier.detectMultiScale(image, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))
        # 大于0则检测到人脸
        if len(faceRects):
            s = sorted([(x, y, w, h) for x, y, w, h in faceRects], key=lambda x: x[2] * x[3], reverse=True)[0]
            x = s[0]
            y = s[1]
            w = s[2]
            h = s[3]
            # 单独框出每一张人脸
            # cv.rectangle(image, (x, y), (x + h, y + w), color, 2)
            # cv.imshow('dsfsad', image)
            a = y - h / 2
            b = x - w / 3
            if a < 0:
                a = 0
            if b < 0:
                b = 0
            image = image[int(a):int(a + 15 / 8 * h), int(b):int(b + 5 * w / 3)]
            self.CutImage(image)

    # 图片大小调整
    def CutImage(self, image):
        # 获取图片的长和宽
        h, w = image.shape[:2]
        # 图片缩放
        image = cv.resize(image, None, fx=413 / w, fy=626 / h)
        # cv.imshow('image', image)
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
        with open("EditImage/ImageFile.log", "a", encoding="utf8") as f:
            f.write('<img src="'+res.json()+'"'+' style="width:200px">'  + "\n")

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
        # self.num = int(sys.argv[2]) if len(sys.argv) >= 3 else 5
        # params = json.loads(params_json)
        # for param in params:
        #     with open("EditImage/ImageFile.log", "a", encoding="utf8") as f:
        #         f.write('<img src="'+param +'"'+' style="width:200px"> ')
        #     self.ReadImage(param)

        # param = "http://cp.lettours.com/Uploads/letter/154/2018-05-10/5af3b42c1b4e6.jpg"
        # param = "http://qiniuyun.mobtop.com.cn/jpg1548062714_14799978.jpg"
        # param = "http://qiniuyun.mobtop.com.cn/png1548317090_77043396.png"
        # param = "http://qiniuyun.mobtop.com.cn/jpg1548302342_87794741.jpg"
        # param = "http://qiniuyun.mobtop.com.cn/jpg1548302159_53494974.jpg"
        # param = "http://qiniuyun.mobtop.com.cn/png1548144003_92555017.png"
        # param = "http://qiniuyun.mobtop.com.cn/jpg1548379410_73525907.jpg"
        # param = 'http://qiniuyun.mobtop.com.cn/jpg1550888044_84912805.jpg'
        # param = "http://qiniuyun.mobtop.com.cn/jpg1551017466_50667231.jpg"
        # param = 'http://qiniuyun.mobtop.com.cn/jpg1550900113_81493860.jpg'
        # param = 'http://qiniuyun.mobtop.com.cn/jpg1553843577_36173729.jpg'
        # param = 'http://qiniuyun.mobtop.com.cn/jpg1551189862_38034043.jpg'
        # param ='http://qiniuyun.mobtop.com.cn/jpg1551190417_95837405.jpg'
        # param = 'http://qiniuyun.mobtop.com.cn/jpg1551060440_97659614.jpg'
        # param = 'http://qiniuyun.mobtop.com.cn/jpg1553844231_19429541.jpg'
        # param = "http://qiniuyun.mobtop.com.cn/jpg1553847472_47599143.jpg"
        # param = 'http://qiniuyun.mobtop.com.cn/jpg1553855482_74182484.jpg'
        # param = "http://qiniuyun.mobtop.com.cn/jpg1553842497_55091547.jpg"
        # param = "http://qiniuyun.mobtop.com.cn/jpg1553908428_65917470.jpg"
        # param = "http://qiniuyun.mobtop.com.cn/jpg1553908531_17516434.jpg"
        # param = "http://qiniuyun.mobtop.com.cn/jpg1553909251_23835384.jpg"
        # param = "http://qiniuyun.mobtop.com.cn/jpg1553917060_27281556.jpg"
        # param = "http://qiniuyun.mobtop.com.cn/jpg1553917131_71318349.jpg"
        # param = "http://qiniuyun.mobtop.com.cn/jpg1553917161_20463123.jpg"
        param = "https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1553932484560&di=721f5365277d0002a0ccb589dc3d6e44&imgtype=0&src=http%3A%2F%2Fgss0.baidu.com%2F7Po3dSag_xI4khGko9WTAnF6hhy%2Fzhidao%2Fpic%2Fitem%2F6609c93d70cf3bc7f8f9af3cd300baa1cc112a60.jpg"


        self.ReadImage(param)
        cv.waitKey()
        cv.destroyAllWindows()


if __name__ == "__main__":
    cutout = Cutout()
    cutout.WorkOn()
