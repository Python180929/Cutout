# from aip import AipBodyAnalysis
# from PIL import Image
# import base64
#
# """ 你的 APPID AK SK """
# APP_ID = '15883042'
# API_KEY = 'BEzFt5rPE1rGFpQOwS5RD5Vu'
# SECRET_KEY = 'a9UmtZHTQrYhTR85X7iI3ofC4WRGbcVm'
# client = AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY)
#
# """ 读取图片 """
# def get_file_content(filePath):
#     with open(filePath, 'rb') as fp:
#         return fp.read()
#
# image = get_file_content('432.jpg')
# print(image)
#
# """ 调用人像分割 """
# res = client.bodySeg(image)
# rsp = res['foreground']
# result = base64.b64decode(rsp)
# with open('test.png', 'wb') as f:
#     f.write(result)


import cv2 as cv



