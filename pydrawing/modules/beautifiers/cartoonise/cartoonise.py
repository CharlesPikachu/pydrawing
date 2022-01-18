'''
Function:
    图像卡通化
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import cv2
import numpy as np
from ..base import BaseBeautifier


'''图像卡通化'''
class CartooniseBeautifier(BaseBeautifier):
    def __init__(self, mode='rgb', **kwargs):
        super(CartooniseBeautifier, self).__init__(**kwargs)
        assert mode in ['rgb', 'hsv']
        self.mode = mode
    '''迭代图片'''
    def iterimage(self, image):
        if self.mode == 'rgb':
            return self.processinrgb(image)
        elif self.mode == 'hsv':
            return self.processinhsv(image)
    '''在RGB空间操作'''
    def processinrgb(self, image):
        # Step1: 利用双边滤波器对原图像进行保边去噪处理
        # --下采样
        image_bilateral = image
        for _ in range(2):
            image_bilateral = cv2.pyrDown(image_bilateral)
        # --进行多次的双边滤波
        for _ in range(7):
            image_bilateral = cv2.bilateralFilter(image_bilateral, d=9, sigmaColor=9, sigmaSpace=7)
        # --上采样
        for _ in range(2):
            image_bilateral = cv2.pyrUp(image_bilateral)
        # Step2: 将步骤一中获得的图像灰度化后，使用中值滤波器去噪
        image_gray = cv2.cvtColor(image_bilateral, cv2.COLOR_RGB2GRAY)
        image_median = cv2.medianBlur(image_gray, 7)
        # Step3: 对步骤二中获得的图像使用自适应阈值从而获得原图像的轮廓
        image_edge = cv2.adaptiveThreshold(image_median, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)
        image_edge = cv2.cvtColor(image_edge, cv2.COLOR_GRAY2RGB)
        # Step4: 将步骤一中获得的图像与步骤三中获得的图像轮廓合并即可实现将照片变为卡通图片的效果了
        image_cartoon = cv2.bitwise_and(image_bilateral, image_edge)
        # 返回
        return image_cartoon
    '''在HSV空间操作'''
    def processinhsv(self, image):
        # Step1: 图像BGR空间转HSV空间, 在HSV空间进行直方图均衡化, 中值滤波和形态学变换
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image_hsv)
        # --直方图均衡化
        v = cv2.equalizeHist(v)
        image_hsv = cv2.merge((h, s, v))
        # --中值滤波
        image_hsv = cv2.medianBlur(image_hsv, 7)
        # --形态学变换-开/闭运算
        kernel = np.ones((5, 5), np.uint8)
        image_hsv = cv2.morphologyEx(image_hsv, cv2.MORPH_CLOSE, kernel, iterations=2)
        # --中值滤波
        image_hsv = cv2.medianBlur(image_hsv, 7)
        # Step2: 对步骤一中获得的图像使用自适应阈值从而获得原图像的轮廓
        image_mask = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
        image_mask = cv2.cvtColor(image_mask, cv2.COLOR_RGB2GRAY)
        image_mask = cv2.medianBlur(image_mask, 7)
        image_edge = cv2.adaptiveThreshold(image_mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)
        image_edge = cv2.cvtColor(image_edge, cv2.COLOR_GRAY2RGB)
        # Step3: 将步骤二中获得的图像轮廓与原图像合并即可实现将照片变为卡通图片的效果了
        image_cartoon = cv2.bitwise_and(image, image_edge)
        # 返回
        return image_cartoon