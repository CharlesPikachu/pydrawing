'''
Function:
    简单的照片矫正
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import cv2
import numpy as np
from ..base import BaseBeautifier
from imutils.perspective import four_point_transform


'''简单的照片矫正'''
class PhotocorrectionBeautifier(BaseBeautifier):
    def __init__(self, epsilon_factor=0.08, canny_boundaries=[100, 200], use_preprocess=False, **kwargs):
        super(PhotocorrectionBeautifier, self).__init__(**kwargs)
        self.epsilon_factor = epsilon_factor
        self.canny_boundaries = canny_boundaries
        self.use_preprocess = use_preprocess
    '''迭代图片'''
    def iterimage(self, image):
        # 预处理
        if self.use_preprocess:
            image_edge = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_edge = cv2.GaussianBlur(image_edge, (5, 5), 0)
            image_edge = cv2.dilate(image_edge, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        else:
            image_edge = image.copy()
        image_edge = cv2.Canny(image_edge, self.canny_boundaries[0], self.canny_boundaries[1], 3)
        # 找到最大轮廓
        cnts = cv2.findContours(image_edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]
        if len(cnts) < 1: return image
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for cnt in cnts:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, self.epsilon_factor * peri, True)
            if len(approx) == 4: break
        if len(approx) != 4: return image
        # 矫正
        image_processed = four_point_transform(image, approx.reshape(4, 2))
        # 返回
        return image_processed