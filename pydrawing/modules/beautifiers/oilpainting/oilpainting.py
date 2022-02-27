'''
Function:
    照片油画化
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import cv2
import random
import numpy as np
from scipy import ndimage
from ..base import BaseBeautifier


'''照片油画化'''
class OilpaintingBeautifier(BaseBeautifier):
    def __init__(self, brush_width=5, palette=0, edge_operator='sobel', **kwargs):
        super(OilpaintingBeautifier, self).__init__(**kwargs)
        assert edge_operator in ['scharr', 'prewitt', 'sobel', 'roberts']
        self.brush_width = brush_width
        self.palette = palette
        self.edge_operator = edge_operator
    '''迭代图片'''
    def iterimage(self, image):
        # 计算图像梯度
        r = 2 * int(image.shape[0] / 50) + 1
        gx, gy = self.getgradient(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (r, r), self.edge_operator)
        gh = np.sqrt(np.sqrt(np.square(gx) + np.square(gy)))
        ga = (np.arctan2(gy, gx) / np.pi) * 180 + 90
        # 画油画的所有位置
        canvas = cv2.medianBlur(image, 11)
        order = self.getdraworder(image.shape[0], image.shape[1], scale=self.brush_width * 2)
        # 画椭圆
        colors = np.array(image, dtype=np.float)
        for i, (y, x) in enumerate(order):
            length = int(round(self.brush_width + self.brush_width * gh[y, x]))
            if self.palette != 0: 
                color = np.array([round(colors[y, x][0] / self.palette) * self.palette + random.randint(-5, 5), \
                                  round(colors[y, x][1] / self.palette) * self.palette + random.randint(-5, 5), \
                                  round(colors[y, x][2] / self.palette) * self.palette + random.randint(-5, 5)], dtype=np.float)
            else:
                color = colors[y, x]
            cv2.ellipse(canvas, (x, y), (length, self.brush_width), ga[y, x], 0, 360, color, -1, cv2.LINE_AA)
        # 返回结果
        return canvas
    '''画油画的所有位置'''
    def getdraworder(self, h, w, scale):
        order = []
        for i in range(0, h, scale):
            for j in range(0, w, scale):
                y = random.randint(-scale // 2, scale // 2) + i
                x = random.randint(-scale // 2, scale // 2) + j
                order.append((y % h, x % w))
        return order
    '''prewitt算子'''
    def prewitt(self, img):
        img_gaussian = cv2.GaussianBlur(img, (3, 3), 0)
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
        img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
        return img_prewittx // 15.36, img_prewitty // 15.36
    '''roberts算子'''
    def roberts(self, img):
        roberts_cross_v = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])
        roberts_cross_h = np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])
        vertical = ndimage.convolve(img, roberts_cross_v)
        horizontal = ndimage.convolve(img, roberts_cross_h)
        return vertical // 50.0, horizontal // 50.0
    '''利用边缘检测算子获得梯度'''
    def getgradient(self, img_o, ksize, edge_operator):
        if edge_operator == 'scharr':
            X = cv2.Scharr(img_o, cv2.CV_32F, 1, 0) / 50.0
            Y = cv2.Scharr(img_o, cv2.CV_32F, 0, 1) / 50.0
        elif edge_operator == 'prewitt':
            X, Y = self.prewitt(img_o)
        elif edge_operator == 'sobel':
            X = cv2.Sobel(img_o, cv2.CV_32F, 1, 0, ksize=5)  / 50.0
            Y = cv2.Sobel(img_o, cv2.CV_32F, 0, 1, ksize=5)  / 50.0
        elif edge_operator == 'roberts':
            X, Y = self.roberts(img_o)
        X = cv2.GaussianBlur(X, ksize, 0)
        Y = cv2.GaussianBlur(Y, ksize, 0)
        return X, Y