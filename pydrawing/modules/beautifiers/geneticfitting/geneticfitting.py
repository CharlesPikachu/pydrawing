'''
Function:
    利用遗传算法画画
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import os
import cv2
import random
import imageio
import numpy as np
from ..base import BaseBeautifier


'''利用遗传算法画画'''
class GeneticFittingBeautifier(BaseBeautifier):
    def __init__(self, **kwargs):
        super(GeneticFittingBeautifier, self).__init__(**kwargs)
    '''迭代图片'''
    def iterimage(self, image):
        pass