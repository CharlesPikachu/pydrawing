'''
Function:
    图像抖音特效画
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import cv2
import copy
import numpy as np
from PIL import Image
from ..base import BaseBeautifier


'''图像抖音特效画'''
class DouyinEffectBeautifier(BaseBeautifier):
    def __init__(self, **kwargs):
        super(DouyinEffectBeautifier, self).__init__(**kwargs)
    '''迭代图片'''
    def iterimage(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        # 提取R
        image_arr_r = copy.deepcopy(image)
        image_arr_r[:, :, 1:3] = 0
        # 提取GB
        image_arr_gb = copy.deepcopy(image)
        image_arr_gb[:, :, 0] = 0
        # 创建画布把图片错开放
        image_r = Image.fromarray(image_arr_r).convert('RGBA')
        image_gb = Image.fromarray(image_arr_gb).convert('RGBA')
        canvas_r = Image.new('RGB', (image.shape[1], image.shape[0]), color=(0, 0, 0))
        canvas_gb = Image.new('RGB', (image.shape[1], image.shape[0]), color=(0, 0, 0))
        canvas_r.paste(image_r, (6, 6), image_r)
        canvas_gb.paste(image_gb, (0, 0), image_gb)
        output_image = np.array(canvas_gb) + np.array(canvas_r)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        # 返回结果
        return output_image