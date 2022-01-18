'''
Function:
    拼马赛克图片
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import cv2
import glob
import numpy as np
from tqdm import tqdm
from itertools import product
from ..base import BaseBeautifier


'''拼马赛克图片'''
class PhotomosaicBeautifier(BaseBeautifier):
    def __init__(self, block_size=15, src_images_dir=None, **kwargs):
        super(PhotomosaicBeautifier, self).__init__(**kwargs)
        self.block_size = block_size
        self.src_images_dir = src_images_dir
        self.src_images, self.avg_colors = self.ReadSourceImages()
    '''迭代图片'''
    def iterimage(self, image):
        output_image = np.zeros(image.shape, np.uint8)
        src_images, avg_colors = self.src_images, self.avg_colors
        for i, j in tqdm(product(range(int(image.shape[1]/self.block_size)), range(int(image.shape[0]/self.block_size)))):
            block = image[j*self.block_size: (j+1)*self.block_size, i*self.block_size: (i+1)*self.block_size, :]
            avg_color = np.sum(np.sum(block, axis=0), axis=0) / (self.block_size * self.block_size)
            distances = np.linalg.norm(avg_color - avg_colors, axis=1)
            idx = np.argmin(distances)
            output_image[j*self.block_size: (j+1)*self.block_size, i*self.block_size: (i+1)*self.block_size, :] = src_images[idx]
        return output_image
    '''读取所有源图片并计算对应的颜色平均值'''
    def ReadSourceImages(self):
        src_images, avg_colors = [], []
        for path in tqdm(glob.glob("{}/*.jpg".format(self.src_images_dir))):
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image.shape[-1] != 3: continue
            image = cv2.resize(image, (self.block_size, self.block_size))
            avg_color = np.sum(np.sum(image, axis=0), axis=0) / (self.block_size * self.block_size)
            src_images.append(image)
            avg_colors.append(avg_color)
        return src_images, np.array(avg_colors)