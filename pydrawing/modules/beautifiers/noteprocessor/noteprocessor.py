'''
Function:
    手写笔记处理
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import os
import cv2
import numpy as np
from PIL import Image
from ..base import BaseBeautifier
from scipy.cluster.vq import kmeans, vq


'''手写笔记处理'''
class NoteprocessorBeautifier(BaseBeautifier):
    def __init__(self, value_threshold=0.25, sat_threshold=0.20, num_colors=8, sample_fraction=0.05, white_bg=False, saturate=True, **kwargs):
        super(NoteprocessorBeautifier, self).__init__(**kwargs)
        self.num_colors = num_colors
        self.sample_fraction = sample_fraction
        self.value_threshold = value_threshold
        self.sat_threshold = sat_threshold
        self.white_bg = white_bg
        self.saturate = saturate
    '''迭代图片'''
    def iterimage(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sampled_pixels = self.getsampledpixels(image, self.sample_fraction)
        palette = self.getpalette(sampled_pixels)
        labels = self.applypalette(image, palette)
        if self.saturate:
            palette = palette.astype(np.float32)
            pmin = palette.min()
            pmax = palette.max()
            palette = 255 * (palette - pmin) / (pmax - pmin)
            palette = palette.astype(np.uint8)
        if self.white_bg:
            palette = palette.copy()
            palette[0] = (255, 255, 255)
        image_processed = Image.fromarray(labels, 'P')
        image_processed.putpalette(palette.flatten())
        image_processed.save('tmp.png', dpi=(300, 300))
        image_processed = cv2.imread('tmp.png')
        os.remove('tmp.png')
        return image_processed
    '''将调色板应用到给定的图像, 第一步是将所有背景像素设置为背景颜色, 之后是使用最邻近匹配将每个前景色映射到调色板中最接近的一个'''
    def applypalette(self, image, palette):
        bg_color = palette[0]
        fg_mask = self.getfgmask(bg_color, image)
        orig_shape = image.shape
        pixels = image.reshape((-1, 3))
        fg_mask = fg_mask.flatten()
        num_pixels = pixels.shape[0]
        labels = np.zeros(num_pixels, dtype=np.uint8)
        labels[fg_mask], _ = vq(pixels[fg_mask], palette)
        return labels.reshape(orig_shape[:-1])
    '''选取图像中固定百分比的像素, 以随机顺序返回'''
    def getsampledpixels(self, image, sample_fraction):
        pixels = image.reshape((-1, 3))
        num_pixels = pixels.shape[0]
        num_samples = int(num_pixels * sample_fraction)
        idx = np.arange(num_pixels)
        np.random.shuffle(idx)
        return pixels[idx[:num_samples]]
    '''提取采样的RGB值集合的调色板, 调色板第一个条目始终是背景色, 其余的是通过运行K均值聚类从前景像素确定的'''
    def getpalette(self, samples, return_mask=False, kmeans_iter=40):
        bg_color = self.getbgcolor(samples, 6)
        fg_mask = self.getfgmask(bg_color, samples)
        centers, _ = kmeans(samples[fg_mask].astype(np.float32), self.num_colors-1, iter=kmeans_iter)
        palette = np.vstack((bg_color, centers)).astype(np.uint8)
        if not return_mask: return palette
        return palette, fg_mask
    '''通过与背景颜色进行比较来确定一组样本中的每个像素是否为前景, 如果像素的值或饱和度与背景的阈值不同, 则像素被分类为前景像素'''
    def getfgmask(self, bg_color, samples):
        s_bg, v_bg = self.rgbtosv(bg_color)
        s_samples, v_samples = self.rgbtosv(samples)
        s_diff = np.abs(s_bg - s_samples)
        v_diff = np.abs(v_bg - v_samples)
        return ((v_diff >= self.value_threshold) | (s_diff >= self.sat_threshold))
    '''将RGB图像或RGB颜色数组转换为饱和度和数值, 每个都作为单独的32位浮点数组或值返回'''
    def rgbtosv(self, rgb):
        if not isinstance(rgb, np.ndarray): rgb = np.array(rgb)
        axis = len(rgb.shape) - 1
        cmax = rgb.max(axis=axis).astype(np.float32)
        cmin = rgb.min(axis=axis).astype(np.float32)
        delta = cmax - cmin
        saturation = delta.astype(np.float32) / cmax.astype(np.float32)
        saturation = np.where(cmax==0, 0, saturation)
        value = cmax / 255.0
        return saturation, value
    '''从图像或RGB颜色数组中获得背景颜色, 方法为通过将相似的颜色分组为相同的颜色并找到最常见的颜色'''
    def getbgcolor(self, image, bits_per_channel=6):
        assert image.shape[-1] == 3
        image_quantized = self.quantize(image, bits_per_channel).astype(int)
        image_packed = self.packrgb(image_quantized)
        unique, counts = np.unique(image_packed, return_counts=True)
        packed_mode = unique[counts.argmax()]
        return self.unpackrgb(packed_mode)
    '''减少给定图像中RGB三通道的位数'''
    def quantize(self, image, bits_per_channel=6):
        assert image.dtype == np.uint8
        shift = 8 - bits_per_channel
        halfbin = (1 << shift) >> 1
        return ((image.astype(int) >> shift) << shift) + halfbin
    '''将24位RGB三元组打包成一个整数, 参数rgb为元组或者数组'''
    def packrgb(self, rgb):
        orig_shape = None
        if isinstance(rgb, np.ndarray):
            assert rgb.shape[-1] == 3
            orig_shape = rgb.shape[:-1]
        else:
            assert len(rgb) == 3
            rgb = np.array(rgb)
        rgb = rgb.astype(int).reshape((-1, 3))
        packed = (rgb[:, 0] << 16 | rgb[:, 1] << 8 | rgb[:, 2])
        if orig_shape is None: return packed
        return packed.reshape(orig_shape)
    '''将一个整数或整数数组解压缩为一个或多个24位RGB值'''
    def unpackrgb(self, packed):
        orig_shape = None
        if isinstance(packed, np.ndarray):
            assert packed.dtype == int
            orig_shape = packed.shape
            packed = packed.reshape((-1, 1))
        rgb = ((packed >> 16) & 0xff, (packed >> 8) & 0xff, (packed) & 0xff)
        if orig_shape is None: return rgb
        return np.hstack(rgb).reshape(orig_shape + (3,))