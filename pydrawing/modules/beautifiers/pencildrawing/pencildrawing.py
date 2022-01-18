'''
Function:
    复现论文"Combining Sketch and Tone for Pencil Drawing Production"
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import os
import cv2
import math
import numpy as np
from PIL import Image
from scipy import signal
from ..base import BaseBeautifier
from scipy.ndimage import interpolation
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, spdiags


'''图像处理工具'''
class ImageProcessor():
    '''将像素值压缩到[0, 1]'''
    @staticmethod
    def im2double(img):
        if len(img.shape) == 2: return (img - img.min()) / (img.max() - img.min())
        else: return cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    '''拉普拉斯分布'''
    @staticmethod
    def Laplace(x, sigma=9):
        value = (1. / sigma) * math.exp(-(256 - x) / sigma) * (256 - x)
        return value
    '''均匀分布'''
    @staticmethod
    def Uniform(x, ua=105, ub=225):
        value = (1. / (ub - ua)) * (max(x - ua, 0) - max(x - ub, 0))
        return value
    '''高斯分布'''
    @staticmethod
    def Gaussian(x, u=90, sigma=11):
        value = (1. / math.sqrt(2 * math.pi * sigma)) * math.exp(-((x - u) ** 2) / (2 * (sigma ** 2)))
        return value
    '''水平方向拼接'''
    @staticmethod
    def horizontalStitch(img, width):
        img_stitch = img.copy()
        while img_stitch.shape[1] < width:
            window_size = int(round(img.shape[1] / 4.))
            left = img[:, (img.shape[1]-window_size): img.shape[1]]
            right = img[:, :window_size]
            aleft = np.zeros((left.shape[0], window_size))
            aright = np.zeros((left.shape[0], window_size))
            for i in range(window_size):
                aleft[:, i] = left[:, i] * (1 - (i + 1.) / window_size)
                aright[:, i] = right[:, i] * (i + 1.) / window_size
            img_stitch = np.column_stack((img_stitch[:, :(img_stitch.shape[1]-window_size)], aleft+aright, img_stitch[:, window_size: img_stitch.shape[1]]))
        img_stitch = img_stitch[:, :width]
        return img_stitch
    '''垂直方向拼接'''
    @staticmethod
    def verticalStitch(img, height):
        img_stitch = img.copy()
        while img_stitch.shape[0] < height:
            window_size = int(round(img.shape[0] / 4.))
            up = img[(img.shape[0]-window_size): img.shape[0], :]
            down = img[0:window_size, :]
            aup = np.zeros((window_size, up.shape[1]))
            adown = np.zeros((window_size, up.shape[1]))
            for i in range(window_size):
                aup[i, :] = up[i, :] * (1 - (i + 1.) / window_size)
                adown[i, :] = down[i, :] * (i + 1.) / window_size
            img_stitch = np.row_stack((img_stitch[:img_stitch.shape[0]-window_size, :], aup+adown, img_stitch[window_size: img_stitch.shape[0], :]))
        img_stitch = img_stitch[:height, :]
        return img_stitch


'''复现论文"Combining Sketch and Tone for Pencil Drawing Production"'''
class PencilDrawingBeautifier(BaseBeautifier):
    def __init__(self, mode='gray', kernel_size_scale=1/40, stroke_width=1, color_depth=1, weights_color=[62, 30, 5], weights_gray=[76, 22, 2], texture_path=None, **kwargs):
        super(PencilDrawingBeautifier, self).__init__(**kwargs)
        assert mode in ['gray', 'color']
        self.rootdir = os.path.split(os.path.abspath(__file__))[0]
        self.image_processor = ImageProcessor()
        self.mode = mode
        # 铅笔笔画相关参数
        self.kernel_size_scale, self.stroke_width = kernel_size_scale, stroke_width
        # 铅笔色调相关参数
        self.weights_color, self.weights_gray, self.color_depth = weights_color, weights_gray, color_depth
        if (texture_path is None) or (not os.path.exists(texture_path)): self.texture_path = os.path.join(self.rootdir, 'textures/default.jpg')
    '''迭代图片'''
    def iterimage(self, image):
        if self.mode == 'color':
            img = Image.fromarray(image)
            img_ycbcr = img.convert('YCbCr')
            img = np.ndarray((img.size[1], img.size[0], 3), 'u1', img_ycbcr.tobytes())
            img_out = img.copy()
            img_out.flags.writeable = True
            img_out[:, :, 0] = self.__strokeGeneration(img[:, :, 0]) * self.__toneGeneration(img[:, :, 0]) * 255
            img_out = cv2.cvtColor(img_out, cv2.COLOR_YCR_CB2BGR)
        else:
            img = image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_s = self.__strokeGeneration(img)
            img_t = self.__toneGeneration(img)
            img_out = img_s * img_t * 255
        return img_out
    '''铅笔笔画生成'''
    def __strokeGeneration(self, img):
        h, w = img.shape
        kernel_size = int(min(w, h) * self.kernel_size_scale)
        kernel_size += kernel_size % 2
        # 计算梯度，产生幅度
        img_double = self.image_processor.im2double(img)
        dx = np.concatenate((np.abs(img_double[:, :-1]-img_double[:, 1:]), np.zeros((h, 1))), 1)
        dy = np.concatenate((np.abs(img_double[:-1, :]-img_double[1:, :]), np.zeros((1, w))), 0)
        img_gradient = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
        # 选择八个参考方向
        line_segments = np.zeros((kernel_size, kernel_size, 8))
        for i in [0, 1, 2, 7]:
            for x in range(kernel_size):
                y = round((x + 1 - kernel_size / 2) * math.tan(math.pi / 8 * i))
                y = kernel_size / 2 - y
                if y > 0 and y <= kernel_size:
                    line_segments[int(y-1), x, i] = 1
                if i == 7:
                    line_segments[:, :, 3] = np.rot90(line_segments[:, :, 7], -1)
                else:
                    line_segments[:, :, i+4] = np.rot90(line_segments[:, :, i], 1)
        # 获取参考方向的响应图
        response_maps = np.zeros((h, w, 8))
        for i in range(8):
            response_maps[:, :, i] = signal.convolve2d(img_gradient, line_segments[:, :, i], 'same')
        response_maps_maxvalueidx = response_maps.argmax(axis=-1)
        # 通过在所有方向的响应中选择最大值来进行分类
        magnitude_maps = np.zeros_like(response_maps)
        for i in range(8):
            magnitude_maps[:, :, i] = img_gradient * (response_maps_maxvalueidx == i).astype('float')
        # 线条整形
        stroke_maps = np.zeros_like(response_maps)
        for i in range(8):
            stroke_maps[:, :, i] = signal.convolve2d(magnitude_maps[:, :, i], line_segments[:, :, i], 'same')
        stroke_maps = stroke_maps.sum(axis=-1)
        stroke_maps = (stroke_maps - stroke_maps.min()) / (stroke_maps.max() - stroke_maps.min())
        stroke_maps = (1 - stroke_maps) ** self.stroke_width
        return stroke_maps
    '''铅笔色调生成'''
    def __toneGeneration(self, img, mode=None):
        height, width = img.shape
        # 直方图匹配
        img_hist_match = self.__histogramMatching(img, mode) ** self.color_depth
        # 获得纹理
        texture = cv2.imread(self.texture_path)
        texture = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)[99: texture.shape[0]-100, 99: texture.shape[1]-100]
        ratio = 0.2 * min(img.shape[0], img.shape[1]) / float(1024)
        texture = interpolation.zoom(texture, (ratio, ratio))
        texture = self.image_processor.im2double(texture)
        texture = self.image_processor.horizontalStitch(texture, img.shape[1])
        texture = self.image_processor.verticalStitch(texture, img.shape[0])
        size = img.size
        nzmax = 2 * (size-1)
        i = np.zeros((nzmax, 1))
        j = np.zeros((nzmax, 1))
        s = np.zeros((nzmax, 1))
        for m in range(1, nzmax+1):
            i[m-1] = int(math.ceil((m + 0.1) / 2)) - 1
            j[m-1] = int(math.ceil((m - 0.1) / 2)) - 1
            s[m-1] = -2 * (m % 2) + 1
        dx = csr_matrix((s.T[0], (i.T[0], j.T[0])), shape=(size, size))
        nzmax = 2 * (size - img.shape[1])
        i = np.zeros((nzmax, 1))
        j = np.zeros((nzmax, 1))
        s = np.zeros((nzmax, 1))
        for m in range(1, nzmax+1):
            i[m-1, :] = int(math.ceil((m - 1 + 0.1) / 2) + img.shape[1] * (m % 2)) - 1
            j[m-1, :] = math.ceil((m - 0.1) / 2) - 1
            s[m-1, :] = -2 * (m % 2) + 1
        dy = csr_matrix((s.T[0], (i.T[0], j.T[0])), shape=(size, size))
        texture_sparse = spdiags(np.log(np.reshape(texture.T, (1, texture.size), order="f") + 0.01), 0, size, size)
        img_hist_match1d = np.log(np.reshape(img_hist_match.T, (1, img_hist_match.size), order="f").T + 0.01)
        nat = texture_sparse.T.dot(img_hist_match1d)
        a = np.dot(texture_sparse.T, texture_sparse)
        b = dx.T.dot(dx)
        c = dy.T.dot(dy)
        mat = a + 0.2 * (b + c)
        beta1d = spsolve(mat, nat)
        beta = np.reshape(beta1d, (img.shape[0], img.shape[1]), order="c")
        tone = texture ** beta
        tone = (tone - tone.min()) / (tone.max() - tone.min())
        return tone
    '''直方图匹配'''
    def __histogramMatching(self, img, mode=None):
        weights = self.weights_color if mode == 'color' else self.weights_gray
        # 图像
        histogram_img = cv2.calcHist([img], [0], None, [256], [0, 256])
        histogram_img.resize(histogram_img.size)
        histogram_img /= histogram_img.sum()
        histogram_img_cdf = np.cumsum(histogram_img)
        # 自然图像
        histogram_natural = np.zeros_like(histogram_img)
        for x in range(256):
            histogram_natural[x] = weights[0] * self.image_processor.Laplace(x) + weights[1] * self.image_processor.Uniform(x) + weights[2] * self.image_processor.Gaussian(x)
        histogram_natural /= histogram_natural.sum()
        histogram_natural_cdf = np.cumsum(histogram_natural)
        # 做直方图匹配
        img_hist_match = np.zeros_like(img)
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                value = histogram_img_cdf[img[x, y]]
                img_hist_match[x, y] = (np.abs(histogram_natural_cdf-value)).argmin()
        img_hist_match = np.true_divide(img_hist_match, 255)
        return img_hist_match