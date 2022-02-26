'''
Function:
    照片怀旧风格
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import numpy as np
from ...utils import checkdir
from ..base import BaseBeautifier


'''照片怀旧风格'''
class NostalgicstyleBeautifier(BaseBeautifier):
    def __init__(self, **kwargs):
        super(NostalgicstyleBeautifier, self).__init__(**kwargs)
    '''迭代图片'''
    def iterimage(self, image):
        image = image.astype(np.float32)
        image_processed = image.copy()
        image_processed[..., 0] = image[..., 2] * 0.272 + image[..., 1] * 0.534 + image[..., 0] * 0.131
        image_processed[..., 1] = image[..., 2] * 0.349 + image[..., 1] * 0.686 + image[..., 0] * 0.168
        image_processed[..., 2] = image[..., 2] * 0.393 + image[..., 1] * 0.769 + image[..., 0] * 0.189
        image_processed[image_processed > 255.0] = 255.0
        return image_processed