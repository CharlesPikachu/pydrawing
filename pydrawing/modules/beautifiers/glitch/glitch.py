'''
Function:
    信号故障的效果
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import os
import random
from ...utils import checkdir
from ..base import BaseBeautifier


'''信号故障的效果'''
class GlitchBeautifier(BaseBeautifier):
    def __init__(self, header_size=200, intensity=0.1, block_size=100, **kwargs):
        super(GlitchBeautifier, self).__init__(**kwargs)
        self.header_size, self.intensity, self.block_size = header_size, intensity, block_size
    '''处理文件'''
    def process(self, filepath):
        checkdir(self.savedir)
        ext = filepath.split('.')[-1]
        assert ext.lower() in ['mp4', 'avi']
        with open(filepath, 'rb') as fp_in:
            with open(os.path.join(self.savedir, f'{self.savename}.{ext}'), 'wb') as fp_out:
                fp_out.write(fp_in.read(self.header_size))
                while True:
                    block_data = fp_in.read(self.block_size)
                    if not block_data: break
                    if random.random() < self.intensity / 100: block_data = os.urandom(self.block_size)
                    fp_out.write(block_data)
        self.logger_handle.info(f'Video is saved into {self.savename}.{ext}')