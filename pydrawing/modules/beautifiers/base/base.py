'''
Function:
    Beautifier基类
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import cv2
from ...utils import Images2VideoAndSave, SaveImage


'''Beautifier基类'''
class BaseBeautifier():
    def __init__(self, savedir='outputs', savename='output', **kwargs):
        self.savename, self.savedir = savename, savedir
        for key, value in kwargs.items(): setattr(self, key, value)
    '''处理文件'''
    def process(self, filepath, images=None):
        assert images is None or filepath is None, 'please input filepath or images rather than both'
        if images is None:
            # 图片
            if filepath.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']:
                images = [cv2.imread(filepath)]
            # 视频
            elif filepath.split('.')[-1].lower() in ['mp4', 'avi']:
                images = ReadVideo(filepath)
            # 不支持的数据格式
            else:
                raise RuntimeError('Unsupport file type %s...' % filepath.split('.')[-1])
        outputs = []
        for image in images:
            outputs.append(self.iterimage(image))
        if len(outputs) > 1:
            fps, ext = 25, 'avi'
            if hasattr(self, 'fps'): fps = self.fps
            if hasattr(self, 'ext'): ext = self.ext
            Images2VideoAndSave(outputs, savedir=self.savedir, savename=self.savename, fps=fps, ext=ext, logger_handle=self.logger_handle)
        else:
            ext = 'png'
            if hasattr(self, 'ext'): ext = self.ext
            SaveImage(outputs[0], savedir=self.savedir, savename=self.savename, ext=ext, logger_handle=self.logger_handle)
    '''迭代图片'''
    def iterimage(self, image):
        raise NotImplementedError('not to be implemented')