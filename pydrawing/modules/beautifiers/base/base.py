'''
Function:
    Beautifier基类
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import os
import cv2
import subprocess
from tqdm import tqdm
from ...utils import Images2VideoAndSave, SaveImage, ReadVideo


'''Beautifier基类'''
class BaseBeautifier():
    def __init__(self, savedir='outputs', savename='output', **kwargs):
        self.savename, self.savedir = savename, savedir
        self.merge_audio, self.tmp_audio_path = False, 'cache.mp3'
        for key, value in kwargs.items(): setattr(self, key, value)
    '''处理文件'''
    def process(self, filepath, images=None):
        assert images is None or filepath is None, 'please input filepath or images rather than both'
        # 图片/视频处理
        if images is None:
            # --图片
            if filepath.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']:
                images = [cv2.imread(filepath)]
            # --视频
            elif filepath.split('.')[-1].lower() in ['mp4', 'avi']:
                images, self.fps = ReadVideo(filepath)
                if self.merge_audio:
                    p = subprocess.Popen(f'ffmpeg -i {filepath} -f mp3 {self.tmp_audio_path}')
                    while True:
                        if subprocess.Popen.poll(p) is not None: break
            # --不支持的数据格式
            else:
                raise RuntimeError('Unsupport file type %s...' % filepath.split('.')[-1])
        outputs, pbar = [], tqdm(images)
        for image in pbar:
            pbar.set_description('Process image')
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
        # 如果有音频, 则把音频merge进新视频
        if self.merge_audio:
            p = subprocess.Popen(f'ffmpeg -i {os.path.join(self.savedir, self.savename+f".{ext}")} -i {self.tmp_audio_path} -strict -2 -f mp4 {os.path.join(self.savedir, self.savename+"_audio.mp4")}')
            while True:
                if subprocess.Popen.poll(p) is not None: break
            os.remove(self.tmp_audio_path)
            self.logger_handle.info(f'Video with merged audio is saved into {os.path.join(self.savedir, self.savename+"_audio.mp4")}')
    '''迭代图片'''
    def iterimage(self, image):
        raise NotImplementedError('not to be implemented')