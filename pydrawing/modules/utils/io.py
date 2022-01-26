'''
Function:
    IO相关的工具函数
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import os
import cv2
from tqdm import tqdm


'''检查文件是否存在'''
def checkdir(dirname):
    if os.path.exists(dirname): return True
    os.mkdir(dirname)
    return False


'''将图片转为视频并保存'''
def Images2VideoAndSave(images, savedir='outputs', savename='output', fps=25, ext='avi', logger_handle=None):
    checkdir(savedir)
    savepath = os.path.join(savedir, savename + f'.{ext}')
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_writer = cv2.VideoWriter(savepath, fourcc, fps, (images[0].shape[1], images[0].shape[0]))
    pbar = tqdm(images)
    for image in pbar:
        pbar.set_description(f'Writing image to {savepath}')
        video_writer.write(image)
    if logger_handle is not None: logger_handle.info(f'Video is saved into {savepath}')


'''保存图片'''
def SaveImage(image, savedir='outputs', savename='output', ext='png', logger_handle=None):
    checkdir(savedir)
    savepath = os.path.join(savedir, savename + f'.{ext}')
    cv2.imwrite(savepath, image)
    if logger_handle is not None: logger_handle.info(f'Image is saved into {savepath}')


'''读取视频'''
def ReadVideo(videopath):
    capture, images = cv2.VideoCapture(videopath), []
    fps = capture.get(cv2.CAP_PROP_FPS)
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret: break
        images.append(frame)
    capture.release()
    return images, fps