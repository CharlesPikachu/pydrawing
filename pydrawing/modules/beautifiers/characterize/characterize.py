'''
Function:
    视频转字符画
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import cv2
import numpy as np
from ..base import BaseBeautifier
from PIL import Image, ImageFont, ImageDraw


'''视频转字符画'''
class CharacterizeBeautifier(BaseBeautifier):
    CHARS = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
    def __init__(self, **kwargs):
        super(CharacterizeBeautifier, self).__init__(**kwargs)
    '''迭代图片'''
    def iterimage(self, image):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # 每个字符的大小
        font = ImageFont.load_default().font
        font_w, font_h = font.getsize(self.CHARS[1])
        # 输入图像变成可整除字符大小
        image = image.resize((font_w * (image.width // font_w), font_h * (image.height // font_h)), Image.NEAREST)
        # 原始大小
        h_ori, w_ori = image.height, image.width
        # resize
        image = image.resize((w_ori // font_w, h_ori // font_h), Image.NEAREST)
        h, w = image.height, image.width
        # 图像RGB转字符
        txts, colors = '', []
        for i in range(h):
            for j in range(w):
                pixel = image.getpixel((j, i))
                colors.append(pixel[:3])
                txts += self.rgb2char(*pixel)
        image = Image.new('RGB', (w_ori, h_ori), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        x = y = 0
        for j in range(len(txts)):
            if x == w_ori: x, y = 0, y + font_h
            draw.text((x, y), txts[j], font=font, fill=colors[j])
            x += font_w
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    '''RGB转字符'''
    def rgb2char(self, r, g, b, alpha=256):
        if alpha == 0: return ''
        gray = int(0.2126 * r + 0.7152 * g + 0.0722 * b)
        return self.CHARS[gray % len(self.CHARS)]