'''
Function:
    利用贝塞尔曲线画画
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import os
import re
import cv2
import turtle
import numpy as np
from bs4 import BeautifulSoup
from ..base import BaseBeautifier


'''利用贝塞尔曲线画画'''
class BezierCurveBeautifier(BaseBeautifier):
    def __init__(self, num_samples=15, width=600, height=600, num_colors=32, **kwargs):
        super(BezierCurveBeautifier, self).__init__(**kwargs)
        self.num_samples = num_samples
        self.width = width
        self.height = height
        self.num_colors = num_colors
        self.rootdir = os.path.split(os.path.abspath(__file__))[0]
    '''迭代图片'''
    def iterimage(self, image):
        data = image.reshape((-1, 3))
        data = np.float32(data)
        # 聚类迭代停止的模式(停止的条件, 迭代最大次数, 精度)
        criteria = (cv2.TERM_CRITERIA_EPS, 10, 1.0)
        # 数据, 分类数, 预设的分类标签, 迭代停止的模式, 重复试验kmeans算法次数, 初始类中心的选择方式
        compactness, labels, centers = cv2.kmeans(data, self.num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        data_compress = centers[labels.flatten()]
        img_new = data_compress.reshape(image.shape)
        count = 0
        for center in centers:
            count += 1
            part = cv2.inRange(img_new, center, center)
            part = cv2.bitwise_not(part)
            cv2.imwrite('.tmp.bmp', part)
            os.system(f'{os.path.join(self.rootdir, "potrace.exe")} .tmp.bmp -s --flat')
            if count == 1:
                self.drawsvg('.tmp.svg', '#%02x%02x%02x' % (center[2], center[1], center[0]), True)
            else:
                self.drawsvg('.tmp.svg', '#%02x%02x%02x' % (center[2], center[1], center[0]), False)
        os.remove('.tmp.bmp')
        os.remove('.tmp.svg')
        turtle.done()
        return image
    '''一阶贝塞尔'''
    def FirstOrderBezier(self, p0, p1, t):
        assert (len(p0) == 2 and len(p1) == 2) or (len(p0) == 1 and len(p1) == 1)
        if len(p0) == 2 and len(p1) == 2:
            return p0[0] * (1 - t) + p1[0] * t, p0[1] * (1 - t) + p1[1] * t
        else:
            return p0 * (1 - t) + p1 * t
    '''二阶贝塞尔'''
    def SecondOrderBezier(self, p0, p1, p2):
        turtle.goto(p0)
        turtle.pendown()
        for t in range(0, self.num_samples+1):
            p = self.FirstOrderBezier(self.FirstOrderBezier(p0, p1, t/self.num_samples), self.FirstOrderBezier(p1, p2, t/self.num_samples), t/self.num_samples)
            turtle.goto(p)
        turtle.penup()
    '''三阶贝塞尔'''
    def ThirdOrderBezier(self, p0, p1, p2, p3):
        p0 = -self.width / 2 + p0[0], self.height / 2 - p0[1]
        p1 = -self.width / 2 + p1[0], self.height / 2 - p1[1]
        p2 = -self.width / 2 + p2[0], self.height / 2 - p2[1]
        p3 = -self.width / 2 + p3[0], self.height / 2 - p3[1]
        turtle.goto(p0)
        turtle.pendown()
        for t in range(0, self.num_samples+1):
            p = self.FirstOrderBezier(
                self.FirstOrderBezier(self.FirstOrderBezier(p0, p1, t/self.num_samples), self.FirstOrderBezier(p1, p2, t/self.num_samples), t/self.num_samples), 
                self.FirstOrderBezier(self.FirstOrderBezier(p1, p2, t/self.num_samples), self.FirstOrderBezier(p2, p3, t/self.num_samples), t/self.num_samples), 
                t/self.num_samples
            )
            turtle.goto(p)
        turtle.penup()
    '''画图(SVG)'''
    def drawsvg(self, filename, color, is_first=True, speed=1000):
        svgfile = open(filename, 'r')
        soup = BeautifulSoup(svgfile.read(), 'lxml')
        height, width = float(soup.svg.attrs['height'][:-2]), float(soup.svg.attrs['width'][:-2])
        scale = tuple(map(float, re.findall(r'scale\((.*?)\)', soup.g.attrs['transform'])[0].split(',')))
        scale = scale[0], -scale[1]
        if is_first:
            turtle.setup(height=height, width=width)
            turtle.setworldcoordinates(-width/2, 300, width-width/2, -height+300)
        turtle.tracer(100)
        turtle.pensize(1)
        turtle.speed(speed)
        turtle.penup()
        turtle.color(color)
        for path in soup.find_all('path'):
            attrs = path.attrs['d'].replace('\n', ' ')
            attrs = attrs.split(' ')
            attrs_yield = self.yieldattrs(attrs)
            endl = ''
            for attr in attrs_yield:
                if attr == 'M':
                    turtle.end_fill()
                    x, y = attrs_yield.__next__() * scale[0], attrs_yield.__next__() * scale[1]
                    turtle.penup()
                    turtle.goto(-self.width/2+x, self.height/2-y)
                    turtle.pendown()
                    turtle.begin_fill()
                elif attr == 'm':
                    turtle.end_fill()
                    dx, dy = attrs_yield.__next__() * scale[0], attrs_yield.__next__() * scale[1]
                    turtle.penup()
                    turtle.goto(turtle.xcor()+dx, turtle.ycor()-dy)
                    turtle.pendown()
                    turtle.begin_fill()
                elif attr == 'C':
                    p1 = attrs_yield.__next__() * scale[0], attrs_yield.__next__() * scale[1]
                    p2 = attrs_yield.__next__() * scale[0], attrs_yield.__next__() * scale[1]
                    p3 = attrs_yield.__next__() * scale[0], attrs_yield.__next__() * scale[1]
                    turtle.penup()
                    p0 = turtle.xcor() + self.width / 2, self.height / 2 - turtle.ycor()
                    self.ThirdOrderBezier(p0, p1, p2, p3)
                    endl = attr
                elif attr == 'c':
                    turtle.penup()
                    p0 = turtle.xcor() + self.width / 2, self.height / 2 - turtle.ycor()
                    p1 = attrs_yield.__next__() * scale[0] + p0[0], attrs_yield.__next__() * scale[1] + p0[1]
                    p2 = attrs_yield.__next__() * scale[0] + p0[0], attrs_yield.__next__() * scale[1] + p0[1]
                    p3 = attrs_yield.__next__() * scale[0] + p0[0], attrs_yield.__next__() * scale[1] + p0[1]
                    self.ThirdOrderBezier(p0, p1, p2, p3)
                    endl = attr
                elif attr == 'L':
                    x, y = attrs_yield.__next__() * scale[0], attrs_yield.__next__() * scale[1]
                    turtle.pendown()
                    turtle.goto(-self.width/2+x, self.height/2-y)
                    turtle.penup()
                elif attr == 'l':
                    dx, dy = attrs_yield.__next__() * scale[0], attrs_yield.__next__() * scale[1]
                    turtle.pendown()
                    turtle.goto(turtle.xcor()+dx, turtle.ycor()-dy)
                    turtle.penup()
                    endl = attr
                elif endl == 'C':
                    p1 = attr * scale[0], attrs_yield.__next__() * scale[1]
                    p2 = attrs_yield.__next__() * scale[0], attrs_yield.__next__() * scale[1]
                    p3 = attrs_yield.__next__() * scale[0], attrs_yield.__next__() * scale[1]
                    turtle.penup()
                    p0 = turtle.xcor() + self.width / 2, self.height / 2 - turtle.ycor()
                    self.ThirdOrderBezier(p0, p1, p2, p3)
                elif endl == 'c':
                    turtle.penup()
                    p0 = turtle.xcor() + self.width / 2, self.height / 2 - turtle.ycor()
                    p1 = attr * scale[0] + p0[0], attrs_yield.__next__() * scale[1] + p0[1]
                    p2 = attrs_yield.__next__() * scale[0] + p0[0], attrs_yield.__next__() * scale[1] + p0[1]
                    p3 = attrs_yield.__next__() * scale[0] + p0[0], attrs_yield.__next__() * scale[1] + p0[1]
                    self.ThirdOrderBezier(p0, p1, p2, p3)
                elif endl == 'L':
                    x, y = attr * scale[0], attrs_yield.__next__() * scale[1]
                    turtle.pendown()
                    turtle.goto(-self.width/2+x, self.height/2-y)
                    turtle.penup()
                elif endl == 'l':
                    dx, dy = attr * scale[0], attrs_yield.__next__() * scale[1]
                    turtle.pendown()
                    turtle.goto(turtle.xcor()+dx, turtle.ycor()-dy)
                    turtle.penup()
        turtle.penup()
        turtle.hideturtle()
        turtle.update()
        svgfile.close()
    '''attrs生成器'''
    @staticmethod
    def yieldattrs(attrs):
        for attr in attrs:
            if attr.isdigit():
                yield float(attr)
            elif attr[0].isalpha():
                yield attr[0]
                yield float(attr[1:])
            elif attr[-1].isalpha():
                yield float(attr[0: -1])
            elif attr[0] == '-':
                yield float(attr)