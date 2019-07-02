'''
Function:
	Some utils
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import cv2
import math
import numpy as np


'''
Function:
	normalize the image value between [0.0, 1.0]
Input:
	--img: np.array
Output:
	--img: np.array
'''
def im2double(img):
	if len(img.shape) == 2:
		return (img - img.min()) / (img.max() - img.min())
	else:
		return cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)


'''
Function:
	Laplacian distribution with its peak at the brightest value
Input:
	--x: pixel value between [0, 255]
	--sigma: the scale of the distribution
Output:
	--value: value represent bright layer
'''
def Laplace(x, sigma=9):
	value = (1./sigma) * math.exp(-(256-x)/sigma) * (256 - x)
	return value


'''
Function:
	use the uniform distribution and encourage full utilization of different gray levels to enrich pencil drawing
Input:
	--x: pixel value between [0, 255]
	--ua, ub: two controlling parameters defining the range of the distribution
Output:
	--value: value represent mild tone layer
'''
def Uniform(x, ua=105, ub=225):
	value = (1. / (ub - ua)) * (max(x-ua, 0) - max(x-ub, 0))
	return value


'''
Function:
	gaussian distribution for dark layer
Input:
	--x: pixel value between [0, 255]
	--u: the mean value of the dark strokes
	--sigma: the scale parameter
Output:
	--value: value represent dark layer
'''
def Gaussian(x, u=90, sigma=11):
	value = (1./math.sqrt(2*math.pi*sigma)) * math.exp(-((x-u)**2)/(2*(sigma**2)))
	return value


'''horizontal stitch'''
def horizontalStitch(img, width):
	img_stitch = img.copy()
	while img_stitch.shape[1] < width:
		window_size = int(round(img.shape[1] / 4.))
		left = img[:, (img.shape[1]-window_size): img.shape[1]]
		right = img[:, 0:window_size]
		aleft = np.zeros((left.shape[0], window_size))
		aright = np.zeros((left.shape[0], window_size))
		for i in range(window_size):
			aleft[:, i] = left[:, i] * (1 - (i + 1.) / window_size)
			aright[:, i] = right[:, i] * (i + 1.) / window_size
		img_stitch = np.column_stack((img_stitch[:, 0: (img_stitch.shape[1]-window_size)], aleft+aright, img_stitch[:, window_size: img_stitch.shape[1]]))
	img_stitch = img_stitch[:, 0: width]
	return img_stitch


'''vertical stitch'''
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
		img_stitch = np.row_stack((img_stitch[0: img_stitch.shape[0]-window_size, :], aup+adown, img_stitch[window_size: img_stitch.shape[0], :]))
	img_stitch = img_stitch[0: height, :]
	return img_stitch