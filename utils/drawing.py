'''
Function:
	Algorithm implementation.
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import cv2
import math
import numpy as np
from PIL import Image
from scipy import signal
from utils.utils import *
from scipy.ndimage import interpolation
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, spdiags
import warnings
warnings.filterwarnings("ignore")


'''pencil drawing'''
class PencilDrawing():
	def __init__(self, **kwargs):
		self.kernel_size_scale = kwargs.get('kernel_size_scale')
		self.stroke_width = kwargs.get('stroke_width')
		self.weights_color = kwargs.get('weights_color')
		self.weights_gray = kwargs.get('weights_gray')
		self.texture_path = kwargs.get('texture_path')
		self.color_depth = kwargs.get('color_depth')
	'''in order to call'''
	def draw(self, image_path, mode='gray', savename='output.jpg'):
		img = cv2.imread(image_path)
		if mode == 'color':
			'''
			img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
			Y = img_ycbcr[:, :, 0]
			img_ycbcr_new = img_ycbcr.copy()
			img_ycbcr_new.flags.writeable = True
			img_ycbcr_new[:, :, 0] = self.__strokeGeneration(Y) * self.__toneGeneration(Y) * 255
			img_out = cv2.cvtColor(img_ycbcr_new, cv2.COLOR_YCR_CB2BGR)
			img = cv2.imwrite(savename, img_out)
			'''
			img = Image.open(image_path)
			img_ycbcr = img.convert('YCbCr')
			img = np.ndarray((img.size[1], img.size[0], 3), 'u1', img_ycbcr.tobytes())
			img_out = img.copy()
			img_out.flags.writeable = True
			img_out[:, :, 0] = self.__strokeGeneration(img[:, :, 0]) * self.__toneGeneration(img[:, :, 0]) * 255
			img_out = cv2.cvtColor(img_out, cv2.COLOR_YCR_CB2BGR)
			img_out = Image.fromarray(img_out)
			img_out.save(savename)
		elif mode == 'gray':
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img_s = self.__strokeGeneration(img)
			img_t = self.__toneGeneration(img)
			img_out = img_s * img_t * 255
			img = cv2.imwrite(savename, img_out)
		else:
			raise ValueError('PencilDrawing.draw unsupport mode <%s>...' % mode)
	'''pencil stroke generation'''
	def __strokeGeneration(self, img):
		h, w = img.shape
		kernel_size = int(min(w, h) * self.kernel_size_scale)
		kernel_size += kernel_size % 2
		# compute gradients, yielding magnitude
		img_double = im2double(img)
		dx = np.concatenate((np.abs(img_double[:, 0:-1]-img_double[:, 1:]), np.zeros((h, 1))), 1)
		dy = np.concatenate((np.abs(img_double[0:-1, :]-img_double[1:, :]), np.zeros((1, w))), 0)
		img_gradient = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
		# choose eight reference directions
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
		# get response maps for the reference directions
		response_maps = np.zeros((h, w, 8))
		for i in range(8):
			response_maps[:, :, i] = signal.convolve2d(img_gradient, line_segments[:, :, i], 'same')
		response_maps_maxvalueidx = response_maps.argmax(axis=-1)
		# the classification is performed by selecting the maximum value among the responses in all directions
		magnitude_maps = np.zeros_like(response_maps)
		for i in range(8):
			magnitude_maps[:, :, i] = img_gradient * (response_maps_maxvalueidx == i).astype('float')
		# line shaping
		stroke_maps = np.zeros_like(response_maps)
		for i in range(8):
			stroke_maps[:, :, i] = signal.convolve2d(magnitude_maps[:, :, i], line_segments[:, :, i], 'same')
		stroke_maps = stroke_maps.sum(axis=-1)
		stroke_maps = (stroke_maps - stroke_maps.min()) / (stroke_maps.max() - stroke_maps.min())
		stroke_maps = (1 - stroke_maps) * self.stroke_width
		return stroke_maps
	'''pencil tone drawing'''
	def __toneGeneration(self, img, mode=None):
		height, width = img.shape
		# histogram matching
		img_hist_match = self.__histogramMatching(img, mode) ** self.color_depth
		# get texture
		texture = cv2.imread(self.texture_path)
		texture = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)[99: texture.shape[0]-100, 99: texture.shape[1]-100]
		ratio = 0.2 * min(img.shape[0], img.shape[1]) / float(1024)
		texture = interpolation.zoom(texture, (ratio, ratio))
		texture = im2double(texture)
		texture = horizontalStitch(texture, img.shape[1])
		texture = verticalStitch(texture, img.shape[0])
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
	'''histogram matching'''
	def __histogramMatching(self, img, mode=None):
		weights = self.weights_color if mode == 'color' else self.weights_gray
		# img
		histogram_img = cv2.calcHist([img], [0], None, [256], [0, 256])
		histogram_img.resize(histogram_img.size)
		histogram_img /= histogram_img.sum()
		histogram_img_cdf = np.cumsum(histogram_img)
		# natural
		histogram_natural = np.zeros_like(histogram_img)
		for x in range(256):
			histogram_natural[x] = weights[0] * Laplace(x) + weights[1] * Uniform(x) + weights[2] * Gaussian(x)
		histogram_natural /= histogram_natural.sum()
		histogram_natural_cdf = np.cumsum(histogram_natural)
		# do the histogram matching
		img_hist_match = np.zeros_like(img)
		for x in range(img.shape[0]):
			for y in range(img.shape[1]):
				value = histogram_img_cdf[img[x, y]]
				img_hist_match[x, y] = (np.abs(histogram_natural_cdf-value)).argmin()
		img_hist_match = np.true_divide(img_hist_match, 255)
		return img_hist_match