'''
Function:
    复现论文"Perceptual Losses for Real-Time Style Transfer and Super-Resolution"
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
from ..base import BaseBeautifier


'''ConvBlock'''
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, upsample=False, normalize=True, relu=True):
        super(ConvBlock, self).__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2), nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if normalize else None
        self.relu = relu
    '''forward'''
    def forward(self, x):
        if self.upsample: x = F.interpolate(x, scale_factor=2)
        x = self.block(x)
        if self.norm is not None: x = self.norm(x)
        if self.relu: x = F.relu(x)
        return x


'''ResidualBlock'''
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=True),
            ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=False),
        )
    '''forward'''
    def forward(self, x):
        return self.block(x) + x


'''TransformerNet, 模型修改自: https://github.com/eriklindernoren/Fast-Neural-Style-Transfer'''
class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 32, kernel_size=9, stride=1),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            ConvBlock(64, 128, kernel_size=3, stride=2),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ConvBlock(128, 64, kernel_size=3, upsample=True),
            ConvBlock(64, 32, kernel_size=3, upsample=True),
            ConvBlock(32, 3, kernel_size=9, stride=1, normalize=False, relu=False),
        )
    '''forward'''
    def forward(self, x):
        return self.model(x)


'''复现论文"Perceptual Losses for Real-Time Style Transfer and Super-Resolution"'''
class FastNeuralStyleTransferBeautifier(BaseBeautifier):
    def __init__(self, style='starrynight', **kwargs):
        super(FastNeuralStyleTransferBeautifier, self).__init__(**kwargs)
        self.model_urls = {
            'cuphead': 'https://github.com/CharlesPikachu/pydrawing/releases/download/checkpoints/fastneuralstyletransfer_cuphead.pth',
            'mosaic': 'https://github.com/CharlesPikachu/pydrawing/releases/download/checkpoints/fastneuralstyletransfer_mosaic.pth',
            'starrynight': 'https://github.com/CharlesPikachu/pydrawing/releases/download/checkpoints/fastneuralstyletransfer_starrynight.pth',
        }
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
        assert style in self.model_urls
        self.style = style
        self.transformer = TransformerNet()
        self.transformer.load_state_dict(model_zoo.load_url(self.model_urls[style]))
        self.transformer.eval()
        if torch.cuda.is_available(): self.transformer = self.transformer.cuda()
    '''迭代图片'''
    def iterimage(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = self.preprocess(image).unsqueeze(0)
        if torch.cuda.is_available(): input_image = input_image.cuda()
        with torch.no_grad(): output_image = self.transformer(input_image)[0]
        output_image = output_image.data.cpu().float()
        for c in range(3):
            output_image[c, :].mul_(self.std[c]).add_(self.mean[c])
        output_image = output_image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        output_image = cv2.resize(output_image, (image.shape[1], image.shape[0]))
        return output_image