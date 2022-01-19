'''
Function:
    人脸卡通化
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
import torch.utils.model_zoo as model_zoo
from PIL import Image
from ..base import BaseBeautifier
from .facedetector import FaceDetector
from .facesegmentor import FaceSegmentor
from torch.nn.parameter import Parameter


'''ConvBlock'''
class ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConvBlock, self).__init__()
        self.dim_out = dim_out
        self.ConvBlock1 = nn.Sequential(
            nn.InstanceNorm2d(dim_in),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim_in, dim_out//2, kernel_size=3, stride=1, bias=False)
        )
        self.ConvBlock2 = nn.Sequential(
            nn.InstanceNorm2d(dim_out//2),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim_out//2, dim_out//4, kernel_size=3, stride=1, bias=False)
        )
        self.ConvBlock3 = nn.Sequential(
            nn.InstanceNorm2d(dim_out//4),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim_out//4, dim_out//4, kernel_size=3, stride=1, bias=False)
        )
        self.ConvBlock4 = nn.Sequential(
            nn.InstanceNorm2d(dim_in),
            nn.ReLU(True),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, bias=False)
        )
    '''forward'''
    def forward(self, x):
        residual = x
        x1 = self.ConvBlock1(x)
        x2 = self.ConvBlock2(x1)
        x3 = self.ConvBlock3(x2)
        out = torch.cat((x1, x2, x3), 1)
        if residual.size(1) != self.dim_out: residual = self.ConvBlock4(residual)
        return residual + out


'''HourGlassBlock'''
class HourGlassBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(HourGlassBlock, self).__init__()
        self.ConvBlock1_1 = ConvBlock(dim_in, dim_out)
        self.ConvBlock1_2 = ConvBlock(dim_out, dim_out)
        self.ConvBlock2_1 = ConvBlock(dim_out, dim_out)
        self.ConvBlock2_2 = ConvBlock(dim_out, dim_out)
        self.ConvBlock3_1 = ConvBlock(dim_out, dim_out)
        self.ConvBlock3_2 = ConvBlock(dim_out, dim_out)
        self.ConvBlock4_1 = ConvBlock(dim_out, dim_out)
        self.ConvBlock4_2 = ConvBlock(dim_out, dim_out)
        self.ConvBlock5 = ConvBlock(dim_out, dim_out)
        self.ConvBlock6 = ConvBlock(dim_out, dim_out)
        self.ConvBlock7 = ConvBlock(dim_out, dim_out)
        self.ConvBlock8 = ConvBlock(dim_out, dim_out)
        self.ConvBlock9 = ConvBlock(dim_out, dim_out)
    '''forward'''
    def forward(self, x):
        skip1 = self.ConvBlock1_1(x)
        down1 = F.avg_pool2d(x, 2)
        down1 = self.ConvBlock1_2(down1)
        skip2 = self.ConvBlock2_1(down1)
        down2 = F.avg_pool2d(down1, 2)
        down2 = self.ConvBlock2_2(down2)
        skip3 = self.ConvBlock3_1(down2)
        down3 = F.avg_pool2d(down2, 2)
        down3 = self.ConvBlock3_2(down3)
        skip4 = self.ConvBlock4_1(down3)
        down4 = F.avg_pool2d(down3, 2)
        down4 = self.ConvBlock4_2(down4)
        center = self.ConvBlock5(down4)
        up4 = self.ConvBlock6(center)
        up4 = F.upsample(up4, scale_factor=2)
        up4 = skip4 + up4
        up3 = self.ConvBlock7(up4)
        up3 = F.upsample(up3, scale_factor=2)
        up3 = skip3 + up3
        up2 = self.ConvBlock8(up3)
        up2 = F.upsample(up2, scale_factor=2)
        up2 = skip2 + up2
        up1 = self.ConvBlock9(up2)
        up1 = F.upsample(up1, scale_factor=2)
        up1 = skip1 + up1
        return up1


'''HourGlass'''
class HourGlass(nn.Module):
    def __init__(self, dim_in, dim_out, use_res=True):
        super(HourGlass, self).__init__()
        self.use_res = use_res
        self.HG = nn.Sequential(
            HourGlassBlock(dim_in, dim_out),
            ConvBlock(dim_out, dim_out),
            nn.Conv2d(dim_out, dim_out, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(dim_out),
            nn.ReLU(True)
        )
        self.Conv1 = nn.Conv2d(dim_out, 3, kernel_size=1, stride=1)
        if self.use_res:
            self.Conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=1, stride=1)
            self.Conv3 = nn.Conv2d(3, dim_out, kernel_size=1, stride=1)
    '''forward'''
    def forward(self, x):
        ll = self.HG(x)
        tmp_out = self.Conv1(ll)
        if self.use_res:
            ll = self.Conv2(ll)
            tmp_out_ = self.Conv3(tmp_out)
            return x + ll + tmp_out_
        else:
            return tmp_out


'''ResnetBlock'''
class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias=False):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias), nn.InstanceNorm2d(dim), nn.ReLU(True)]
        conv_block += [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias), nn.InstanceNorm2d(dim)]
        self.conv_block = nn.Sequential(*conv_block)
    '''forward'''
    def forward(self, x):
        out = x + self.conv_block(x)
        return out


'''adaLIN'''
class adaLIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(adaLIN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)
    '''forward'''
    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        return out


'''SoftAdaLIN'''
class SoftAdaLIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(SoftAdaLIN, self).__init__()
        self.norm = adaLIN(num_features, eps)
        self.w_gamma = Parameter(torch.zeros(1, num_features))
        self.w_beta = Parameter(torch.zeros(1, num_features))
        self.c_gamma = nn.Sequential(nn.Linear(num_features, num_features), nn.ReLU(True), nn.Linear(num_features, num_features))
        self.c_beta = nn.Sequential(nn.Linear(num_features, num_features), nn.ReLU(True), nn.Linear(num_features, num_features))
        self.s_gamma = nn.Linear(num_features, num_features)
        self.s_beta = nn.Linear(num_features, num_features)
    '''forward'''
    def forward(self, x, content_features, style_features):
        content_gamma, content_beta = self.c_gamma(content_features), self.c_beta(content_features)
        style_gamma, style_beta = self.s_gamma(style_features), self.s_beta(style_features)
        w_gamma, w_beta = self.w_gamma.expand(x.shape[0], -1), self.w_beta.expand(x.shape[0], -1)
        soft_gamma = (1. - w_gamma) * style_gamma + w_gamma * content_gamma
        soft_beta = (1. - w_beta) * style_beta + w_beta * content_beta
        out = self.norm(x, soft_gamma, soft_beta)
        return out


'''ResnetSoftAdaLINBlock'''
class ResnetSoftAdaLINBlock(nn.Module):
    def __init__(self, dim, use_bias=False):
        super(ResnetSoftAdaLINBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = SoftAdaLIN(dim)
        self.relu1 = nn.ReLU(True)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = SoftAdaLIN(dim)
    '''forward'''
    def forward(self, x, content_features, style_features):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, content_features, style_features)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, content_features, style_features)
        return out + x


'''LIN'''
class LIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(LIN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)
    '''forward'''
    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)
        return out


'''ResnetGenerator, 模型修改自: https://github.com/minivision-ai/photo2cartoon'''
class ResnetGenerator(nn.Module):
    def __init__(self, ngf=64, img_size=256, light=False):
        super(ResnetGenerator, self).__init__()
        self.light = light
        self.ConvBlock1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, kernel_size=7, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        )
        self.HourGlass1 = HourGlass(ngf, ngf)
        self.HourGlass2 = HourGlass(ngf, ngf)
        # Down-Sampling
        self.DownBlock1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        self.DownBlock2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(ngf*4),
            nn.ReLU(True)
        )
        # Encoder Bottleneck
        self.EncodeBlock1 = ResnetBlock(ngf*4)
        self.EncodeBlock2 = ResnetBlock(ngf*4)
        self.EncodeBlock3 = ResnetBlock(ngf*4)
        self.EncodeBlock4 = ResnetBlock(ngf*4)
        # Class Activation Map
        self.gap_fc = nn.Linear(ngf*4, 1)
        self.gmp_fc = nn.Linear(ngf*4, 1)
        self.conv1x1 = nn.Conv2d(ngf*8, ngf*4, kernel_size=1, stride=1)
        self.relu = nn.ReLU(True)
        # Gamma, Beta block
        if self.light:
            self.FC = nn.Sequential(nn.Linear(ngf*4, ngf*4), nn.ReLU(True), nn.Linear(ngf*4, ngf*4), nn.ReLU(True))
        else:
            self.FC = nn.Sequential(nn.Linear(img_size//4*img_size//4*ngf*4, ngf*4), nn.ReLU(True), nn.Linear(ngf*4, ngf*4), nn.ReLU(True))
        # Decoder Bottleneck
        self.DecodeBlock1 = ResnetSoftAdaLINBlock(ngf*4)
        self.DecodeBlock2 = ResnetSoftAdaLINBlock(ngf*4)
        self.DecodeBlock3 = ResnetSoftAdaLINBlock(ngf*4)
        self.DecodeBlock4 = ResnetSoftAdaLINBlock(ngf*4)
        # Up-Sampling
        self.UpBlock1 = nn.Sequential(
            nn.Upsample(scale_factor=2), 
            nn.ReflectionPad2d(1), 
            nn.Conv2d(ngf*4, ngf*2, kernel_size=3, stride=1, padding=0, bias=False),
            LIN(ngf*2),
            nn.ReLU(True)
        )
        self.UpBlock2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf*2, ngf, kernel_size=3, stride=1, padding=0, bias=False),
            LIN(ngf),
            nn.ReLU(True)
        )
        self.HourGlass3 = HourGlass(ngf, ngf)
        self.HourGlass4 = HourGlass(ngf, ngf, False)
        self.ConvBlock2 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 3, kernel_size=7, stride=1, padding=0, bias=False),
            nn.Tanh()
        )
    '''forward'''
    def forward(self, x):
        x = self.ConvBlock1(x)
        x = self.HourGlass1(x)
        x = self.HourGlass2(x)
        x = self.DownBlock1(x)
        x = self.DownBlock2(x)
        x = self.EncodeBlock1(x)
        content_features1 = F.adaptive_avg_pool2d(x, 1).view(x.shape[0], -1)
        x = self.EncodeBlock2(x)
        content_features2 = F.adaptive_avg_pool2d(x, 1).view(x.shape[0], -1)
        x = self.EncodeBlock3(x)
        content_features3 = F.adaptive_avg_pool2d(x, 1).view(x.shape[0], -1)
        x = self.EncodeBlock4(x)
        content_features4 = F.adaptive_avg_pool2d(x, 1).view(x.shape[0], -1)
        gap = F.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)
        gmp = F.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)
        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))
        heatmap = torch.sum(x, dim=1, keepdim=True)
        if self.light:
            x_ = F.adaptive_avg_pool2d(x, 1)
            style_features = self.FC(x_.view(x_.shape[0], -1))
        else:
            style_features = self.FC(x.view(x.shape[0], -1))
        x = self.DecodeBlock1(x, content_features4, style_features)
        x = self.DecodeBlock2(x, content_features3, style_features)
        x = self.DecodeBlock3(x, content_features2, style_features)
        x = self.DecodeBlock4(x, content_features1, style_features)
        x = self.UpBlock1(x)
        x = self.UpBlock2(x)
        x = self.HourGlass3(x)
        x = self.HourGlass4(x)
        out = self.ConvBlock2(x)
        return out, cam_logit, heatmap


'''人脸卡通化'''
class CartoonizeFaceBeautifier(BaseBeautifier):
    def __init__(self, use_cuda=False, use_face_segmentor=True, **kwargs):
        super(CartoonizeFaceBeautifier, self).__init__(**kwargs)
        self.model_urls = {
            'transformer': 'https://github.com/CharlesPikachu/pydrawing/releases/download/checkpoints/cartoonizeface_transformer.pth',
        }
        self.use_cuda = use_cuda
        self.use_face_segmentor = use_face_segmentor
        self.face_detector = FaceDetector(use_cuda=(torch.cuda.is_available() and self.use_cuda))
        self.transformer = ResnetGenerator(ngf=32, img_size=256, light=True)
        self.transformer.load_state_dict(model_zoo.load_url(self.model_urls['transformer'])['genA2B'])
        self.transformer.eval()
        if use_face_segmentor: 
            self.face_segmentor = FaceSegmentor()
            self.face_segmentor.eval()
        if torch.cuda.is_available() and self.use_cuda: 
            self.transformer = self.transformer.cuda()
            if use_face_segmentor: self.face_segmentor = self.face_segmentor.cuda()
    '''迭代图片'''
    def iterimage(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 人脸提取
        face_rgb = self.face_detector(image)
        # 人脸分割
        if self.use_face_segmentor:
            face_rgb_for_seg = self.face_segmentor.preprocess(face_rgb)
            mask = self.face_segmentor(face_rgb_for_seg)
            mask = F.interpolate(mask, size=face_rgb.shape[:2][::-1], mode='bilinear', align_corners=False)
            mask = mask[0].argmax(0).cpu().numpy().astype(np.int32)
            mask = self.face_segmentor.getfacemask(mask)
        else:
            mask = np.ones(face_rgb.shape[:2]) * 255
        mask = mask[:, :, np.newaxis]
        face_rgba = np.dstack((face_rgb, mask))
        # 人脸处理
        face_rgba = cv2.resize(face_rgba, (256, 256), interpolation=cv2.INTER_AREA)
        face = face_rgba[:, :, :3].copy()
        mask = face_rgba[:, :, 3][:, :, np.newaxis].copy() / 255.
        face = (face * mask + (1 - mask) * 255) / 127.5 - 1
        face = np.transpose(face[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)
        face = torch.from_numpy(face).type(torch.FloatTensor)
        if torch.cuda.is_available() and self.use_cuda: 
            face = face.cuda()
        # 推理
        with torch.no_grad(): 
            face_cartoon = self.transformer(face)[0][0]
        # 后处理
        face_cartoon = np.transpose(face_cartoon.cpu().numpy(), (1, 2, 0))
        face_cartoon = (face_cartoon + 1) * 127.5
        face_cartoon = (face_cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
        face_cartoon = cv2.cvtColor(face_cartoon, cv2.COLOR_RGB2BGR)
        # 返回
        return face_cartoon