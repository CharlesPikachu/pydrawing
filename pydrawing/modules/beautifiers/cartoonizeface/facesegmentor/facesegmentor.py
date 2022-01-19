'''
Function:
    人脸分割
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


'''config'''
SEGMENTOR_CFG = {
    'type': 'ce2p',
    'benchmark': True,
    'num_classes': -1,
    'align_corners': False,
    'is_multi_gpus': True,
    'distributed': {'is_on': True, 'backend': 'nccl'},
    'norm_cfg': {'type': 'batchnorm2d', 'opts': {}},
    'act_cfg': {'type': 'leakyrelu', 'opts': {'negative_slope': 0.01, 'inplace': True}},
    'backbone': {
        'type': 'resnet101',
        'series': 'resnet',
        'pretrained': False,
        'outstride': 16,
        'use_stem': True,
        'selected_indices': (0, 1, 2, 3),
    },
    'ppm': {
        'in_channels': 2048,
        'out_channels': 512,
        'pool_scales': [1, 2, 3, 6],
    },
    'epm': {
        'in_channels_list': [256, 512, 1024],
        'hidden_channels': 256,
        'out_channels': 2
    },
    'shortcut': {
        'in_channels': 256,
        'out_channels': 48,
    },
    'decoder':{ 
        'stage1': {
            'in_channels': 560,
            'out_channels': 512,
            'dropout': 0,
        },
        'stage2': {
            'in_channels': 1280,
            'out_channels': 512,
            'dropout': 0.1
        },
    },
}
SEGMENTOR_CFG.update(
    {
        'num_classes': 20,
        'backbone': {
            'type': 'resnet50',
            'series': 'resnet',
            'pretrained': True,
            'outstride': 8,
            'use_stem': True,
            'selected_indices': (0, 1, 2, 3),
        }
    }
)


'''FaceSegmentor'''
class FaceSegmentor(nn.Module):
    def __init__(self, **kwargs):
        super(FaceSegmentor, self).__init__()
        try:
            from ssseg.modules.models.segmentors.ce2p import CE2P
        except:
            raise RuntimeError('Please run "pip install sssegmentation" to install "ssseg"')
        self.ce2p = CE2P(SEGMENTOR_CFG, mode='TEST')
        self.ce2p.load_state_dict(model_zoo.load_url('https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_ce2p/ce2p_resnet50os8_lip_train.pth')['model'])
    '''forward'''
    def forward(self, x):
        return self.ce2p(x)
    '''preprocess'''
    def preprocess(self, image):
        # Resize
        output_size = (473, 473)
        if image.shape[0] > image.shape[1]:
            dsize = min(output_size), max(output_size)
        else:
            dsize = max(output_size), min(output_size)
        image = cv2.resize(image, dsize=dsize, interpolation=cv2.INTER_LINEAR)
        # Normalize
        mean, std = np.array([123.675, 116.28, 103.53]), np.array([58.395, 57.12, 57.375])
        image = image.astype(np.float32)
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
        cv2.subtract(image, mean, image)
        cv2.multiply(image, stdinv, image)
        # ToTensor
        image = torch.from_numpy((image.transpose((2, 0, 1))).astype(np.float32))
        # Return
        return image.unsqueeze(0)
    '''get face mask'''
    def getfacemask(self, mask):
        output_mask = np.zeros(mask.shape[:2])
        face_idxs = [2, 13]
        for idx in face_idxs:
            output_mask[mask == idx] = 255
        return output_mask