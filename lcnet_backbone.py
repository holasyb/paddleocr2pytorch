# -*- coding: utf-8 -*-

# crnn model mbv3

from __future__ import absolute_import, division, print_function

import os
import random
import shutil
import time
import traceback
from collections import OrderedDict
from importlib import import_module
from sys import api_version
import sys
import cv2
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
import pathlib
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__.parent.parent))
print(sys.path)
# print(__dir__)
# from rec.RecDataSet import RecDataProcess
# from rec.label_convert import CTCLabelConverter


class HSwish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class HardSigmoid(nn.Module):
    def __init__(self, type):
        super().__init__()
        self.type = type

    def forward(self, x):
        if self.type == 'paddle':
            x = (1.2 * x).add_(3.).clamp_(0., 6.).div_(6.)
        else:
            x = F.relu6(x + 3, inplace=True) / 6
        return x


class HSigmoid(nn.Module):
    def forward(self, x):
        x = (1.2 * x).add_(3.).clamp_(0., 6.).div_(6.)
        return x



class ConvBNLayer(nn.Module):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 padding,
                 channels=None,
                 num_groups=1,
                 act='hard_swish'):
        super().__init__()
        # print(in_channels, out_channels, kernel_size, stride, padding, groups, act)
        self.conv = nn.Conv2d(in_channels=num_channels, out_channels=num_filters, kernel_size=filter_size,
                              stride=stride, padding=padding, groups=num_groups,
                              bias=False)
        self.bn = nn.BatchNorm2d(num_filters)
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'hard_swish':
            self.act = HSwish()
        elif act is None:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

class HardSigmoid(nn.Module):
    def __init__(self, type):
        super().__init__()
        self.type = type

    def forward(self, x):
        if self.type == 'paddle':
            x = (1.2 * x).add_(3.).clamp_(0., 6.).div_(6.)
        else:
            x = F.relu6(x + 3, inplace=True) / 6
        return x

class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        num_mid_filter = channel // reduction
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=num_mid_filter, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_mid_filter, kernel_size=1, out_channels=channel, bias=True)
        self.relu2 = HardSigmoid('paddle')

    def forward(self, x):
        attn = self.pool(x)
        attn = self.conv1(attn)
        attn = self.relu1(attn)
        attn = self.conv2(attn)
        attn = self.relu2(attn)
        return x * attn



class DepthwiseSeparable(nn.Module):
    def __init__(self,
                 num_channels,
                 num_filters1,
                 num_filters2,
                 num_groups,
                 stride,
                 scale,
                 dw_size=3,
                 padding=1,
                 use_se=False):
        super().__init__()
        self.use_se = use_se
        self._depthwise_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=int(num_filters1 * scale),
            filter_size=dw_size,
            stride=stride,
            padding=padding,
            num_groups=int(num_groups * scale))

        if use_se:
            self._se = SEModule(int(num_filters1 * scale))
        self._pointwise_conv = ConvBNLayer(
            num_channels=int(num_filters1 * scale),
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0)

    def forward(self, inputs):
        y = self._depthwise_conv(inputs)
        if self.use_se:
            y = self._se(y)
        y = self._pointwise_conv(y)
        return y


class MobileNetV1Enhance(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()
        self.scale = kwargs.get('scale', 0.5)
        scale = kwargs.get('scale', 0.5)
        self.block_list = []

        self.conv1 = ConvBNLayer(
            num_channels=3,
            filter_size=3,
            channels=3,
            num_filters=int(32 * scale),
            stride=2,
            padding=1)

        conv2_1 = DepthwiseSeparable(
            num_channels=int(32 * scale),
            num_filters1=32,
            num_filters2=64,
            num_groups=32,
            stride=1,
            scale=scale)
        self.block_list.append(conv2_1)

        conv2_2 = DepthwiseSeparable(
            num_channels=int(64 * scale),
            num_filters1=64,
            num_filters2=128,
            num_groups=64,
            stride=1,
            scale=scale)
        self.block_list.append(conv2_2)

        conv3_1 = DepthwiseSeparable(
            num_channels=int(128 * scale),
            num_filters1=128,
            num_filters2=128,
            num_groups=128,
            stride=1,
            scale=scale)
        self.block_list.append(conv3_1)

        conv3_2 = DepthwiseSeparable(
            num_channels=int(128 * scale),
            num_filters1=128,
            num_filters2=256,
            num_groups=128,
            stride=(2, 1),
            scale=scale)
        self.block_list.append(conv3_2)

        conv4_1 = DepthwiseSeparable(
            num_channels=int(256 * scale),
            num_filters1=256,
            num_filters2=256,
            num_groups=256,
            stride=1,
            scale=scale)
        self.block_list.append(conv4_1)

        conv4_2 = DepthwiseSeparable(
            num_channels=int(256 * scale),
            num_filters1=256,
            num_filters2=512,
            num_groups=256,
            stride=(2, 1),
            scale=scale)
        self.block_list.append(conv4_2)

        for _ in range(5):
            conv5 = DepthwiseSeparable(
                num_channels=int(512 * scale),
                num_filters1=512,
                num_filters2=512,
                num_groups=512,
                stride=1,
                dw_size=5,
                padding=2,
                scale=scale,
                use_se=False)
            self.block_list.append(conv5)

        conv5_6 = DepthwiseSeparable(
            num_channels=int(512 * scale),
            num_filters1=512,
            num_filters2=1024,
            num_groups=512,
            stride=(2, 1),
            dw_size=5,
            padding=2,
            scale=scale,
            use_se=True)
        self.block_list.append(conv5_6)

        conv6 = DepthwiseSeparable(
            num_channels=int(1024 * scale),
            num_filters1=1024,
            num_filters2=1024,
            num_groups=1024,
            stride=1,
            dw_size=5,
            padding=2,
            use_se=True,
            scale=scale)
        self.block_list.append(conv6)

        self.block_list = nn.Sequential(*self.block_list)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.out_channels = int(1024 * scale)

    def forward(self, inputs):
        y = self.conv1(inputs)
        y = self.block_list(y)
        y = self.pool(y)
        return y

a = MobileNetV1Enhance(scale=0.5)


class Im2Seq(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        # print(x.shape)
        B, C, H, W = x.shape
        assert H == 1
        x = x.reshape(B, C, H * W)
        x = x.permute((0, 2, 1))
        return x


class EncoderWithRNN(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(EncoderWithRNN, self).__init__()
        hidden_size = kwargs.get('hidden_size', 256)
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(in_channels, hidden_size,
                            bidirectional=True, num_layers=2, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class SequenceEncoder(nn.Module):
    def __init__(self, in_channels, encoder_type='rnn',  **kwargs):
        super(SequenceEncoder, self).__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        if encoder_type == 'reshape':
            self.only_reshape = True
        else:
            support_encoder_dict = {
                'reshape': Im2Seq,
                'rnn': EncoderWithRNN
            }
            assert encoder_type in support_encoder_dict, '{} must in {}'.format(
                encoder_type, support_encoder_dict.keys())

            self.encoder = support_encoder_dict[encoder_type](
                self.encoder_reshape.out_channels, **kwargs)
            self.out_channels = self.encoder.out_channels
            self.only_reshape = False

    def forward(self, x):
        x = self.encoder_reshape(x)
        if not self.only_reshape:
            x = self.encoder(x)

        return x


class CTC(nn.Module):
    def __init__(self, in_channels, mid_channels, n_class, **kwargs):
        super().__init__()
        # print(in_channels, n_class)
        # print(asdasd)# 96 6625
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, n_class)
        self.n_class = n_class

    def forward(self, x):
        # print('head fc', x.shape)
        y = self.fc1(x)
        return self.fc2(y)


class CTCLoss(nn.Module):

    def __init__(self, blank_idx, reduction='mean'):
        super().__init__()
        self.loss_func = torch.nn.CTCLoss(
            blank=blank_idx, reduction=reduction, zero_infinity=True)

    def forward(self, pred, args):
        batch_size = pred.size(0)
        label, label_length = args['targets'], args['targets_lengths']
        pred = pred.log_softmax(2)
        pred = pred.permute(1, 0, 2)
        preds_lengths = torch.tensor(
            [pred.size(0)] * batch_size, dtype=torch.long)
        loss = self.loss_func(pred, label, preds_lengths, label_length)
        return {'loss': loss}


class CrnnModel(nn.Module):
    def __init__(self, in_channels=3, backbone_out_channels=512, hidden_size=64, n_classes=7000):
        super().__init__()
        self.in_channels = in_channels
        self.backbone = MobileNetV1Enhance(self.in_channels)
        self.neck = SequenceEncoder(backbone_out_channels, hidden_size=hidden_size)
        self.head = CTC(in_channels=hidden_size * 2, mid_channels=96, n_class=n_classes)


    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        # print('after neck', x.shape)
        x = self.head(x)
        return x

# MobileNetV3(in_channels=3).cuda()

class rec_handle():
	def __init__(self, cfg_path, alphabet_path, model_path='rec/crnn_mbv3.pth'):
		print('load')
		mod = import_module(cfg_path)
		self.config = mod.config
		self.alphabet = alphabet_path# 'keys_6625.txt'
		self.alphbet = open(self.alphabet,'r',encoding='utf8').read().strip().split('\n')
		if ' ' not in self.alphbet:
			self.alphbet.append(' ')
		self.n_class = len(self.alphbet) + 1
		self.config['model']['head']['n_class'] = self.n_class
		# print(self.config['model']['head']['n_class'])
		self.process = RecDataProcess(self.config['dataset']['eval']['dataset'])
		self.converter = CTCLabelConverter(self.alphabet)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.load_state_dict(model_path)
		# self.
		
	
	def load_state_dict(self, model_path):
		self.crnn = CrnnModel(n_classes=self.n_class)
		ckpt = torch.load(model_path, map_location='cpu')# ['state_dict']
		state_dict = {}
		for k, v in ckpt['state_dict'].items():
			state_dict[k.replace('module.', '')] = v
		self.crnn.load_state_dict(state_dict)
		self.crnn.to(self.device)
		self.crnn.eval()
	
	def data_process(self, imgs):
		# [print(img.shape) for img in imgs]# 二维
		return [self.process.normalize_img(self.process.resize_with_specific_height(img)) for img in imgs]
		
	def rec(self, imgs):
		#[print('asdasda', img.shape) for img in imgs]
		
		if not isinstance(imgs, list):
			imgs = [imgs]
		# avg_color = []
		# avg_color = [np.mean(np.mean(img, axis = 0), axis = 0) for img in imgs]
		imgs = self.data_process(imgs)
		widths = np.array([img.shape[1] for img in imgs])
		idxs = np.argsort(widths)
		txts = []
		batch_size = 4
		for idx in range(0, len(imgs), batch_size):
		    batch_idxs = idxs[idx:min(len(imgs), idx+batch_size)]
		    batch_imgs = [self.process.width_pad_img(imgs[idx], imgs[batch_idxs[-1]].shape[1]) for idx in batch_idxs]
		    batch_imgs = np.stack(batch_imgs)
		    tensor = torch.from_numpy(batch_imgs.transpose([0,3, 1, 2])).float()
		    tensor = tensor.to(self.device)
		    with torch.no_grad():
		        out = self.crnn(tensor)
		        out = out.softmax(dim=2)
		    out = out.cpu().numpy()
		    txts.extend([self.converter.decode(np.expand_dims(txt, 0)) for txt in out])
		    
		#按输入图像的顺序排序
		idxs = np.argsort(idxs)
		out_txts = [txts[idx] for idx in idxs]
		return out_txts
		# print(out_txts)
		
