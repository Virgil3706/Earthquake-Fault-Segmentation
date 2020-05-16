import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from model_zoo import vgg16_c

from model_zoo.bdcn import MSBlock, get_upsampling_weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os, sys
import torch
import torch.nn as nn
import torchvision.models as models
import torch.autograd.variable as Variable
import numpy as np
import scipy.io as sio
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.conv import _single, _pair, _triple
import torch.nn.functional as F
from torchsummary import summary




# coding: utf-8

class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.25),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.25))

    def forward(self, x):
        x = self.conv(x)
        return x


start_fm = 32


class Unet(nn.Module):

    def __init__(self):
        super(Unet, self).__init__()



        # Input 72*72*fm

        # Contracting Path

        # (Double) Convolution 1
        self.double_conv1 = double_conv(1, start_fm, 3, 1, 1)  # 72*72*fm
        # Max Pooling 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # 36*36*fm

        # Convolution 2
        self.double_conv2 = double_conv(start_fm, start_fm * 2, 3, 1, 1)  # 36*36*fm*2
        # Max Pooling 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)  # 18*18*fm*2

        # Convolution 3
        self.double_conv3 = double_conv(start_fm * 2, start_fm * 4, 3, 1, 1)  # 18*18*fm*4
        # Max Pooling 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)  # 9*9*fm*4

        # Convolution 4
        self.double_conv4 = double_conv(start_fm * 4, start_fm * 8, 3, 1, 2)  # 9*9*fm*8

        # Transposed Convolution 3
        self.t_conv3 = nn.ConvTranspose2d(start_fm * 8, start_fm * 4, 2, 2)  # 96*152*64
        # Convolution 3
        self.ex_double_conv3 = double_conv(start_fm * 8, start_fm * 4, 3, 1, 2)  # 96*152*64

        # Transposed Convolution 2
        self.t_conv2 = nn.ConvTranspose2d(start_fm * 4, start_fm * 2, 2, 2)  # 288*456*32
        # Convolution 2
        self.ex_double_conv2 = double_conv(start_fm * 4, start_fm * 2, 3, 1, 2)  # 288*456*32

        # Transposed Convolution 1
        self.t_conv1 = nn.ConvTranspose2d(start_fm * 2, start_fm, 2, 2)  # 864*1368*16
        # Convolution 1
        self.ex_double_conv1 = double_conv(start_fm * 2, start_fm, 3, 1, 2)  # 864*1368*16

        # One by One Conv
        self.one_by_one = nn.Conv2d(start_fm, 1, 1, 1, 0)
        # self.final_act = nn.Sigmoid()

    def forward(self, inputs):

        def crop(variable, th, tw):
            h, w = variable.shape[2], variable.shape[3]
            x1 = int(round((w - tw) / 2.))
            y1 = int(round((h - th) / 2.))
            return variable[:, :, y1: y1 + th, x1: x1 + tw]


        # Contracting Path

        img_H, img_W = inputs.shape[2], inputs.shape[3]
        conv1 = self.double_conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.double_conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.double_conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        # Bottom
        conv4 = self.double_conv4(maxpool3)

        #         print(conv1.shape, maxpool1.shape, conv2.shape, maxpool2.shape, conv3.shape, maxpool3.shape, conv4.shape)

        # Expanding Path
        t_conv3 = self.t_conv3(conv4)
        #         print(t_conv3.shape)
        h, w = conv3.shape[2], conv3.shape[3]
        t_conv3 = crop(t_conv3, h, w)
        #         print(t_conv3.shape)
        cat3 = torch.cat([conv3, t_conv3], 1)
        ex_conv3 = self.ex_double_conv3(cat3)

        t_conv2 = self.t_conv2(ex_conv3)
        #         print(t_conv2.shape)
        h, w = conv2.shape[2], conv2.shape[3]
        t_conv2 = crop(t_conv2, h, w)
        #         print(t_conv2.shape)
        cat2 = torch.cat([conv2, t_conv2], 1)
        ex_conv2 = self.ex_double_conv2(cat2)

        t_conv1 = self.t_conv1(ex_conv2)
        #         print(t_conv1.shape)

        h, w = conv1.shape[2], conv1.shape[3]
        t_conv1 = crop(t_conv1, h, w)
        #         print(t_conv1.shape)
        #         print(conv1.shape)
        cat1 = torch.cat([conv1, t_conv1], 1)
        ex_conv1 = self.ex_double_conv1(cat1)
        #         print("ex_conv1", ex_conv1.shape)

        one_by_one = self.one_by_one(ex_conv1)
        one_by_one = crop(one_by_one, img_H, img_W)
        #         print(one_by_one.shape)


        return torch.sigmoid(one_by_one)


# model = Unet()
# model.to(device);
# summary(model, (3, 128, 128))


# coding: utf-8



class HED(nn.Module):
    def __init__(self):
        super(HED, self).__init__()
        # lr 1 2 decay 1 0
        self.dropout = nn.Dropout(0.2)
        self.conv1_1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv5_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.maxpool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)

        # lr 0.1 0.2 decay 1 0
        #         self.conv1_1_down = nn.Conv2d(64, 21, 1, padding=0)
        self.conv1_2_down = nn.Conv2d(32, 21, 1, padding=0)

        #         self.conv2_1_down = nn.Conv2d(128, 21, 1, padding=0)
        self.conv2_2_down = nn.Conv2d(64, 21, 1, padding=0)

        #         self.conv3_1_down = nn.Conv2d(256, 21, 1, padding=0)
        #         self.conv3_2_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv3_3_down = nn.Conv2d(128, 21, 1, padding=0)

        #         self.conv4_1_down = nn.Conv2d(512, 21, 1, padding=0)
        #         self.conv4_2_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv4_3_down = nn.Conv2d(256, 21, 1, padding=0)

        #         self.conv5_1_down = nn.Conv2d(512, 21, 1, padding=0)
        #         self.conv5_2_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv5_3_down = nn.Conv2d(256, 21, 1, padding=0)

        # lr 0.01 0.02 decay 1 0
        self.score_dsn1 = nn.Conv2d(21, 1, 1)
        self.score_dsn2 = nn.Conv2d(21, 1, 1)
        self.score_dsn3 = nn.Conv2d(21, 1, 1)
        self.score_dsn4 = nn.Conv2d(21, 1, 1)
        self.score_dsn5 = nn.Conv2d(21, 1, 1)
        # lr 0.001 0.002 decay 1 0
        self.score_final = nn.Conv2d(5, 1, 1)

    def forward(self, x):

        def crop(variable, th, tw):
            h, w = variable.shape[2], variable.shape[3]
            x1 = int(round((w - tw) / 2.))
            y1 = int(round((h - th) / 2.))
            return variable[:, :, y1: y1 + th, x1: x1 + tw]

        def make_bilinear_weights(size, num_channels):
            factor = (size + 1) // 2
            if size % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            og = np.ogrid[:size, :size]
            filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
            # print(filt)
            filt = torch.from_numpy(filt)
            w = torch.zeros(num_channels, num_channels, size, size)
            w.requires_grad = False
            for i in range(num_channels):
                for j in range(num_channels):
                    if i == j:
                        w[i, j] = filt
            return w


        # VGG
        img_H, img_W = x.shape[2], x.shape[3]
        conv1_1 = self.dropout(self.relu(self.conv1_1(x)))
        conv1_2 = self.dropout(self.relu(self.conv1_2(conv1_1)))
        pool1 = self.maxpool(conv1_2)

        conv2_1 = self.dropout(self.relu(self.conv2_1(pool1)))
        conv2_2 = self.dropout(self.relu(self.conv2_2(conv2_1)))
        #         conv2_2 = self.relu(self.conv2_2(pool1))
        pool2 = self.maxpool(conv2_2)

        conv3_1 = self.dropout(self.relu(self.conv3_1(pool2)))
        conv3_2 = self.dropout(self.relu(self.conv3_2(conv3_1)))
        conv3_3 = self.dropout(self.relu(self.conv3_3(conv3_2)))
        #         conv3_3 = self.relu(self.conv3_3(pool2))
        pool3 = self.maxpool(conv3_3)

        conv4_1 = self.dropout(self.relu(self.conv4_1(pool3)))
        conv4_2 = self.dropout(self.relu(self.conv4_2(conv4_1)))
        conv4_3 = self.dropout(self.relu(self.conv4_3(conv4_2)))
        #         conv4_3 = self.relu(self.conv4_3(pool3))
        pool4 = self.maxpool4(conv4_3)

        conv5_1 = self.dropout(self.relu(self.conv5_1(pool4)))
        conv5_2 = self.dropout(self.relu(self.conv5_2(conv5_1)))
        conv5_3 = self.dropout(self.relu(self.conv5_3(conv5_2)))
        #         conv5_3 = self.relu(self.conv5_3(pool4))

        #         conv1_1_down = self.conv1_1_down(conv1_1)
        conv1_2_down = self.dropout(self.conv1_2_down(conv1_2))
        #         conv2_1_down = self.conv2_1_down(conv2_1)
        conv2_2_down = self.dropout(self.conv2_2_down(conv2_2))
        #         conv3_1_down = self.conv3_1_down(conv3_1)
        #         conv3_2_down = self.conv3_2_down(conv3_2)
        conv3_3_down = self.dropout(self.conv3_3_down(conv3_3))
        #         conv4_1_down = self.conv4_1_down(conv4_1)
        #         conv4_2_down = self.conv4_2_down(conv4_2)
        conv4_3_down = self.dropout(self.conv4_3_down(conv4_3))
        #         conv5_1_down = self.conv5_1_down(conv5_1)
        #         conv5_2_down = self.conv5_2_down(conv5_2)
        conv5_3_down = self.dropout(self.conv5_3_down(conv5_3))

        so1_out = self.score_dsn1(conv1_2_down)
        so2_out = self.score_dsn2(conv2_2_down)
        so3_out = self.score_dsn3(conv3_3_down)
        so4_out = self.score_dsn4(conv4_3_down)
        so5_out = self.score_dsn5(conv5_3_down)

        #         so1_out = self.score_dsn1(conv1_1_down + conv1_2_down)
        #         so2_out = self.score_dsn2(conv2_1_down + conv2_2_down)
        #         so3_out = self.score_dsn3(conv3_1_down + conv3_2_down + conv3_3_down)
        #         so4_out = self.score_dsn4(conv4_1_down + conv4_2_down + conv4_3_down)
        #         so5_out = self.score_dsn5(conv5_1_down + conv5_2_down + conv5_3_down)

        ## transpose and crop way
        weight_deconv2 = make_bilinear_weights(4, 1).to(device)
        weight_deconv3 = make_bilinear_weights(8, 1).to(device)
        weight_deconv4 = make_bilinear_weights(16, 1).to(device)
        weight_deconv5 = make_bilinear_weights(32, 1).to(device)

        upsample2 = torch.nn.functional.conv_transpose2d(so2_out, weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(so3_out, weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(so4_out, weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(so5_out, weight_deconv5, stride=8)
        ### center crop
        so1 = crop(so1_out, img_H, img_W)
        so2 = crop(upsample2, img_H, img_W)
        so3 = crop(upsample3, img_H, img_W)
        so4 = crop(upsample4, img_H, img_W)
        so5 = crop(upsample5, img_H, img_W)
        ### crop way suggested by liu
        # so1 = crop_caffe(0, so1, img_H, img_W)
        # so2 = crop_caffe(1, upsample2, img_H, img_W)
        # so3 = crop_caffe(2, upsample3, img_H, img_W)
        # so4 = crop_caffe(4, upsample4, img_H, img_W)
        # so5 = crop_caffe(8, upsample5, img_H, img_W)
        ## upsample way
        # so1 = F.upsample_bilinear(so1, size=(img_H,img_W))
        # so2 = F.upsample_bilinear(so2, size=(img_H,img_W))
        # so3 = F.upsample_bilinear(so3, size=(img_H,img_W))
        # so4 = F.upsample_bilinear(so4, size=(img_H,img_W))
        # so5 = F.upsample_bilinear(so5, size=(img_H,img_W))

        fusecat = torch.cat((so1, so2, so3, so4, so5), dim=1)
        fuse = self.score_final(fusecat)
        results = [so1, so2, so3, so4, so5, fuse]
        results = [torch.sigmoid(r) for r in results]
        return results



# model = HED()
# model.to(device);
# summary(model, (1, 92, 92))


class DilateConv(nn.Module):
    """
    d_rate: dilation rate
    H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)
    set kernel size to 3, stride to 1, padding==d_rate ==> spatial size kept
    """

    def __init__(self, d_rate, in_ch, out_ch):
        super(DilateConv, self).__init__()
        self.d_conv = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                                stride=1, padding=d_rate, dilation=d_rate)

    def forward(self, x):
        return self.d_conv(x)


class RCF(nn.Module):
    def __init__(self):
        super(RCF, self).__init__()
        # lr 1 2 decay 1 0
        self.conv1_1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_3 = nn.Conv2d(256, 256, 3, padding=1)
        # lr 100 200 decay 1 0
        # self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        # self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        # self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)

        # self.conv5_1 = DilateConv(d_rate=2, in_ch=512, out_ch=512) # error ! name conv5_1.dconv.weight erro in load vgg16
        # self.conv5_2 = DilateConv(d_rate=2, in_ch=512, out_ch=512)
        # self.conv5_3 = DilateConv(d_rate=2, in_ch=512, out_ch=512)

        self.conv5_1 = nn.Conv2d(256, 256, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(256, 256, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.maxpool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)

        # lr 0.1 0.2 decay 1 0
        self.conv1_1_down = nn.Conv2d(32, 21, 1, padding=0)
        self.conv1_2_down = nn.Conv2d(32, 21, 1, padding=0)

        self.conv2_1_down = nn.Conv2d(64, 21, 1, padding=0)
        self.conv2_2_down = nn.Conv2d(64, 21, 1, padding=0)

        self.conv3_1_down = nn.Conv2d(128, 21, 1, padding=0)
        self.conv3_2_down = nn.Conv2d(128, 21, 1, padding=0)
        self.conv3_3_down = nn.Conv2d(128, 21, 1, padding=0)

        self.conv4_1_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv4_2_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv4_3_down = nn.Conv2d(256, 21, 1, padding=0)

        self.conv5_1_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv5_2_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv5_3_down = nn.Conv2d(256, 21, 1, padding=0)

        # lr 0.01 0.02 decay 1 0
        self.score_dsn1 = nn.Conv2d(21, 1, 1)
        self.score_dsn2 = nn.Conv2d(21, 1, 1)
        self.score_dsn3 = nn.Conv2d(21, 1, 1)
        self.score_dsn4 = nn.Conv2d(21, 1, 1)
        self.score_dsn5 = nn.Conv2d(21, 1, 1)
        # lr 0.001 0.002 decay 1 0
        self.score_final = nn.Conv2d(5, 1, 1)
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):

        def crop(variable, th, tw):
            h, w = variable.shape[2], variable.shape[3]
            x1 = int(round((w - tw) / 2.))
            y1 = int(round((h - th) / 2.))
            return variable[:, :, y1: y1 + th, x1: x1 + tw]

        def crop_caffe(location, variable, th, tw):
            h, w = variable.shape[2], variable.shape[3]
            x1 = int(location)
            y1 = int(location)
            return variable[:, :, y1: y1 + th, x1: x1 + tw]

        # make a bilinear interpolation kernel
        def upsample_filt(size):
            factor = (size + 1) // 2
            if size % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            og = np.ogrid[:size, :size]
            return (1 - abs(og[0] - center) / factor) * \
                   (1 - abs(og[1] - center) / factor)

        # set parameters s.t. deconvolutional layers compute bilinear interpolation
        # N.B. this is for deconvolution without groups
        def interp_surgery(in_channels, out_channels, h, w):
            weights = np.zeros([in_channels, out_channels, h, w])
            if in_channels != out_channels:
                raise ValueError("Input Output channel!")
            if h != w:
                raise ValueError("filters need to be square!")
            filt = upsample_filt(h)
            weights[range(in_channels), range(out_channels), :, :] = filt
            return np.float32(weights)

        def make_bilinear_weights(size, num_channels):
            factor = (size + 1) // 2
            if size % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            og = np.ogrid[:size, :size]
            filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
            # print(filt)
            filt = torch.from_numpy(filt)
            w = torch.zeros(num_channels, num_channels, size, size)
            w.requires_grad = False
            for i in range(num_channels):
                for j in range(num_channels):
                    if i == j:
                        w[i, j] = filt
            return w

        def upsample(input, stride, num_channels=1):
            kernel_size = stride * 2
            kernel = make_bilinear_weights(kernel_size, num_channels).to(device)
            return torch.nn.functional.conv_transpose2d(input, kernel, stride=stride)


        # VGG
        img_H, img_W = x.shape[2], x.shape[3]
        conv1_1 = self.dropout(self.relu(self.conv1_1(x)))
        conv1_2 = self.dropout(self.relu(self.conv1_2(conv1_1)))
        pool1 = self.maxpool(conv1_2)

        conv2_1 = self.dropout(self.relu(self.conv2_1(pool1)))
        conv2_2 = self.dropout(self.relu(self.conv2_2(conv2_1)))
        pool2 = self.maxpool(conv2_2)

        conv3_1 = self.dropout(self.relu(self.conv3_1(pool2)))
        conv3_2 = self.dropout(self.relu(self.conv3_2(conv3_1)))
        conv3_3 = self.dropout(self.relu(self.conv3_3(conv3_2)))
        pool3 = self.maxpool(conv3_3)

        conv4_1 = self.dropout(self.relu(self.conv4_1(pool3)))
        conv4_2 = self.dropout(self.relu(self.conv4_2(conv4_1)))
        conv4_3 = self.dropout(self.relu(self.conv4_3(conv4_2)))
        pool4 = self.maxpool4(conv4_3)

        conv5_1 = self.dropout(self.relu(self.conv5_1(pool4)))
        conv5_2 = self.dropout(self.relu(self.conv5_2(conv5_1)))
        conv5_3 = self.dropout(self.relu(self.conv5_3(conv5_2)))

        conv1_1_down = self.conv1_1_down(conv1_1)
        conv1_2_down = self.conv1_2_down(conv1_2)
        conv2_1_down = self.conv2_1_down(conv2_1)
        conv2_2_down = self.conv2_2_down(conv2_2)
        conv3_1_down = self.conv3_1_down(conv3_1)
        conv3_2_down = self.conv3_2_down(conv3_2)
        conv3_3_down = self.conv3_3_down(conv3_3)
        conv4_1_down = self.conv4_1_down(conv4_1)
        conv4_2_down = self.conv4_2_down(conv4_2)
        conv4_3_down = self.conv4_3_down(conv4_3)
        conv5_1_down = self.conv5_1_down(conv5_1)
        conv5_2_down = self.conv5_2_down(conv5_2)
        conv5_3_down = self.conv5_3_down(conv5_3)

        so1_out = self.score_dsn1(conv1_1_down + conv1_2_down)
        so2_out = self.score_dsn2(conv2_1_down + conv2_2_down)
        so3_out = self.score_dsn3(conv3_1_down + conv3_2_down + conv3_3_down)
        so4_out = self.score_dsn4(conv4_1_down + conv4_2_down + conv4_3_down)
        so5_out = self.score_dsn5(conv5_1_down + conv5_2_down + conv5_3_down)
        ## transpose and crop way
        weight_deconv2 = make_bilinear_weights(4, 1).to(device)
        weight_deconv3 = make_bilinear_weights(8, 1).to(device)
        weight_deconv4 = make_bilinear_weights(16, 1).to(device)
        weight_deconv5 = make_bilinear_weights(32, 1).to(device)

        upsample2 = torch.nn.functional.conv_transpose2d(so2_out, weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(so3_out, weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(so4_out, weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(so5_out, weight_deconv5, stride=8)
        ### center crop
        so1 = crop(so1_out, img_H, img_W)
        so2 = crop(upsample2, img_H, img_W)
        so3 = crop(upsample3, img_H, img_W)
        so4 = crop(upsample4, img_H, img_W)
        so5 = crop(upsample5, img_H, img_W)
        ### crop way suggested by liu
        # so1 = crop_caffe(0, so1, img_H, img_W)
        # so2 = crop_caffe(1, upsample2, img_H, img_W)
        # so3 = crop_caffe(2, upsample3, img_H, img_W)
        # so4 = crop_caffe(4, upsample4, img_H, img_W)
        # so5 = crop_caffe(8, upsample5, img_H, img_W)
        ## upsample way
        # so1 = F.upsample_bilinear(so1, size=(img_H,img_W))
        # so2 = F.upsample_bilinear(so2, size=(img_H,img_W))
        # so3 = F.upsample_bilinear(so3, size=(img_H,img_W))
        # so4 = F.upsample_bilinear(so4, size=(img_H,img_W))
        # so5 = F.upsample_bilinear(so5, size=(img_H,img_W))

        fusecat = torch.cat((so1, so2, so3, so4, so5), dim=1)
        fuse = self.score_final(fusecat)
        results = [so1, so2, so3, so4, so5, fuse]
        results = [torch.sigmoid(r) for r in results]
        return results



# model = RCF()
# model.to(device);
# summary(model, (1, 100, 100))



import torch
import torch.nn as nn
import torch.nn.functional as F
from model_zoo.DEEPLAB.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from model_zoo.DEEPLAB.aspp import build_aspp
from model_zoo.DEEPLAB.decoder import build_decoder
from model_zoo.DEEPLAB.backbone import build_backbone
# from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
# from aspp import build_aspp
# from decoder import build_decoder
# from backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return torch.sigmoid(x)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p



#BDCN
import numpy as np
import torch
import torch.nn as nn

import model_zoo.vgg16_c



import numpy as np
import torch
import torch.nn as nn

#import vgg16_c
from model_zoo import vgg16_c


def crop(data1, data2, crop_h, crop_w):
    _, _, h1, w1 = data1.size()
    _, _, h2, w2 = data2.size()
    assert(h2 <= h1 and w2 <= w1)
    data = data1[:, :, crop_h:crop_h+h2, crop_w:crop_w+w2]
    return data

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()
#好像是SEM的part？
class MSBlock(nn.Module):
    #def _init_(self,rate=4):
    def __init__(self, c_in, rate=4):
        super(MSBlock, self).__init__()
        c_out = c_in
        #c_out=1
        self.rate = rate

        self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        #self.conv = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        dilation = self.rate*1 if self.rate >= 1 else 1
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu1 = nn.ReLU(inplace=True)
        dilation = self.rate*2 if self.rate >= 1 else 1
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu2 = nn.ReLU(inplace=True)
        dilation = self.rate*3 if self.rate >= 1 else 1
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu3 = nn.ReLU(inplace=True)

        self._initialize_weights()

    def forward(self, x):
        o = self.relu(self.conv(x))
        o1 = self.relu1(self.conv1(o))
        o2 = self.relu2(self.conv2(o))
        o3 = self.relu3(self.conv3(o))
        out = o + o1 + o2 + o3
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

#这里之所以没有池化是因为这是基于vgg16上的框架弄的，框架上弄了池化操作所以在这个上面就不用再弄
class BDCN(nn.Module):
    def __init__(self, pretrain=None, logger=None, rate=4):
        super(BDCN, self).__init__()
        self.pretrain = pretrain
        t = 1
        #
        self.features = vgg16_c.VGG16_C(pretrain, logger)
        self.msblock1_1 = MSBlock(64, rate)
        self.msblock1_2 = MSBlock(64, rate)
        self.conv1_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv1_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.score_dsn1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn1_1 = nn.Conv2d(21, 1, 1, stride=1)
        self.msblock2_1 = MSBlock(128, rate)
        self.msblock2_2 = MSBlock(128, rate)
        self.conv2_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv2_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.score_dsn2 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn2_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.msblock3_1 = MSBlock(256, rate)
        self.msblock3_2 = MSBlock(256, rate)
        self.msblock3_3 = MSBlock(256, rate)
        self.conv3_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_3_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.score_dsn3 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn3_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.msblock4_1 = MSBlock(512, rate)
        self.msblock4_2 = MSBlock(512, rate)
        self.msblock4_3 = MSBlock(512, rate)
        self.conv4_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv4_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv4_3_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.score_dsn4 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn4_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.msblock5_1 = MSBlock(512, rate)
        self.msblock5_2 = MSBlock(512, rate)
        self.msblock5_3 = MSBlock(512, rate)
        self.conv5_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv5_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv5_3_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.score_dsn5 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn5_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.upsample_2 = nn.ConvTranspose2d(1, 1, 4, stride=2, bias=False)
        self.upsample_4 = nn.ConvTranspose2d(1, 1, 8, stride=4, bias=False)
        self.upsample_8 = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        self.upsample_8_5 = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        self.fuse = nn.Conv2d(10, 1, 1, stride=1)
        # self.features = vgg16_c.VGG16_C(pretrain, logger)
        # self.msblock1_1 = MSBlock(32, rate)
        # self.msblock1_2 = MSBlock(32, rate)
        # self.conv1_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)#kernel_size=(1,1)
        # self.conv1_2_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        # self.score_dsn1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        # self.score_dsn1_1 = nn.Conv2d(21, 1, 1, stride=1)
        # self.msblock2_1 = MSBlock(64, rate)
        # self.msblock2_2 = MSBlock(64, rate)
        # self.conv2_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        # self.conv2_2_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        # self.score_dsn2 = nn.Conv2d(21, 1, (1, 1), stride=1)
        # self.score_dsn2_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        # self.msblock3_1 = MSBlock(128, rate)
        # self.msblock3_2 = MSBlock(128, rate)
        # self.msblock3_3 = MSBlock(128, rate)
        # self.conv3_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        # self.conv3_2_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        # self.conv3_3_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        # self.score_dsn3 = nn.Conv2d(21, 1, (1, 1), stride=1)
        # self.score_dsn3_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        # self.msblock4_1 = MSBlock(256, rate)
        # self.msblock4_2 = MSBlock(256, rate)
        # self.msblock4_3 = MSBlock(256, rate)
        # self.conv4_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        # self.conv4_2_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        # self.conv4_3_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        # self.score_dsn4 = nn.Conv2d(21, 1, (1, 1), stride=1)
        # self.score_dsn4_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        # self.msblock5_1 = MSBlock(256, rate)
        # self.msblock5_2 = MSBlock(256, rate)
        # self.msblock5_3 = MSBlock(256, rate)
        # self.conv5_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        # self.conv5_2_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        # self.conv5_3_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        # self.score_dsn5 = nn.Conv2d(21, 1, (1, 1), stride=1)
        # self.score_dsn5_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        # self.upsample_2 = nn.ConvTranspose2d(1, 1, 4, stride=2, bias=False)
        # self.upsample_4 = nn.ConvTranspose2d(1, 1, 8, stride=4, bias=False)
        # self.upsample_8 = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        # self.upsample_8_5 = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        # #self.upsample_2=F.upsample_bilinear(self)
        # #这个代码里怎么没有含有关池化操作的
        # #self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # #self.maxpool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)
        # self.fuse = nn.Conv2d(10, 1, 1, stride=1)

        self._initialize_weights(logger)
        #print("wtf")
        #import pdb
        #pdb.set_trace()

    def forward(self, x):
        features = self.features(x)
        #print("features", features)
        import pdb
        #pdb.set_trace()
        sum1 = self.conv1_1_down(self.msblock1_1(features[0])) + \
                self.conv1_2_down(self.msblock1_2(features[1]))
        s1 = self.score_dsn1(sum1)
        s11 = self.score_dsn1_1(sum1)
        # print(s1.data.shape, s11.data.shape)
        sum2 = self.conv2_1_down(self.msblock2_1(features[2])) + \
            self.conv2_2_down(self.msblock2_2(features[3]))
        s2 = self.score_dsn2(sum2)
        s21 = self.score_dsn2_1(sum2)
        s2 = self.upsample_2(s2)
        s21 = self.upsample_2(s21)
        # print(s2.data.shape, s21.data.shape)
        s2 = crop(s2, x, 1, 1)
        s21 = crop(s21, x, 1, 1)
        sum3 = self.conv3_1_down(self.msblock3_1(features[4])) + \
            self.conv3_2_down(self.msblock3_2(features[5])) + \
            self.conv3_3_down(self.msblock3_3(features[6]))
        s3 = self.score_dsn3(sum3)
        s3 =self.upsample_4(s3)
        # print(s3.data.shape)
        s3 = crop(s3, x, 2, 2)
        s31 = self.score_dsn3_1(sum3)
        s31 =self.upsample_4(s31)
        # print(s31.data.shape)
        s31 = crop(s31, x, 2, 2)
        sum4 = self.conv4_1_down(self.msblock4_1(features[7])) + \
            self.conv4_2_down(self.msblock4_2(features[8])) + \
            self.conv4_3_down(self.msblock4_3(features[9]))
        s4 = self.score_dsn4(sum4)
        s4 = self.upsample_8(s4)
        # print(s4.data.shape)
        s4 = crop(s4, x, 4, 4)
        s41 = self.score_dsn4_1(sum4)
        s41 = self.upsample_8(s41)
        # print(s41.data.shape)
        s41 = crop(s41, x, 4, 4)
        sum5 = self.conv5_1_down(self.msblock5_1(features[10])) + \
            self.conv5_2_down(self.msblock5_2(features[11])) + \
            self.conv5_3_down(self.msblock5_3(features[12]))
        s5 = self.score_dsn5(sum5)
        s5 = self.upsample_8_5(s5)
        # print(s5.data.shape)
        s5 = crop(s5, x, 0, 0)
        s51 = self.score_dsn5_1(sum5)
        s51 = self.upsample_8_5(s51)
        # print(s51.data.shape)
        s51 = crop(s51, x, 0, 0)
        o1, o2, o3, o4, o5 = s1.detach(), s2.detach(), s3.detach(), s4.detach(), s5.detach()
        o11, o21, o31, o41, o51 = s11.detach(), s21.detach(), s31.detach(), s41.detach(), s51.detach()
        p1_1 = s1
        p2_1 = s2 + o1
        p3_1 = s3 + o2 + o1
        p4_1 = s4 + o3 + o2 + o1
        p5_1 = s5 + o4 + o3 + o2 + o1
        p1_2 = s11 + o21 + o31 + o41 + o51
        p2_2 = s21 + o31 + o41 + o51
        p3_2 = s31 + o41 + o51
        p4_2 = s41 + o51
        p5_2 = s51

        fuse = self.fuse(torch.cat([p1_1, p2_1, p3_1, p4_1, p5_1, p1_2, p2_2, p3_2, p4_2, p5_2], 1))
       # print("fuse", fuse)

        return p1_1, p2_1, p3_1, p4_1, p5_1, p1_2, p2_2, p3_2, p4_2, p5_2, fuse

    def _initialize_weights(self, logger=None):
        for name, param in self.state_dict().items():
            if self.pretrain and 'features' in name:
                continue
            # elif 'down' in name:
            #     param.zero_()
            elif 'upsample' in name:
                if logger:
                    logger.info('init upsamle layer %s ' % name)
                k = int(name.split('.')[0].split('_')[1])
                param.copy_(get_upsampling_weight(1, 1, k*2))
            elif 'fuse' in name:
                if logger:
                    logger.info('init params %s ' % name)
                if 'bias' in name:
                    param.zero_()
                else:
                    nn.init.constant(param, 0.080)
            else:
                if logger:
                    logger.info('init params %s ' % name)
                if 'bias' in name:
                    param.zero_()
                else:
                    param.normal_(0, 0.01)
        # print self.conv1_1_down.weight
def crop_caffe(location, variable, th, tw):
    h, w = variable.shape[2], variable.shape[3]
    x1 = int(location)
    y1 = int(location)
    return variable[:, :, y1: y1 + th, x1: x1 + tw]
def make_bilinear_weights(size, num_channels):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    # print(filt)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w
# if __name__ == '__main__':
# #    model = BDCN('./caffemodel2pytorch/vgg16.pth')
#     a=torch.rand((2,3,100,100))
#     a=torch.autograd.Variable(a)
#     for x in model(a):
#         print (x.data.shape)
#     # for name, param in model.state_dict().items():
#     #     print name, param
#     #copy from the website


#if __name__ == '__main__':
#    model = BDCN('./caffemodel2pytorch/vgg16.pth')
#    a=torch.rand((2,3,100,100))
#    a=torch.autograd.Variable(a)
#    for x in model(a):
#        print (x.data.shape)
    # for name, param in model.state_dict().items():
    #     print name, param
class rcf_model(nn.Module):
    def __init__(self):

            super(rcf_model, self).__init__()
            self.dropout = nn.Dropout(0.2)
            self.conv1_1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)

            self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)

            self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
            self.conv3_3 = nn.Conv2d(128, 128, 3, padding=1)

            self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
            self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
            self.conv4_3 = nn.Conv2d(256, 256, 3, padding=1)
            self.conv5_1 = nn.Conv2d(256, 256, kernel_size=3,
                                     stride=1, padding=2, dilation=2)
            self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3,
                                     stride=1, padding=2, dilation=2)
            self.conv5_3 = nn.Conv2d(256, 256, kernel_size=3,
                                     stride=1, padding=2, dilation=2)
            # self.conv5_1=nn.Conv2d(256,256,3,padding=1)
            # self.conv5_2=nn.Conv2d(256,256,3,padding=1)
            # self.conv5_3=nn.Conv2d(256,256,3,padding=1)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
            self.maxpool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)

            self.conv1_1_down = nn.Conv2d(32, 21, 1, padding=0)
            self.conv1_2_down = nn.Conv2d(32, 21, 1, padding=0)

            self.conv2_1_down = nn.Conv2d(64, 21, 1, padding=0)
            self.conv2_2_down = nn.Conv2d(64, 21, 1, padding=0)

            self.conv3_1_down = nn.Conv2d(128, 21, 1, padding=0)
            self.conv3_2_down = nn.Conv2d(128, 21, 1, padding=0)
            self.conv3_3_down = nn.Conv2d(128, 21, 1, padding=0)

            self.conv4_1_down = nn.Conv2d(256, 21, 1, padding=0)
            self.conv4_2_down = nn.Conv2d(256, 21, 1, padding=0)
            self.conv4_3_down = nn.Conv2d(256, 21, 1, padding=0)

            self.conv5_1_down = nn.Conv2d(256, 21, 1, padding=0)
            self.conv5_2_down = nn.Conv2d(256, 21, 1, padding=0)
            self.conv5_3_down = nn.Conv2d(256, 21, 1, padding=0)

            self.dsn1 = nn.Conv2d(21, 1, 1)
            self.dsn2 = nn.Conv2d(21, 1, 1)
            self.dsn3 = nn.Conv2d(21, 1, 1)
            self.dsn4 = nn.Conv2d(21, 1, 1)
            self.dsn5 = nn.Conv2d(21, 1, 1)
            self.fuse = nn.Conv2d(5, 1, 1)
            self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
                h, w = x.shape[2], x.shape[3]
                conv1_1 = self.dropout(self.relu(self.conv1_1(x)))
                conv1_2 = self.dropout(self.relu(self.conv1_2(conv1_1)))
                pool1 = self.maxpool(conv1_2)

                conv2_1 = self.dropout(self.relu(self.conv2_1(pool1)))
                conv2_2 = self.dropout(self.relu(self.conv2_2(conv2_1)))
                pool2 = self.maxpool(conv2_2)

                conv3_1 = self.dropout(self.relu(self.conv3_1(pool2)))
                conv3_2 = self.dropout(self.relu(self.conv3_2(conv3_1)))
                conv3_3 = self.dropout(self.relu(self.conv3_3(conv3_2)))
                pool3 = self.maxpool(conv3_3)

                conv4_1 = self.dropout(self.relu(self.conv4_1(pool3)))
                conv4_2 = self.dropout(self.relu(self.conv4_2(conv4_1)))
                conv4_3 = self.dropout(self.relu(self.conv4_3(conv4_2)))
                pool4 = self.maxpool4(conv4_3)

                conv5_1 = self.dropout(self.relu(self.conv5_1(pool4)))
                conv5_2 = self.dropout(self.relu(self.conv5_2(conv5_1)))
                conv5_3 = self.dropout(self.relu(self.conv5_3(conv5_2)))

                conv1_1_down = self.conv1_1_down(conv1_1)
                conv1_2_down = self.conv1_2_down(conv1_2)
                conv2_1_down = self.conv2_1_down(conv2_1)
                conv2_2_down = self.conv2_2_down(conv2_2)
                conv3_1_down = self.conv3_1_down(conv3_1)
                conv3_2_down = self.conv3_2_down(conv3_2)
                conv3_3_down = self.conv3_3_down(conv3_3)
                conv4_1_down = self.conv4_1_down(conv4_1)
                conv4_2_down = self.conv4_2_down(conv4_2)
                conv4_3_down = self.conv4_3_down(conv4_3)
                conv5_1_down = self.conv5_1_down(conv5_1)
                conv5_2_down = self.conv5_2_down(conv5_2)
                conv5_3_down = self.conv5_3_down(conv5_3)

                sumconv1 = self.dsn1(conv1_1_down + conv1_2_down)
                sumconv2 = self.dsn2(conv2_2_down + conv2_1_down)
                sumconv3 = self.dsn3(conv3_1_down + conv3_2_down + conv3_3_down)
                sumconv4 = self.dsn4(conv4_1_down + conv4_2_down + conv4_3_down)
                sumconv5 = self.dsn5(conv5_1_down + conv5_2_down + conv5_3_down)
                weight_deconv2 = make_bilinear_weights(4, 1).to(device)
                weight_deconv3 = make_bilinear_weights(8, 1).to(device)
                weight_deconv4 = make_bilinear_weights(16, 1).to(device)
                weight_deconv5 = make_bilinear_weights(32, 1).to(device)
                upsample2 = torch.nn.functional.conv_transpose2d(sumconv2, weight_deconv2, stride=2)
                upsample3 = torch.nn.functional.conv_transpose2d(sumconv3, weight_deconv3, stride=4)
                upsample4 = torch.nn.functional.conv_transpose2d(sumconv4, weight_deconv4, stride=8)
                upsample5 = torch.nn.functional.conv_transpose2d(sumconv5, weight_deconv5, stride=8)

                so1 = crop_caffe(0, sumconv1, h, w)
                so2 = crop_caffe(1, upsample2, h, w)
                so3 = crop_caffe(2, upsample3, h, w)
                so4 = crop_caffe(4, upsample4, h, w)
                so5 = crop_caffe(8, upsample5, h, w)

               #  d1 = F.upsample_bilinear(so1,size=(h,w))
               #  #d1=sumconv1
               #  d2 = F.upsample_bilinear(so2, size=(h, w))
               #  d3 = F.upsample_bilinear(so3,size=(h,w))
               #
               # # d3 = F.upsample_bilinear(self.dsn3(sumconv3), size=(h, w))
               #  d4 = F.upsample_bilinear(so4, size=(h, w))
               #  d5 = F.upsample_bilinear(so5, size=(h, w))

                fuse = self.fuse(torch.cat((so1, so2, so3, so4, so5), 1))

                # fuse1=torch.cat(d1,d2,d3,d4,d5)
                d1 = torch.sigmoid(so1)
                d2 = torch.sigmoid(so2)
                d3 = torch.sigmoid(so3)
                d4 = torch.sigmoid(so4)
                d5=torch.sigmoid(so5)
                # fuse = self.fuse(torch.cat((d1, d2, d3, d4, d5), 1))
                # fuse=nn.Conv2d(32,1,1)

                fuse_result = torch.sigmoid(fuse)

                # results = [d1, d2, d3, d4, d5, fuse]
                # results = [torch.sigmoid(r) for r in results]
                return d1, d2, d3, d4, d5, fuse_result

def crop_caffe(location, variable, th, tw):
                     h, w = variable.shape[2], variable.shape[3]
                     x1 = int(location)
                     y1 = int(location)
                     return variable[:, :, y1: y1 + th, x1: x1 + tw]

def make_bilinear_weights(size, num_channels):
                    factor = (size + 1) // 2
                    if size % 2 == 1:
                        center = factor - 1
                    else:
                        center = factor - 0.5
                    og = np.ogrid[:size, :size]
                    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
                    # print(filt)
                    filt = torch.from_numpy(filt)
                    w = torch.zeros(num_channels, num_channels, size, size)
                    w.requires_grad = False
                    for i in range(num_channels):
                        for j in range(num_channels):
                            if i == j:
                                w[i, j] = filt
                    return w


class hed_model(nn.Module):
    def __init__(self):
        super(hed_model, self).__init__()
        self.conv1 = nn.Sequential(
            # conv1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),

        )
        self.conv2 = nn.Sequential(
            # conv2
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),

        )
        self.conv3 = nn.Sequential(
            # conv3
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),

        )
        self.conv4 = nn.Sequential(
            # conv4
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),

        )
        self.conv5 = nn.Sequential(
            # conv5
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dsn1 = nn.Conv2d(32, 1, 1)
        self.dsn2 = nn.Conv2d(64, 1, 1)
        self.dsn3 = nn.Conv2d(128, 1, 1)
        self.dsn4 = nn.Conv2d(256, 1, 1)
        self.dsn5 = nn.Conv2d(256, 1, 1)
        self.fuse = nn.Conv2d(5, 1, 1)

    def forward(self, x):
        h = x.size(2)
        w = x.size(3)

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        so1_out = self.dsn1(conv1)
        so2_out = self.dsn2(conv2)
        so3_out = self.dsn3(conv3)
        so4_out = self.dsn4(conv4)
        so5_out = self.dsn5(conv5)
        weight_deconv2 = make_bilinear_weights(4, 1).to(device)
        weight_deconv3 = make_bilinear_weights(8, 1).to(device)
        weight_deconv4 = make_bilinear_weights(16, 1).to(device)
        weight_deconv5 = make_bilinear_weights(32, 1).to(device)
        upsample2 = torch.nn.functional.conv_transpose2d(so2_out, weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(so3_out, weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(so4_out, weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(so5_out, weight_deconv5, stride=8)
        d1 = crop_caffe(0, so1_out, h, w)
        d2 = crop_caffe(1, upsample2, h, w)
        d3 = crop_caffe(2, upsample3, h, w)
        d4 = crop_caffe(4, upsample4, h, w)
        d5 = crop_caffe(8, upsample5, h, w)
        ## side output
        d1 = self.dsn1(conv1)
        d2 = F.upsample_bilinear(self.dsn2(conv2), size=(h, w))
        d3 = F.upsample_bilinear(self.dsn3(conv3), size=(h, w))
        d4 = F.upsample_bilinear(self.dsn4(conv4), size=(h, w))
        d5 = F.upsample_bilinear(self.dsn5(conv5), size=(h, w))

        # dsn fusion output
        fuse = self.fuse(torch.cat((d1, d2, d3, d4, d5), 1))
        #results=[d1,d2,d3,d4,d5,fuse]
        d1 = F.sigmoid(d1)
        d2 = F.sigmoid(d2)
        d3 = F.sigmoid(d3)
        d4 = F.sigmoid(d4)
        d5 = F.sigmoid(d5)
        fuse = F.sigmoid(fuse)
        #results=(d1+d2+d3+d4+d5+fuse)/6
       # results = [d1, d2, d3, d4, d5, fuse]
        #fusecat = torch.cat((d1, d2, d3, d4, d5), dim=1)
        #fuse_1 = self.fuse(fusecat)
        #results = [d1, d2,d3, d4, d5, fuse]
       # results = [torch.sigmoid(r) for r in results]
        #fuse = F.sigmoid(fuse)
        #results=[d1,d2,d3,d4,d5,fuse]
        return d1,d2,d3,d4,d5,fuse

    def crop_caffe(location, variable, th, tw):
        h, w = variable.shape[2], variable.shape[3]
        x1 = int(location)
        y1 = int(location)
        return variable[:, :, y1: y1 + th, x1: x1 + tw]

    def make_bilinear_weights(size, num_channels):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        # print(filt)
        filt = torch.from_numpy(filt)
        w = torch.zeros(num_channels, num_channels, size, size)
        w.requires_grad = False
        for i in range(num_channels):
            for j in range(num_channels):
                if i == j:
                    w[i, j] = filt
        return w
