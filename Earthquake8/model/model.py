import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
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

def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation, )


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        activation, )


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim), )


class unet3d(nn.Module):
    def __init__(self):
        super(unet3d, self).__init__()

        num_filters = 4
        activation = nn.LeakyReLU(0.2, inplace=True)

        # Down sampling
        self.down_1 = conv_block_2_3d(1, num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(num_filters, num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(num_filters * 2, num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(num_filters * 4, num_filters * 8, activation)
        self.pool_4 = max_pooling_3d()
        self.down_5 = conv_block_2_3d(num_filters * 8, num_filters * 16, activation)
        self.pool_5 = max_pooling_3d()

        # Bridge
        self.bridge = conv_block_2_3d(num_filters * 16, num_filters * 32, activation)

        # Up sampling
        self.trans_1 = conv_trans_block_3d(num_filters * 32, num_filters * 32, activation)
        self.up_1 = conv_block_2_3d(num_filters * 48, num_filters * 16, activation)
        self.trans_2 = conv_trans_block_3d(num_filters * 16, num_filters * 16, activation)
        self.up_2 = conv_block_2_3d(num_filters * 24, num_filters * 8, activation)
        self.trans_3 = conv_trans_block_3d(num_filters * 8, num_filters * 8, activation)
        self.up_3 = conv_block_2_3d(num_filters * 12, num_filters * 4, activation)
        self.trans_4 = conv_trans_block_3d(num_filters * 4, num_filters * 4, activation)
        self.up_4 = conv_block_2_3d(num_filters * 6, num_filters * 2, activation)
        self.trans_5 = conv_trans_block_3d(num_filters * 2, num_filters * 2, activation)
        self.up_5 = conv_block_2_3d(num_filters * 3, num_filters * 1, activation)

        # Output
        self.out = conv_block_3d(num_filters, 1, nn.Sigmoid())

    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x)  # -> [1, 4, 128, 128, 128]
        pool_1 = self.pool_1(down_1)  # -> [1, 4, 64, 64, 64]

        down_2 = self.down_2(pool_1)  # -> [1, 8, 64, 64, 64]
        pool_2 = self.pool_2(down_2)  # -> [1, 8, 32, 32, 32]

        down_3 = self.down_3(pool_2)  # -> [1, 16, 32, 32, 32]
        pool_3 = self.pool_3(down_3)  # -> [1, 16, 16, 16, 16]

        down_4 = self.down_4(pool_3)  # -> [1, 32, 16, 16, 16]
        pool_4 = self.pool_4(down_4)  # -> [1, 32, 8, 8, 8]

        down_5 = self.down_5(pool_4)  # -> [1, 64, 8, 8, 8]
        pool_5 = self.pool_5(down_5)  # -> [1, 64, 4, 4, 4]

        # Bridge
        bridge = self.bridge(pool_5)  # -> [1, 128, 4, 4, 4]

        # Up sampling
        trans_1 = self.trans_1(bridge)  # -> [1, 128, 8, 8, 8]
        concat_1 = torch.cat([trans_1, down_5], dim=1)  # -> [1, 192, 8, 8, 8]
        up_1 = self.up_1(concat_1)  # -> [1, 64, 8, 8, 8]

        trans_2 = self.trans_2(up_1)  # -> [1, 64, 16, 16, 16]
        concat_2 = torch.cat([trans_2, down_4], dim=1)  # -> [1, 96, 16, 16, 16]
        up_2 = self.up_2(concat_2)  # -> [1, 32, 16, 16, 16]

        trans_3 = self.trans_3(up_2)  # -> [1, 32, 32, 32, 32]
        concat_3 = torch.cat([trans_3, down_3], dim=1)  # -> [1, 48, 32, 32, 32]
        up_3 = self.up_3(concat_3)  # -> [1, 16, 32, 32, 32]

        trans_4 = self.trans_4(up_3)  # -> [1, 16, 64, 64, 64]
        concat_4 = torch.cat([trans_4, down_2], dim=1)  # -> [1, 24, 64, 64, 64]
        up_4 = self.up_4(concat_4)  # -> [1, 8, 64, 64, 64]

        trans_5 = self.trans_5(up_4)  # -> [1, 8, 128, 128, 128]
        concat_5 = torch.cat([trans_5, down_1], dim=1)  # -> [1, 12, 128, 128, 128]
        up_5 = self.up_5(concat_5)  # -> [1, 4, 128, 128, 128]

        # Output
        out = self.out(up_5)  # -> [1, 3, 128, 128, 128]
        return out


