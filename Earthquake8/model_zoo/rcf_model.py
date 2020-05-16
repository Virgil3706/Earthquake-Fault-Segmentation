import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import torchsummary as summary
import torchvision.models as models
import torch.autograd.variable as Variable
import numpy as np
import scipy.io as sio
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.conv import _single, _pair, _triple
import torch.nn.functional as F
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class rcf_model(nn.Module):
    def __init__(self):
        super(rcf_model, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.conv1_1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv1_2=nn.Conv2d(32,32,3,padding=1)

        self.conv2_1=nn.Conv2d(32,64,3,padding=1)
        self.conv2_2=nn.Conv2d(64,64,3,padding=1)

        self.conv3_1=nn.Conv2d(64,128,3,padding=1)
        self.conv3_2=nn.Conv2d(128,128,3,padding=1)
        self.conv3_3=nn.Conv2d(128,128,3,padding=1)

        self.conv4_1=nn.Conv2d(128,256,3,padding=1)
        self.conv4_2=nn.Conv2d(256,256,3,padding=1)
        self.conv4_3=nn.Conv2d(256,256,3,padding=1)
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
        self.maxpool=nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.maxpool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)

        self.conv1_1_down=nn.Conv2d(32,21,1,padding=0)
        self.conv1_2_down=nn.Conv2d(32,21,1,padding=0)

        self.conv2_1_down=nn.Conv2d(64,21,1,padding=0)
        self.conv2_2_down=nn.Conv2d(64,21,1,padding=0)

        self.conv3_1_down=nn.Conv2d(128,21,1,padding=0)
        self.conv3_2_down=nn.Conv2d(128,21,1,padding=0)
        self.conv3_3_down=nn.Conv2d(128,21,1,padding=0)

        self.conv4_1_down=nn.Conv2d(256,21,1,padding=0)
        self.conv4_2_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv4_3_down = nn.Conv2d(256, 21, 1, padding=0)

        self.conv5_1_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv5_2_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv5_3_down = nn.Conv2d(256, 21, 1, padding=0)

        self.dsn1=nn.Conv2d(21,1,1)
        self.dsn2=nn.Conv2d(21,1,1)
        self.dsn3=nn.Conv2d(21,1,1)
        self.dsn4=nn.Conv2d(21,1,1)
        self.dsn5=nn.Conv2d(21,1,1)
        self.fuse = nn.Conv2d(5, 1, 1)
        self.dropout=nn.Dropout2d(p=0.2)
        def forward(self, x):
            h, w = x.shape[2], x.shape[3]
            conv1_1 = self.dropout(self.relu(self.conv1_1(x)))
            conv1_2 = self.dropout(self.relu(self.conv1_2(conv1_1)))
            pool1=self.maxpool(conv1_2)

            conv2_1= self.dropout(self.relu(self.conv2_1(pool1)))
            conv2_2= self.dropout(self.relu(self.conv2_2(conv2_1)))
            pool2=self.maxpool(conv2_2)

            conv3_1= self.dropout(self.relu(self.conv3_1(pool2)))
            conv3_2= self.dropout(self.relu(self.conv3_2(conv3_1)))
            conv3_3= self.dropout(self.relu(self.conv3_3(conv3_2)))
            pool3=self.maxpool(conv3_3)

            conv4_1= self.dropout(self.relu(self.conv4_1(pool3)))
            conv4_2=self.dropout(self.relu(self.conv4_2(conv4_1)))
            conv4_3= self.dropout(self.relu(self.conv4_3(conv4_2)))
            pool4=self.maxpool4(conv4_3)

            conv5_1= self.dropout(self.relu(self.conv5_1(pool4)))
            conv5_2= self.dropout(self.relu(self.conv5_2(conv5_1)))
            conv5_3= self.dropout(self.relu(self.conv5_3(conv5_2)))

            conv1_1_down=self.conv1_1_down(conv1_1)
            conv1_2_down=self.conv1_2_down(conv1_2)
            conv2_1_down=self.conv2_1_down(conv2_1)
            conv2_2_down=self.conv2_2_down(conv2_2)
            conv3_1_down=self.conv3_1_down(conv3_1)
            conv3_2_down=self.conv3_2_down(conv3_2)
            conv3_3_down=self.conv3_3_down(conv3_3)
            conv4_1_down=self.conv4_1_down(conv4_1)
            conv4_2_down=self.conv4_2_down(conv4_2)
            conv4_3_down=self.conv4_3_down(conv4_3)
            conv5_1_down=self.conv5_1_down(conv5_1)
            conv5_2_down=self.conv5_2_down(conv5_2)
            conv5_3_down=self.conv5_3_down(conv5_3)

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

            # d1 = F.upsample_bilinear(so1, size=(h, w))
            # # d1=sumconv1
            # d2 = F.upsample_bilinear(so2, size=(h, w))
            # d3 = F.upsample_bilinear(so3, size=(h, w))
            #
            # # d3 = F.upsample_bilinear(self.dsn3(sumconv3), size=(h, w))
            # d4 = F.upsample_bilinear(so4, size=(h, w))
            # d5 = F.upsample_bilinear(so5, size=(h, w))

            fuse = self.fuse(torch.cat((so1, so2, so3, so4, so5), 1))

            # fuse1=torch.cat(d1,d2,d3,d4,d5)
            d1 = torch.sigmoid(so1)
            d2 = torch.sigmoid(so2)
            d3 = torch.sigmoid(so3)
            d4 = torch.sigmoid(so4)
            d5 = torch.sigmoid(so5)
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
#model = rcf_model()
#model.to(device);
#summary(model, (1, 100, 100))





