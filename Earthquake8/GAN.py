import argparse
import collections
import torch
import numpy as np
import data_loader.dataloader3 as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from functions import *
from parse_config import ConfigParser
from trainer import Trainer
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
splitsize = 96

from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image


def main(config):
    logger = config.get_logger('train')
    # setup data_loader instances

    # build model architecture, then print to console

    model = config.init_obj('arch', module_arch)
    print('Generator:')
    logger.info(model)
    run_id = datetime.now().strftime(r'%m%d_%H%M%S')
    save_path = 'saved/' + 'GAN' + '/' + run_id
    # print(config['arch']['type'])
    # summary(model, (1, splitsize, splitsize))

    # get function handles of loss and metrics
    # criterion = getattr(module_loss, config['loss'])
    # # print(criterion)
    # metrics = [getattr(module_metric, met) for met in config['metrics']]
    #
    # # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    #
    # #optimizer = config.init_obj('optimizer2', module_opti, trainable_params)
    #
    # lr_scheduler = config.init_obj('lr_scheduler2', torch.optim.lr_scheduler, optimizer)
    # trainer = Trainer(model, criterion, metrics, optimizer,
    #                   config=config,
    #                   data_loader=data_loader,
    #                   valid_data_loader=valid_data_loader,
    #                   lr_scheduler=lr_scheduler)

    import torch
    import torch.nn as nn
    from torch.autograd import Variable
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision import datasets
    from torchvision.utils import save_image
    import os

    # import os
    # save_dir =  .{}/dc_img'.format(save_path))
    # if os.path.exists(save_dir) is False:
    #     os.makedirs(save_dir)
    if not os.path.exists('{}/dc_img'.format(save_path)):
        os.makedirs('{}/dc_img'.format(save_path))

    batch_size = 128
    num_epoch = 700

    z_dimension = 100  # noise dimension

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    data_loader, valid_data_loader = data_loader.train_loader, data_loader.val_loader
    data_loader = valid_data_loader
    h = 96;
    w = 96;

    # for i, (img, label) in enumerate(data_loader):
    #     h=img.shape[2]
    #     w=img.shape[3]
    # print("h",h)
    # print("w",w)

    def to_img(x):
        out = 0.5 * (x + 1)
        out = out.clamp(0, 1)
        out = out.view(-1, 1, h, w)
        return out

    class discriminator2(nn.Module):

        def __init__(self):
            super(discriminator, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, 5, padding=2),  # batch, 32, 28, 28

                nn.LeakyReLU(0.2, True),
                nn.AvgPool2d(2, stride=2),  # batch, 32, 14, 14
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, 5, padding=2),  # batch, 64, 14, 14

                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, True),
                nn.AvgPool2d(2, stride=2)  # batch, 64, 7, 7
            )

            self.fc = nn.Sequential(

                nn.Linear(4 * h * w, 1024),
                nn.LeakyReLU(0.2, True),
                nn.Linear(1024, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            '''
            x: batch, width, height, channel=1
            '''
            b, c, w, h = x.shape
            x = self.conv1(x)

            # print(x.shape)
            x = self.conv2(x)
            # print(x.shape)
            x = x.view(x.size(0), -1)
            # print(x.shape)
            x = self.fc(x)
            # print(x.shape)
            return x

    class discriminator(nn.Module):

        def __init__(self):
            super(discriminator, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 64, 4, stride=2, padding=1),  # batch, 64, 48, 48
                nn.ReLU(0.2)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(64, 128, 4, stride=2, padding=1),  # batch, 128, 24, 24
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, True)

            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(128, 256, 4, stride=2, padding=1),  # batch, 256, 12, 12
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, True)

            )
            self.conv4 = nn.Sequential(
                nn.Conv2d(256, 512, 4, stride=2, padding=1),  # batch, 512, 6, 6
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, True)

            )

            self.fc = nn.Sequential(
                nn.Linear(2 * h * w, 256),
                nn.LeakyReLU(0.2, True),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            '''
            x: batch, width, height, channel=1
            '''
            b, c, w, h = x.shape
            x = self.conv1(x)

            # print(x.shape)
            x = self.conv2(x)
            # print(x.shape)
            x = self.conv3(x)
            x = self.conv4(x)
            x = x.view(x.size(0), -1)
            # print(x.shape)
            x = self.fc(x)
            # print(x.shape)
            return x

    # class generator(nn.Module):
    #     def __init__(self, input_size, num_feature):
    #         super(generator, self).__init__()
    #         self.fc = nn.Linear(input_size, num_feature)  # batch, 3136=1x56x56
    #         self.br = nn.Sequential(
    #             nn.BatchNorm2d(1),
    #             nn.ReLU(True)
    #         )
    #         self.downsample1 = nn.Sequential(
    #             nn.Conv2d(1, 50, 3, stride=1, padding=1),  # batch, 50, 56, 56
    #             nn.BatchNorm2d(50),
    #             nn.ReLU(True)
    #         )
    #         self.downsample2 = nn.Sequential(
    #             nn.Conv2d(50, 25, 3, stride=1, padding=1),  # batch, 25, 56, 56
    #             nn.BatchNorm2d(25),
    #             nn.ReLU(True)
    #         )
    #         self.downsample3 = nn.Sequential(
    #             nn.Conv2d(25, 1, 2, stride=2),  # batch, 1, 28, 28
    #             nn.Tanh()
    #         )
    #
    #     def forward(self, x):
    #         x = self.fc(x)
    #         # print(x.shape)
    #         x = x.view(x.size(0), 1, 96 * 2, 96 * 2)
    #         # print(x.shape)
    #         x = self.br(x)
    #         # print(x.shape)
    #         x = self.downsample1(x)
    #         # print(x.shape)
    #         x = self.downsample2(x)
    #         # print(x.shape)
    #         x = self.downsample3(x)
    #         # print(x.shape)
    #         return x

    D = discriminator().to(device)  # discriminator model
    # D=model
    # G = generator(z_dimension, 36864).to(device)  # generator model
    G = model.to(device)
    criterion = nn.BCELoss()  # binary cross entropy

    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

    # train
    for epoch in range(num_epoch):
        for i, (img, label) in enumerate(data_loader):
            num_img = img.size(0)
            # print("num_img,", num_img)
            # print(img.shape)
            # =================train discriminator
            real_img = Variable(label).to(device)
            lowhd_img = Variable(img).to(device)
            real_label = Variable(torch.ones(num_img)).to(device)
            fake_label = Variable(torch.zeros(num_img)).to(device)

            # compute loss of real_img
            real_out = D(real_img)
            # print("done!")
            d_loss_real = criterion(real_out, real_label)
            real_scores = real_out  # closer to 1 means better
            # print("d_loss_real", d_loss_real)
            # compute loss of fake_img
            z = Variable(torch.randn(num_img, z_dimension)).to(device)
            # fake_img = G(z)
            fake_img = G(lowhd_img)
            # print("fake_img.shape", fake_img.shape)
            fake_out = D(fake_img)
            d_loss_fake = criterion(fake_out, fake_label)
            fake_scores = fake_out  # closer to 0 means better

            # bp and optimize
            d_loss = d_loss_real + d_loss_fake

            # print(d_loss)

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # ===============train generator
            # compute loss of fake_img
            z = Variable(torch.randn(num_img, z_dimension)).to(device)
            # fake_img = G(z)
            fake_img = G(lowhd_img)
            output = D(fake_img)
            g_loss = criterion(output, real_label)

            # bp and optimize
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                      'D real: {:.6f}, D fake: {:.6f}'
                      .format(epoch, num_epoch, d_loss.item(), g_loss.item(),
                              real_scores.data.mean(), fake_scores.data.mean()))
        if epoch == 0:
            print('epoch = 0')
            real_images = to_img(real_img.cpu().data)

            save_image(real_images, '{}/dc_img/real_images.png'.format(save_path))
            lowhd_imgs = to_img(lowhd_img.cpu().data)
            save_image(lowhd_imgs, '{}/dc_img/lowhd_img.png'.format(save_path))
        if (epoch + 1) % 10 == 0:
            fake_images = to_img(fake_img.cpu().data)
            save_image(fake_images, '{}/dc_img/fake_images-{}.png'.format(save_path, epoch + 1))
            # torch.save(G.state_dict(), '/generator{}.pth'.format(epoch + 1))
        if (epoch + 1) % 50 == 0:
            torch.save(G.state_dict(), '{}/dc_img/generator{}.pth'.format(save_path, epoch + 1))

    torch.save(G.state_dict(), '{}/generator_final_one.pth'.format(save_path))
    torch.save(D.state_dict(), '{}/discriminator.pth'.format(save_path))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='EarthquakeProject')
    args.add_argument('-c', '--config', default='config_gan.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--ts', '--train_start'], type=int, target='data_loader;args;train_start'),
        CustomArgs(['--tp', '--train_number_of_pictures'], type=int,
                   target='data_loader;args;train_number_of_pictures'),
        CustomArgs(['--ep', '--epochs'], type=int, target='trainer;epochs'),
        CustomArgs(['--vs', '--val_start'], type=int, target='data_loader;args;val_start'),
        CustomArgs(['--ts2', '--val_number_of_pictures'], type=int, target='data_loader;args;val_number_of_pictures'),

    ]
    config = ConfigParser.from_args(args, options)
    main(config)


