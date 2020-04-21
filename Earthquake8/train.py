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
# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
splitsize = 96


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    data_loader, valid_data_loader = data_loader.train_loader, data_loader.val_loader

    # build model architecture, then print to console

    model = config.init_obj('arch', module_arch)
    logger.info(model)
    # print(config['arch']['type'])
    # summary(model, (1, splitsize, splitsize))

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    # print(criterion)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    #optimizer = config.init_obj('optimizer2', module_opti, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler2', torch.optim.lr_scheduler, optimizer)
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='EarthquakeProject')
    args.add_argument('-c', '--config', default='config0.json', type=str,
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
        CustomArgs(['--tp', '--train_number_of_pictures'], type=int, target='data_loader;args;train_number_of_pictures'),
        CustomArgs(['--ep', '--epochs'], type=int, target='trainer;epochs'),
        CustomArgs(['--vs', '--val_start'], type=int, target='data_loader;args;val_start'),
        CustomArgs(['--ts2', '--val_number_of_pictures'], type=int, target='data_loader;args;val_number_of_pictures'),


    ]
    config = ConfigParser.from_args(args, options)
    main(config)
