import argparse
import collections
import torch
from tqdm import tqdm
import data_loader.dataloader3 as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from functions import *
import os
import matplotlib;
#  matplotlib.use('TkAgg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Only GPU 1 is visible to this code
import data_loader.dataloader3 as module_data
from datetime import datetime



torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1)
random.seed(1)
best_iou_threshold=0.5


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 96, 96)
    return out


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances

    data_loader = config.init_obj('data_loader2', module_data)
    data_loader = data_loader.test_loader
    seis_path = config['data_loader2']['args']['seismic_path']
    seis = np.load(seis_path)
    run_id = datetime.now().strftime(r'%m%d_%H%M%S')
    save_path = 'saved/picture/'+config['arch']['type']+'/'+run_id
    val_start = config['data_loader2']['args']['val_start']
    number_of_pictures = config['data_loader2']['args']['val_number_of_pictures']
    modelNo=config['trainer']['modelNo']
    dimension=config['data_loader2']['args']['dimension']

    # IL, Z, XL = seis.shape
    IL, Z, XL = seis.shape
    if dimension == 0:
        im_height = Z
        im_width = XL
    elif dimension == 1:
        im_height = IL
        im_width = XL
    elif dimension == 2:
        im_height = IL
        im_width = Z

    splitsize = 96
    stepsize = 48  # overlap half
    overlapsize = splitsize - stepsize
    horizontal_splits_number = int(np.ceil((im_width) / stepsize))
    width_after_pad = stepsize * horizontal_splits_number + 2 * overlapsize
    left_pad = int((width_after_pad - im_width) / 2)
    right_pad = width_after_pad - im_width - left_pad
    vertical_splits_number = int(np.ceil((im_height) / stepsize))
    height_after_pad = stepsize * vertical_splits_number + 2 * overlapsize
    top_pad = int((height_after_pad - im_height) / 2)
    bottom_pad = height_after_pad - im_height - top_pad
    horizontal_splits_number = horizontal_splits_number + 1
    # print(horizontal_splits_number)
    vertical_splits_number = vertical_splits_number + 1
    # print(vertical_splits_number)
    halfoverlapsize = int(overlapsize / 2)



    # build model architecture
    model = config.init_obj('arch', module_arch)
    # logger.info(model)
    # print(config['arch']['type'])
    # summary(model, (1, splitsize, splitsize))
    # print(config['arch']['type'])
    # print("start:",config['data_loader2']['args']['val_start'])
    # print("number_of_pictures:",config['data_loader2']['args']['val_number_of_pictures'])


    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    try:
      checkpoint = torch.load(config.resume,map_location=device)
      state_dict = checkpoint['state_dict']
      if config['n_gpu'] > 1:
          model = torch.nn.DataParallel(model)
      model.load_state_dict(state_dict)
    except:
      checkpoint = torch.load(config.resume, map_location=device)
      model.load_state_dict(checkpoint)


    # prepare model for testing
    model = model.to(device)
    model.eval()


    WINDOW_SPLINE_2D = window_2D(window_size=splitsize, power=2)
    os.makedirs(save_path, exist_ok=True)
    test_predictions = []
    imageNo = -1

    with torch.no_grad():
        for images in tqdm(data_loader):
            images = Variable(images.to(device))
            images=to_img(images)
            outputs = model(images)
            y_preds = outputs
            if modelNo == 2:
                y_preds = outputs[-2]
            elif modelNo == 3:
                y_preds = outputs[-1]
            #
            # predicted_mask = y_preds > best_iou_threshold
            # y_preds=predicted_mask
            #

            test_predictions.extend(y_preds.detach().to(device))
            #         print(test_predictions[0].dtype)
            #         print(len(test_predictions))
            if len(test_predictions) >= vertical_splits_number * horizontal_splits_number:
                imageNo = imageNo + 1
                tosave = torch.stack(test_predictions).detach().cpu().numpy()[
                         0:vertical_splits_number * horizontal_splits_number]
                #             print(tosave.shape)
                test_predictions = test_predictions[vertical_splits_number * horizontal_splits_number:]

                tosave = np.moveaxis(tosave, -3, -1)
                #             print(tosave.shape)
                tosave = np.array([patch * WINDOW_SPLINE_2D for patch in tosave])
                #             print(tosave.shape)
                #             break
                tosave = tosave.reshape((vertical_splits_number, horizontal_splits_number, splitsize, splitsize, 1))
                #             print(tosave.shape)

                recover_Y_test_pred = recover_Image2(tosave, (im_height, im_width, 1), left_pad, right_pad, top_pad,
                                                     bottom_pad, overlapsize)

                os.makedirs(save_path, exist_ok=True)
                np.save(os.path.join(save_path, "{}".format(imageNo+val_start)), np.squeeze(recover_Y_test_pred))


    # print("saved")

    for i in range(0,number_of_pictures,1):

        index = i+val_start
        a = np.load(os.path.join(save_path, str(index) + '.npy'))

        import cv2
        import cmapy

        # heatmap_img = cv2.applyColorMap((a * 255).astype(np.uint8), cmapy.cmap('jet_r'))
        plt.figure(figsize=(10, 8))
        plt.imshow(a)
        # plt.imshow(a)
        # plt.colorbar(shrink=0.5)
        plt.axis('off')
        # plt.show()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.margins(0,0)

        plt.savefig('{}/predicted_{}.png'.format(save_path,str(index)),bbox_inches='tight',pad_inches=0.0)
        # save path
        print('{}/predicted_{}.png'.format(save_path,str(index)))

        # label=visu(save_path,
        #      label_path=config['data_loader2']['args']['label_path'],
        #      index=index,dimension=dimension)
        seismic=visu1(save_path,
             seismic_path=config['data_loader2']['args']['seismic_path'],
             index=index,dimension=dimension)
        #label path
        # print(label)
        #seismic path
        print(seismic)
        # a=blendTwoImages(label,seismic,'{}/pre_fusion_{}.png'.format(save_path,str(index)))
        # b=blendTwoImages(a,'{}/predicted_{}.png'.format(save_path,str(index)),'{}/fusion_{}.png'.format(save_path,str(index)))
        #fusion path
        # print(b)
        # a=blendTwoImages(label,'{}/{}_{}.png'.format(save_path,run_id,str(index)),'saved/ab.png')
        # b=blendTwoImages(a,seismic)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default='', type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')



    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--vs', '--val_start'], type=int, target='data_loader2;args;val_start'),
        CustomArgs(['--vp', '--val_number_of_pictures'], type=int, target='data_loader2;args;val_number_of_pictures'),
        CustomArgs(['--sp', '--seismic_path'], type=str, target='data_loader2;args;seismic_path'),
        CustomArgs(['--lp', '--label_path'], type=str, target='data_loader2;args;label_path'),
        CustomArgs(['--dm', '--dimension'], type=int, target='data_loader2;args;dimension')
    ]

    config = ConfigParser.from_args(args,options)
    main(config)




    #
    # config = ConfigParser.from_args(args)
    # main(config)
