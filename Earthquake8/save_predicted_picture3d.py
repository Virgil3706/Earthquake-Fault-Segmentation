import torch
from functions import *
import os
import matplotlib; 
#matplotlib.use('TkAgg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Only GPU 1 is visible to this code
import data_loader.dataloader3 as module_data
from datetime import datetime
from model.model import unet3d
import cv2
import cmapy
from model.loss import bceloss
from model.metric import IoU3d,accuracy3d
import numpy as np

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
args ={"seismic_path":"data/FYP_data/seis_sub_350IL_500t_1200XL.npy",
            "label_path": "data/FYP_data/fault_sub_350IL_500t_1200XL.npy",
            "batch_size": 8,
            "test_number_of_pictures": 64,
            "test_start": 284,
            "dimension": 0
        }
        
def main():
    test_set = module_data.EarthquakeDataLoader3dtest(args['seismic_path'],args['label_path'],args['batch_size'],args['test_number_of_pictures'],args['test_start'],args['dimension'])
    test_set_loader = test_set.test_loader
    model = unet3d()

    model.load_state_dict(torch.load('saved/models/unet3d/0521_022910/model_best.pth')['state_dict'])
    cube = np.zeros((64,500,1200))
    blocks = []
    print('-'*30)

    cnt = 0
    for images in test_set_loader:
        print(cnt)
        cnt+=1
        images = Variable(images)
        outputs = model(images)
        outputs = torch.squeeze(outputs)
        blocks.extend(outputs.detach().cpu())
        
        # blocks.append(tosave)
    blocks = torch.stack(blocks).numpy()
    blocks = np.squeeze(blocks)
    print(blocks.shape)


    r = 32
    step = 48
    x = 0
    for i in range(r, cube.shape[0] - r + 1, step):
        for j in range(r, cube.shape[1] - r + 1, step):
            for k in range(r, cube.shape[2] - r + 1, step):
                cube[i - r:i + r, j - r:j + r, k - r:k + r]+=blocks[x]
                x=x+1
        # print(res.shape)
    cube_overlap = np.zeros((64,500,1200))

    r = 32
    step = 48
    x = 0
    for i in range(r, cube.shape[0] - r + 1, step):
        for j in range(r, cube.shape[1] - r + 1, step):
            for k in range(r, cube.shape[2] - r + 1, step):
                cube_overlap[i - r:i + r, j - r:j + r, k - r:k + r]+=np.ones((64,64,64))
                x=x+1


    cube=np.true_divide(cube,cube_overlap)


    #run_id = datetime.now().strftime(r'%m%d_%H%M%S')
    save_path ='saved/picture/unet3d/new'

    seismic_path = 'data/FYP_data/seis_sub_350IL_500t_1200XL.npy'
    label_path = 'data/FYP_data/fault_sub_350IL_500t_1200XL.npy'

    seismic = np.load(seismic_path, allow_pickle=True)
    fault = np.load(label_path)

    
    # from logger import TensorboardWriter
    # cfg_trainer = trainer
    # writer = TensorboardWriter(cfg_trainer['tensorboard'])

    total_loss = 0.0
    metrics = torch.zeros(2)
    total_metrics = torch.zeros(2)

    print('-'*30)
    cnt1 = 0
    for index in range(0,64):
        a = cube[index,:,:]
        output = a
        target = fault[index+286,:,:]
        output = np.nan_to_num(output)
        output = output.astype(np.float32)
        target = target.astype(np.float32)

        # writer.set_step(1, 'test')
        # writer.add_pr_curve('precision_recall_curve', 1, target, output)
        # from utils import inf_loop, MetricTracker. 

        # loss = torch.zeros(1).to(device)
        # loss = bceloss(output, target)
        # print(loss)

        predicted_mask = output > 0.5

        # total_loss += loss.item()


        target1 = target > 0.5
        predicted_mask = torch.tensor(predicted_mask)
        target1 = torch.tensor(target1)

        metrics[0] = IoU3d(predicted_mask, target1)
        total_metrics[0] += metrics[0]
        print("-"*30)
        print("IoU: ", metrics[0].item()*100,"%")
        metrics[1] = accuracy3d(predicted_mask, target)
        total_metrics[1] += metrics[1]
        print("accuracy: ", metrics[1].item()*100,"%")
        cnt1+=1

        

        heatmap_img = cv2.applyColorMap((a * 255).astype(np.uint8), cmapy.cmap('jet_r'))
        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap_img)
        plt.axis('off')
        plt.savefig('{}/predicted_{}.png'.format(save_path,str(index)))
        plt.imshow(seismic[index+286,:,:])
        plt.axis('off')
        plt.savefig('{}/seismic_{}.png'.format(save_path,str(index)))
        plt.imshow(fault[index+286,:,:])
        plt.axis('off')
        plt.savefig('{}/fault_{}.png'.format(save_path,str(index)))
        # save path
        print('{}/predicted_{}.png'.format(save_path,str(index)))



        a=blendTwoImages('{}/fault_{}.png'.format(save_path,str(index)),'{}/seismic_{}.png'.format(save_path,str(index)),'{}/pre_fusion_{}.png'.format(save_path,str(index)))
        b=blendTwoImages(a,'{}/predicted_{}.png'.format(save_path,str(index)),'{}/fusion_{}.png'.format(save_path,str(index)))
        #fusion path
        # print(b)
        # a=blendTwoImages(label,'{}/{}_{}.png'.format(save_path,run_id,str(index)),'saved/ab.png')
        # b=blendTwoImages(a,seismic)
    
    print("-"*30)
    print("total IoU: ", total_metrics[0].item()/cnt1*100,"%")
    print("total accuracy: ", total_metrics[1].item()/cnt1*100,"%")

if __name__ == '__main__':
    main()
