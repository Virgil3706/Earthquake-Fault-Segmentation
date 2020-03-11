
# coding: utf-8

# In[1]:


from functions import *

# in order to get reproducable results
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1)
random.seed(1)

import cmapy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Only GPU 1 is visible to this code


# In[2]:


seis_path = "/Users/shanyuhai/PycharmProjects/Earthquake/data/FYP_data/seis_sub_350IL_500t_1200XL.npy"
seis = np.load(seis_path)
IL, Z, XL = seis.shape
print(IL, Z, XL)
print(seis.shape)

im_height = Z
im_width = XL
splitsize = 96
stepsize = 48 #overlap half
overlapsize = splitsize-stepsize

horizontal_splits_number = int(np.ceil((im_width)/stepsize))
print(horizontal_splits_number)
width_after_pad = stepsize*horizontal_splits_number+2*overlapsize
print(width_after_pad)
left_pad = int((width_after_pad-im_width)/2)
right_pad = width_after_pad-im_width-left_pad
print(left_pad,right_pad)

vertical_splits_number = int(np.ceil((im_height)/stepsize))
print(vertical_splits_number)
height_after_pad = stepsize*vertical_splits_number+2*overlapsize
print(height_after_pad)
top_pad = int((height_after_pad-im_height)/2)
bottom_pad = height_after_pad-im_height-top_pad
print(top_pad,bottom_pad)

horizontal_splits_number = horizontal_splits_number+1
print(horizontal_splits_number)
vertical_splits_number = vertical_splits_number+1
print(vertical_splits_number)

halfoverlapsize = int(overlapsize/2)
print(halfoverlapsize)


# In[ ]:


print(len(seis))


# In[ ]:

# In[ ]:


t_start = time.time()
X_list = []
for i in range(78,88,1):
    img = seis[i]
    X_list.extend(split_Image(img,True,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number))

print(len(X_list))
print(X_list[0].shape)
print("read images in {} sec".format(time.time()-t_start))
X = np.asarray(X_list)
print(X.shape)
print("read images in {} sec".format(time.time()-t_start))



X = X.astype(np.float32)
X = np.expand_dims(X,1)
print(X.shape)


# idea from: https://www.kaggle.com/erikistre/pytorch-basic-u-net
class faultsDataset(torch.utils.data.Dataset):

    def __init__(self,preprocessed_images):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
#         self.train = train
        self.images = preprocessed_images
#         if self.train:
#             self.masks = preprocessed_masks
# #         self.masks = preprocessed_masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        return image
#         mask = None
#         if self.train:
#             mask = self.masks[idx]
#         return (image, mask)


# In[ ]:


# faults_dataset_test = faultsDataset(X_norm, train=False, preprocessed_masks=None)

# batch_size = 64

# test_loader = torch.utils.data.DataLoader(dataset=faults_dataset_test,
#                                            batch_size=batch_size,
#                                            shuffle=False)
faults_dataset_test = faultsDataset(X)

batch_size = 64

test_loader = torch.utils.data.DataLoader(dataset=faults_dataset_test,
                                           batch_size=batch_size,
                                           shuffle=False)


# In[ ]:


best_iou_threshold = 0.5
modelNo = 3
# device = torch.device("cuda")
if modelNo == 0:
    from model_zoo.UNET import Unet
    model = Unet()
    print("use model Unet")
    best_model_fpath = '/home/anyu/myproject/venv/an/all_model_new/unet_96_48_900200_seed.model' #unet_96_48_weightloss.model
    save_path = 'unet_96_48_900200_seed'
elif modelNo == 1:
    from model_zoo.DEEPLAB.deeplab import DeepLab
    model = DeepLab(backbone='mobilenet', num_classes=1, output_stride=16)
    print("use model DeepLab")
    best_model_fpath = 'saved/models/DeepLab/0307_130554/model_best.pth'
    save_path = 'deeplab_seed1_3'
elif modelNo == 2:
    from model_zoo.HED import HED
    model = HED()
    print("use model HED") #hed_96_48_aug_ShiftScaleRotate.model
    best_model_fpath = 'hed_192_96_900200_seed.model'
    save_path = 'hed_64_32_900200_seed'

elif modelNo == 3:
    from model_zoo.RCF import RCF
    model = RCF()
    print("use model RCF")
    best_model_fpath = 'saved/models/RCF/0309_220055/model_best.pth'
    save_path = 'rcf_96_48_900200_seed'

print(best_model_fpath)
print(save_path)
checkpoint = torch.load(best_model_fpath)
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict) # Choose whatever GPU device number you want
model.to(device)
# model.to(device);
summary(model, (1, splitsize, splitsize))#,device='cpu')


# In[ ]:


def saveResults(save_path, test_loader):
    WINDOW_SPLINE_2D = window_2D(window_size=splitsize, power=2)
    os.makedirs(save_path, exist_ok=True)

    test_predictions = []
    imageNo = -1
    for images in test_loader:
#         images = Variable(images)

        images = Variable(images.to(device))
        outputs = model(images)
        y_preds = outputs
        if modelNo == 2 or modelNo == 21:
            y_preds = outputs[-2]
        elif modelNo == 3:
            y_preds = outputs[-1]
#         predicted_mask = y_preds > best_iou_threshold
        test_predictions.extend(y_preds.detach().cpu())
#         print(test_predictions[0].dtype)
#         print(len(test_predictions))
        if len(test_predictions)>=vertical_splits_number*horizontal_splits_number:
            imageNo = imageNo + 1
            tosave = torch.stack(test_predictions).detach().cpu().numpy()[0:vertical_splits_number*horizontal_splits_number]
#             print(tosave.shape)
            test_predictions = test_predictions[vertical_splits_number*horizontal_splits_number:]


            tosave = np.moveaxis(tosave,-3,-1)
#             print(tosave.shape)
            tosave = np.array([patch * WINDOW_SPLINE_2D for patch in tosave])
#             print(tosave.shape)
#             break

            tosave = tosave.reshape((vertical_splits_number, horizontal_splits_number, splitsize,splitsize,1))
#             print(tosave.shape)

            recover_Y_test_pred = recover_Image2(tosave, (im_height,im_width,1), left_pad,right_pad,top_pad,bottom_pad,overlapsize)

            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path,"{}".format(imageNo)),np.squeeze(recover_Y_test_pred))

            

print("save")
t1 = time.time()
saveResults(save_path, test_loader)
t2 = time.time()
print('save in {} sec'.format(t2-t1))


# In[ ]:


# save_path = 'augtest/noaug'
import numpy as np
import os


# In[ ]:


a = np.load(os.path.join(save_path,"0.npy"))
# b = np.load(os.path.join("/home/anyu/myproject/venv/an/pieces/HED/dropout/0.2/testGTs","0.npy"))

import cv2
import cmapy


# In[ ]:


heatmap_img = cv2.applyColorMap((a*255).astype(np.uint8), cmapy.cmap('jet_r'))
plt.figure(figsize=(10,8))
plt.imshow(heatmap_img)
# plt.colorbar(shrink=0.5)
# plt.axis('off')
# plt.show()

plt.savefig('{}_0.png'.format(save_path))

