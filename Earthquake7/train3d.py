# coding: utf-8
import h5py
import numpy as np

# In[1]:

from model_zoo.UNET_3D import unet3d
from pytorchtools3 import EarlyStopping
import cmapy
from torchsummary import summary
from functions import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# in order to get reproducable results
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1)
random.seed(1)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Only GPU 1 is visible to this code
time1 = time.time()
# print("RandomBrightnessContrast")


# In[2]:

seismic_path = '/Users/taoshutong/Documents/Earthquake3/data/FYP_data/seis_sub_350IL_500t_1200XL.npy'
label_path = '/Users/taoshutong/Documents/Earthquake3/data/FYP_data/fault_sub_350IL_500t_1200XL.npy'
# In[3]:

t_start = time.time()
seismic = np.load(seismic_path, allow_pickle=True)
fault = np.load(label_path)
# print("load in {} sec".format(time.time() - t_start))


# In[4]:


# print(seismic.shape, fault.shape)
# seismic = np.moveaxis(seismic[4:1807],-2,-1)
# print(seismic.shape, fault.shape)
# print(seismic.max(), seismic.min(), fault.max(), fault.min())
# # reorder input data to same order IL, Z, XL


# # In[5]:


# seismic = (seismic-seismic.min(axis=(1,2), keepdims=True))/(seismic.max(axis=(1,2), keepdims=True)-seismic.min(axis=(1,2), keepdims=True))
# print(seismic.shape)


# # In[6]:


IL, Z, XL = fault.shape

# In[7]:


best_model_fpath = 'hed_192_96_900200_seed.model'
best_iou_threshold = 0.5
epoches = 10
patience = 20
im_length = IL
im_height = Z
im_width = XL
splitsize = 64
stepsize = 48
overlapsize = splitsize - stepsize
pixelThre = int(0.03 * splitsize * splitsize)
# print(pixelThre)

# In[8]:


model = unet3d(in_dim=1, out_dim=1, num_filters=4)
print("use model UNET_3D")

model.cpu()
summary(model, input_size=(1, splitsize, splitsize, splitsize))

horizontal_splits_number = int(np.ceil((im_width - overlapsize) / stepsize))
print("horizontal_splits_number", horizontal_splits_number)
width_after_pad = stepsize * horizontal_splits_number + overlapsize
print("width_after_pad", width_after_pad)
left_pad = int((width_after_pad - im_width) / 2)
right_pad = width_after_pad - im_width - left_pad
print("left_pad,right_pad", left_pad, right_pad)

vertical_splits_number = int(np.ceil((im_height - overlapsize) / stepsize))
print("vertical_splits_number", vertical_splits_number)
height_after_pad = stepsize * vertical_splits_number + overlapsize
print("height_after_pad", height_after_pad)
top_pad = int((height_after_pad - im_height) / 2)
bottom_pad = height_after_pad - im_height - top_pad
print("top_pad,bottom_pad", top_pad, bottom_pad)

deepth_splits_number = int(np.ceil((im_length - overlapsize) / stepsize))
print("deepth_splits_number", deepth_splits_number)
deepth_after_pad = stepsize * deepth_splits_number + overlapsize
print("deepth_after_pad", deepth_after_pad)
front_pad = int((deepth_after_pad - im_length) / 2)
back_pad = deepth_after_pad - im_length - front_pad
print("front_pad,back_pad", front_pad, back_pad)

# In[17]:
# train_dataset

t_start = time.time()


def split3D(vol, l, step):
    smallVols = []
    r = int(l / 2)
    for i in range(r, vol.shape[0] - r, step):
        for j in range(r, vol.shape[1] - r, step):
            for k in range(r, vol.shape[2] - r, step):
                smallVols.append(vol[i - r:i + r, j - r:j + r, k - r:k + r])
    return smallVols


X = np.asarray(split3D(seismic, 64, 48), dtype=np.float32)

Y = np.asarray(split3D(fault, 64, 48), dtype=np.float32)

print('-----------><', X.shape)
print(Y.shape)

# In[22]:


if len(Y.shape) == 3:
    Y = np.expand_dims(Y, axis=-1)
if len(X.shape) == 3:
    X = np.expand_dims(X, axis=-1)
print(X.shape)
print(Y.shape)

X = np.expand_dims(X, axis=1)
Y = np.expand_dims(Y, axis=1)
X_train = X[0:1200, :, :, :]
Y_train = Y[0:1200, :, :, :]

X_val = X[1200:1440, :, :, :]
Y_val = Y[1200:1440, :, :, :]

print("X_train", X_train.shape)
print("X_val", X_val.shape)

print("Y_train", Y_train.shape)
print("Y_val", Y_val.shape)


print('=================', X_train.max(), X_train.min(), Y_train.max(), Y_train.min())
print('====================', X_val.max(), X_val.min(), Y_val.max(), Y_val.min())
# In[ ]:

# dataset
# idea from: https://www.kaggle.com/erikistre/pytorch-basic-u-net


class faultsDataset(torch.utils.data.Dataset):

    def __init__(self, preprocessed_images, train=True, preprocessed_masks=None):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.train = train
        self.images = preprocessed_images
        self.masks = preprocessed_masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        #         mask = None
        # #         if self.train:
        mask = self.masks[idx]
        return (image, mask)


# In[32]:


faults_dataset_train = faultsDataset(
    X_train, train=True, preprocessed_masks=Y_train)
faults_dataset_val = faultsDataset(
    X_val, train=False, preprocessed_masks=Y_val)
# faults_dataset_test = faultsDataset(X_test, train=False, preprocessed_masks=Y_test)

batch_size = 8

train_loader = torch.utils.data.DataLoader(dataset=faults_dataset_train,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=faults_dataset_val,
                                         batch_size=batch_size,
                                         shuffle=False)

# test_loader = torch.utils.data.DataLoader(dataset=faults_dataset_test,
#                                            batch_size=batch_size,
#                                            shuffle=False)


# In[ ]:


# criterion = nn.BCEWithLogitsLoss()
# learning_rate = 0.01
optimizer = torch.optim.SGD(
    model.parameters(), lr=1e-6, momentum=0.9, weight_decay=0.0002)
print("optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9, weight_decay=0.0002)")

print("optimizer = torch.optim.Adam(model.parameters(), lr=0.01)")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', factor=0.1, patience=5, verbose=True)

# In[ ]:


bceloss = nn.BCELoss()
mean_train_losses = []
mean_val_losses = []
mean_train_accuracies = []
mean_val_accuracies = []
t_start = time.time()
early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0)
for epoch in range(epoches):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    labelled_val_accuracies = []
    print('----------------------------------------------------')
    model.train()
    print('----------------------------------------------------')
    for images, masks in train_loader:
        images = Variable(images.to(device))
        masks = Variable(masks.to(device))

        outputs = model(images)

        loss = torch.zeros(1).to(device)
        y_preds = outputs

        loss = bceloss(outputs, masks)
        #             loss = cross_entropy_loss_HED(outputs, masks)
        #             loss = nn.BCEWithLogitsLoss(outputs, masks)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_losses.append(loss.data)
        predicted_mask = y_preds > best_iou_threshold
        train_acc = iou_pytorch(predicted_mask.squeeze(
            1).byte(), masks.squeeze(1).byte())
        train_accuracies.append(train_acc.mean())

    model.eval()
    for images, masks in val_loader:
        images = Variable(images.to(device))
        masks = Variable(masks.to(device))

        outputs = model(images)

        loss = torch.zeros(1).to(device)
        y_preds = outputs

        #             print("bceloss")
        loss = bceloss(outputs, masks)
        #             loss = cross_entropy_loss_HED(outputs, masks)
        #             loss = nn.BCEWithLogitsLoss(outputs, masks)

        val_losses.append(loss.data)
        predicted_mask = y_preds > best_iou_threshold
        val_acc = iou_pytorch(predicted_mask.byte(), masks.squeeze(1).byte())
        val_accuracies.append(val_acc.mean())

    #         todelete = torch.sum(masks,dim=(2,3))<1
    #         no_label_element_index = list(compress(range(len(todelete)), todelete))
    # #         print(no_label_element_index)
    #         labelled_val_acc = np.delete(val_acc, no_label_element_index,0)
    #         labelled_val_accuracies.extend(labelled_val_acc)

    mean_train_losses.append(torch.mean(torch.stack(train_losses)))
    mean_val_losses.append(torch.mean(torch.stack(val_losses)))
    mean_train_accuracies.append(torch.mean(torch.stack(train_accuracies)))
    mean_val_accuracies.append(torch.mean(torch.stack(val_accuracies)))

    scheduler.step(torch.mean(torch.stack(val_losses)))
    early_stopping(torch.mean(torch.stack(val_losses)),
                   model, best_model_fpath)

    #     mean_train_losses.append(np.mean(train_losses))
    #     mean_val_losses.append(np.mean(val_losses))
    #     mean_train_accuracies.append(np.mean(train_accuracies))
    #     mean_val_accuracies.append(np.mean(val_accuracies))
    #     scheduler.step(np.mean(val_losses))
    #     early_stopping(np.mean(val_losses), model, best_model_fpath)

    if early_stopping.early_stop:
        print("Early stopping")
        break

    #     # load the last checkpoint with the best model
    #     model.load_state_dict(torch.load('checkpoint.pt'))
    # torch.cuda.empty_cache()

    for param_group in optimizer.param_groups:
        learningRate = param_group['lr']

    # Print Epoch results
    t_end = time.time()
    #     print('Epoch: {}. Train Loss: {}. Val Loss: {}. Train IoU: {}. Val IoU: {}. Time: {}. LR: {}'
    #           .format(epoch+1, np.mean(train_losses), np.mean(val_losses), torch.mean(torch.stack(train_accuracies)).item(), torch.mean(torch.stack(val_accuracies)).item(), t_end-t_start, learningRate))
    #     t_start = time.time()
    #     print('Epoch: {}. Train Loss: {}. Val Loss: {}. Train IoU: {}. Val IoU: {}. Labelled Val IoU: {}. Time: {}. LR: {}'
    #           .format(epoch+1, np.mean(train_losses), np.mean(val_losses),np.mean(train_accuracies), np.mean(val_accuracies), np.mean(labelled_val_accuracies), t_end-t_start, learningRate))
    print('Epoch: {}. Train Loss: {}. Val Loss: {}. Train IoU: {}. Val IoU: {}. Time: {}. LR: {}'
          .format(epoch + 1, torch.mean(torch.stack(train_losses)), torch.mean(torch.stack(val_losses)),
                  torch.mean(torch.stack(train_accuracies)), torch.mean(
            torch.stack(val_accuracies)), t_end - t_start,
                  learningRate))

    t_start = time.time()

#     torch.save(model.state_dict(), best_model_fpath)


# In[ ]:
mean_train_losses = np.asarray(torch.stack(mean_train_losses).cpu())
mean_val_losses = np.asarray(torch.stack(mean_val_losses).cpu())
mean_train_accuracies = np.asarray(torch.stack(mean_train_accuracies).cpu())
mean_val_accuracies = np.asarray(torch.stack(mean_val_accuracies).cpu())

fig = plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
train_loss_series = pd.Series(mean_train_losses)
val_loss_series = pd.Series(mean_val_losses)
train_loss_series.plot(label="train_loss")
val_loss_series.plot(label="validation_loss")
plt.legend()
plt.subplot(1, 2, 2)
train_acc_series = pd.Series(mean_train_accuracies)
val_acc_series = pd.Series(mean_val_accuracies)
train_acc_series.plot(label="train_acc")
val_acc_series.plot(label="validation_acc")
plt.legend()
plt.savefig('{}_loss_acc.png'.format(best_model_fpath))

totaltime = time.time() - time1
print("total cost {} hours".format(totaltime / 3600))

