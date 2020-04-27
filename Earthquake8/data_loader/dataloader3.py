
from functions import *
from scipy.ndimage import filters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train/validation/test  用此dataloader
class EarthquakeDataLoader():
    def __init__(self,seismic_path,label_path,batch_size,train_number_of_pictures=1, val_number_of_pictures=1,train_start=1,val_start=0,dimension=0):
        self.train_loader,self.val_loader = self.train_val_dataset(seismic_path,label_path,batch_size,train_number_of_pictures, val_number_of_pictures,train_start,val_start,dimension)

    def train_val_dataset(self, seismic_path,label_path, batch_size, train_number_of_pictures, val_number_of_pictures, train_start, val_start,dimension):
        # dimension=1;

        os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Only GPU 1 is visible to this code
        time1 = time.time()
        # print("RandomBrightnessContrast")
        # modelNo
        # Unet --> 0
        # Deeplab --> 1
        # HED --> 2
        # RCF --> 3
        # CED --> 4

        t_start = time.time()
        seismic = np.load(seismic_path)
        fault = np.load(label_path)
        # print("load in {} sec".format(time.time() - t_start))

        # In[4]:

        # print(seismic.shape, fault.shape)
        # seismic = np.moveaxis(seismic[4:1807],-2,-1)
        # print("seismic.shape, fault.shape",seismic.shape, fault.shape)
        # print("seismic.max(), seismic.min(), fault.max(), fault.min()",seismic.max(), seismic.min(), fault.max(), fault.min())
        # # reorder input data to same order IL, Z, XL

        # # In[5]:

        # seismic = (seismic-seismic.min(axis=(1,2), keepdims=True))/(seismic.max(axis=(1,2), keepdims=True)-seismic.min(axis=(1,2), keepdims=True))
        # print(seismic.shape)

        # # In[6]:

        IL, Z, XL = fault.shape
        if dimension == 0:
            im_height = Z
            im_width = XL
        elif dimension == 1:
            im_height = IL
            im_width = XL
        elif dimension == 2:
            im_height = IL
            im_width = Z
        # im_height = Z
        # im_width = XL
        splitsize = 96
        stepsize = 48
        overlapsize = splitsize - stepsize
        pixelThre = int(0.03 * splitsize * splitsize)
        # print("pixelThre ",pixelThre)

        # In[8]:


        horizontal_splits_number = int(np.ceil((im_width - overlapsize) / stepsize))
        # print("horizontal_splits_number", horizontal_splits_number)
        width_after_pad = stepsize * horizontal_splits_number + overlapsize
        # print("width_after_pad", width_after_pad)
        left_pad = int((width_after_pad - im_width) / 2)
        right_pad = width_after_pad - im_width - left_pad
        # print("left_pad,right_pad", left_pad, right_pad)

        vertical_splits_number = int(np.ceil((im_height - overlapsize) / stepsize))
        # print("vertical_splits_number", vertical_splits_number)
        height_after_pad = stepsize * vertical_splits_number + overlapsize
        # print("height_after_pad", height_after_pad)
        top_pad = int((height_after_pad - im_height) / 2)
        bottom_pad = height_after_pad - im_height - top_pad
        # print("top_pad,bottom_pad", top_pad, bottom_pad)

        # In[17]:
        # train_dataset

        t_start = time.time()
        X = []
        Y = []

        for i in range(train_start, train_number_of_pictures+train_start, 1):
            if dimension == 0:
                mask =  fault[i]
            elif dimension == 1:
                mask = fault[:,i,:]
            elif dimension ==2:
                mask = fault[:,:,i]
            # mask = fault[i]
            # print("mask:         ",mask.shape)
            splits = split_Image(mask, True, top_pad, bottom_pad, left_pad, right_pad, splitsize, stepsize,
                                 vertical_splits_number, horizontal_splits_number)
            #     print(splits.shape)
            t = (splits.sum((1, 2)) < pixelThre)
            no_label_element_index = list(compress(range(len(t)), t))
            # get all the indexes of the no label pieces by adding elements in axis 2 and 3.
            splits = np.delete(splits, no_label_element_index, 0)  # delete element i along axis 0
            #     print("splits.shape", splits.shape)
            Y.extend(splits)



            if dimension == 0:
                img =  seismic[i]
            elif dimension == 1:
                img = seismic[:,i,:]
            elif dimension ==2:
                img = seismic[:,:,i]
            # img = seismic[i]

            splits = split_Image(img, True, top_pad, bottom_pad, left_pad, right_pad, splitsize, stepsize,
                                 vertical_splits_number, horizontal_splits_number)
            splits = np.delete(splits, no_label_element_index, 0)  # delete element i along axis 0
            #     print("splits.shape", splits.shape)
            X.extend(splits)
        #     break

        # print("len(Y)",len(Y))
        # print("len(X)",len(X))
        # print("X[0].shape",X[0].shape)
        # print("read images in {} sec".format(time.time() - t_start))

        # In[20]:

        t_start = time.time()
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        # print(X.shape)
        # print(Y.shape)
        # print("read images in {} sec".format(time.time() - t_start))

        # In[22]:

        if len(Y.shape) == 3:
            Y = np.expand_dims(Y, axis=-1)
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=-1)
        # print(X.shape)
        # print(Y.shape)

        X_train = X
        Y_train = Y













        t_start = time.time()
        X = []
        Y = []
        for i in range(val_start, val_number_of_pictures+val_start, 1):
            if dimension == 0:
                mask =  fault[i]
            elif dimension == 1:
                mask = fault[:,i,:]
            elif dimension ==2:
                mask = fault[:,:,i]
            # mask = fault[i]
            splits = split_Image(mask, True, top_pad, bottom_pad, left_pad, right_pad, splitsize, stepsize,
                                 vertical_splits_number, horizontal_splits_number)
            #     print(splits.shape)
            t = (splits.sum((1, 2)) < pixelThre)
            no_label_element_index = list(compress(range(len(t)), t))
            # get all the indexes of the no label pieces by adding elements in axis 2 and 3.
            splits = np.delete(splits, no_label_element_index, 0)  # delete element i along axis 0
            #     print("splits.shape", splits.shape)
            Y.extend(splits)


            if dimension == 0:
                img =  seismic[i]
            elif dimension == 1:
                img = seismic[:,i,:]
            elif dimension ==2:
                img = seismic[:,:,i]
            # img = seismic[i]
            splits = split_Image(img, True, top_pad, bottom_pad, left_pad, right_pad, splitsize, stepsize,
                                 vertical_splits_number, horizontal_splits_number)
            splits = np.delete(splits, no_label_element_index, 0)  # delete element i along axis 0
            #     print("splits.shape", splits.shape)
            X.extend(splits)
        #     break

        # print(len(Y))
        # print(len(X))
        # print(X[0].shape)
        # print("read images in {} sec".format(time.time() - t_start))

        # In[20]:

        t_start = time.time()
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        # print(X.shape)
        # print(Y.shape)
        # print("read images in {} sec".format(time.time() - t_start))

        # In[22]:

        if len(Y.shape) == 3:
            Y = np.expand_dims(Y, axis=-1)
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=-1)
        # print(X.shape)
        # print(Y.shape)

        X_val = X
        Y_val = Y

        # print("X_train", X_train.shape)
        # print("X_val", X_val.shape)
        #
        # print("Y_train", Y_train.shape)
        # print("Y_val", Y_val.shape)

        # In[ ]:

        # aug_times = 1

        # t_start = time.time()
        # origin_train_size = len(X_train)
        # print(origin_train_size)
        # X_train_aug = np.zeros((origin_train_size*aug_times,splitsize,splitsize,1))
        # Y_train_aug = np.zeros((origin_train_size*aug_times,splitsize,splitsize,1))
        # for i in range(len(X_train)):
        #     for j in range(aug_times):
        #         aug = strong_aug(p=1)
        #         augmented = aug(image=X_train[i], mask=Y_train[i])
        #         X_train_aug[origin_train_size*j + i] = augmented['image']
        #         Y_train_aug[origin_train_size*j + i] = augmented['mask']
        # print("read images in {} sec".format(time.time()-t_start))

        # X_train_aug = X_train_aug.astype(np.float32)
        # Y_train_aug = Y_train_aug.astype(np.float32)
        # if len(X_train)==origin_train_size:
        #     X_train = np.append(X_train,X_train_aug, axis=0)
        # if len(Y_train)==origin_train_size:
        #     Y_train = np.append(Y_train, Y_train_aug, axis=0)
        # print("X_train after aug",X_train.shape)
        # print("Y_train after aug",Y_train.shape)
        # print("read images in {} sec".format(time.time()-t_start))
        # X_train = X_train.astype(np.float32)
        # Y_train = Y_train.astype(np.float32)
        # #-----------------------
        # t_start = time.time()
        # origin_val_size = len(X_val)
        # print(origin_val_size)
        # X_val_aug = np.zeros((origin_val_size*aug_times,splitsize,splitsize,1))
        # Y_val_aug = np.zeros((origin_val_size*aug_times,splitsize,splitsize,1))
        # for i in range(len(X_val)):
        #     for j in range(aug_times):
        #         aug = strong_aug(p=1)
        #         augmented = aug(image=X_train[i], mask=Y_train[i])
        #         X_val_aug[origin_val_size*j + i] = augmented['image']
        #         Y_val_aug[origin_val_size*j + i] = augmented['mask']
        # print("read images in {} sec".format(time.time()-t_start))

        # X_val_aug = X_val_aug.astype(np.float32)
        # Y_val_aug = Y_val_aug.astype(np.float32)
        # if len(X_val)==origin_val_size:
        #     X_val = np.append(X_val,X_val_aug, axis=0)
        # if len(Y_val)==origin_val_size:
        #     Y_val = np.append(Y_val, Y_val_aug, axis=0)
        # print("X_val after aug",X_val.shape)
        # print("Y_val after aug",Y_val.shape)
        # print("read images in {} sec".format(time.time()-t_start))
        # X_val = X_val.astype(np.float32)
        # Y_val = Y_val.astype(np.float32)
        # -----------------------
        X_train = np.moveaxis(X_train, -1, 1)
        Y_train = np.moveaxis(Y_train, -1, 1)
        X_val = np.moveaxis(X_val, -1, 1)
        Y_val = np.moveaxis(Y_val, -1, 1)
        # print("X_train", X_train.shape)
        # print("X_val", X_val.shape)
        # print("Y_train", Y_train.shape)
        # print("Y_val", Y_val.shape)

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

        faults_dataset_train = faultsDataset(X_train, train=True, preprocessed_masks=Y_train)
        faults_dataset_val = faultsDataset(X_val, train=False, preprocessed_masks=Y_val)
        # faults_dataset_test = faultsDataset(X_test, train=False, preprocessed_masks=Y_test)



        train_loader = torch.utils.data.DataLoader(dataset=faults_dataset_train,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        val_loader = torch.utils.data.DataLoader(dataset=faults_dataset_val,
                                                 batch_size=batch_size,
                                                 shuffle=False)

        # test_loader = torch.utils.data.DataLoader(dataset=faults_dataset_test,
        #                                            batch_size=batch_size,
        #                                            shuffle=False)
        return train_loader,val_loader


# save——predicted——picture 用此dataloader
class EarthquakeDataLoader2():
    def __init__(self,seismic_path,label_path,batch_size, val_number_of_pictures,val_start,dimension=0):
       self.test_loader,self.faults_dataset_test = self.train_val_dataset(seismic_path,batch_size, val_number_of_pictures,val_start,dimension)

    def train_val_dataset(self, seismic_path, batch_size, val_number_of_pictures, val_start,dimension):
        seis = np.load(seismic_path)
        IL, Z, XL = seis.shape
        # print("IL, Z, XL",IL, Z, XL)
        # print(seis.shape)
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
        # print(horizontal_splits_number)
        width_after_pad = stepsize * horizontal_splits_number + 2 * overlapsize
        # print(width_after_pad)
        left_pad = int((width_after_pad - im_width) / 2)
        right_pad = width_after_pad - im_width - left_pad
        # print(left_pad, right_pad)

        vertical_splits_number = int(np.ceil((im_height) / stepsize))
        # print(vertical_splits_number)
        height_after_pad = stepsize * vertical_splits_number + 2 * overlapsize
        # print(height_after_pad)
        top_pad = int((height_after_pad - im_height) / 2)
        bottom_pad = height_after_pad - im_height - top_pad
        # print(top_pad, bottom_pad)

        horizontal_splits_number = horizontal_splits_number + 1
        # print(horizontal_splits_number)
        vertical_splits_number = vertical_splits_number + 1
        # print(vertical_splits_number)

        halfoverlapsize = int(overlapsize / 2)
        # print(halfoverlapsize)

        # In[ ]:

        # print(len(seis))


        t_start = time.time()
        X_list = []
        for i in range(val_start, val_start+val_number_of_pictures, 1):
            if dimension == 0:
                img =  seis[i]
            elif dimension == 1:
                img = seis[:,i,:]
            elif dimension ==2:
                img = seis[:,:,i]
            # img = seis[i]

            X_list.extend(split_Image(img, True, top_pad, bottom_pad, left_pad, right_pad, splitsize, stepsize,
                                      vertical_splits_number, horizontal_splits_number))

        # print(len(X_list))
        # print(X_list[0].shape)
        # print("read images in {} sec".format(time.time() - t_start))
        X = np.asarray(X_list)
        # print(X.shape)
        # print("read images in {} sec".format(time.time() - t_start))


        # In[ ]:

        X = X.astype(np.float32)
        X = np.expand_dims(X, 1)
        # print(X.shape)

        # idea from: https://www.kaggle.com/erikistre/pytorch-basic-u-net
        class faultsDataset(torch.utils.data.Dataset):

            def __init__(self, preprocessed_images):

                self.images = preprocessed_images

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                image = self.images[idx]
                return image

        faults_dataset_test = faultsDataset(X)

        test_loader = torch.utils.data.DataLoader(dataset=faults_dataset_test,
                                                  batch_size=batch_size,
                                                  shuffle=False)

        return test_loader,faults_dataset_test

    
# 3d train/validation/test  用此dataloader
class EarthquakeDataLoader3d():
    def __init__(self,seismic_path,label_path,batch_size,train_number=1, val_number=1,train_start=1,val_start=0,dimension=0):
        self.train_loader,self.val_loader = self.train_val_dataset(seismic_path,label_path,batch_size,train_number, val_number,train_start,val_start,dimension)

    def train_val_dataset(self, seismic_path,label_path, batch_size, train_number, val_number, train_start, val_start, dimension):

        os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Only GPU 1 is visible to this code

        seismic = np.load(seismic_path)
        fault = np.load(label_path)

        splitsize = 64
        stepsize = 48
        pixelThre = int(0.03 * splitsize * splitsize * splitsize)

        # train_dataset

        t_start = time.time()

        def split3D(vol, l, step):
            smallVols = []
            r = int(l / 2)
            for i in range(r, vol.shape[0] - r + 1, step):
                for j in range(r, vol.shape[1] - r + 1, step):
                    for k in range(r, vol.shape[2] - r + 1, step):
                        smallVols.append(vol[i - r:i + r, j - r:j + r, k - r:k + r])

            return smallVols

        X_train = np.asarray(split3D(seismic[train_start:train_number+train_start,:,:], 64, 48), dtype=np.float32)
        Y_train = np.asarray(split3D(fault[train_start:train_number+train_start,:,:], 64, 48), dtype=np.float32)

        X_val = np.asarray(split3D(seismic[val_start:val_number+val_start,:,:], 64, 48), dtype=np.float32)
        Y_val = np.asarray(split3D(fault[val_start:val_number+val_start,:,:], 64, 48), dtype=np.float32)

        # t = (Y_train.sum((1, 2, 3)) < pixelThre)
        # no_label_element_index = list(compress(range(len(t)), t))
        # Y_train = np.delete(Y_train, no_label_element_index, 0)
        # X_train = np.delete(X_train, no_label_element_index, 0)

        # t = (Y_val.sum((1, 2, 3)) < pixelThre)
        # no_label_element_index = list(compress(range(len(t)), t))
        # Y_val = np.delete(Y_val, no_label_element_index, 0)
        # X_val = np.delete(X_val, no_label_element_index, 0)

        print('-----------><', X_train.shape)
        print(Y_train.shape)


        # if len(Y_train.shape) == 3:
        #     Y_train = np.expand_dims(Y_train, axis=-1)
        # if len(X_train.shape) == 3:
        #     X_train = np.expand_dims(X_train, axis=-1)

        X_train = np.expand_dims(X_train, axis=1)
        Y_train = np.expand_dims(Y_train, axis=1)
        
        # if len(Y_val.shape) == 3:
        #     Y_val = np.expand_dims(Y_val, axis=-1)
        # if len(X_val.shape) == 3:
        #     X_val = np.expand_dims(X_val, axis=-1)

        X_val = np.expand_dims(X_val, axis=1)
        Y_val = np.expand_dims(Y_val, axis=1)


        print("X_train", X_train.shape)
        print("X_val", X_val.shape)

        print("Y_train", Y_train.shape)
        print("Y_val", Y_val.shape)

        print('=================', X_train.max(), X_train.min(), Y_train.max(), Y_train.min())
        print('====================', X_val.max(), X_val.min(), Y_val.max(), Y_val.min())


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
                mask = self.masks[idx]
                return (image, mask)

        faults_dataset_train = faultsDataset(X_train, train=True, preprocessed_masks=Y_train)
        faults_dataset_val = faultsDataset(X_val, train=False, preprocessed_masks=Y_val)


        train_loader = torch.utils.data.DataLoader(dataset=faults_dataset_train,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=faults_dataset_val,
                                                 batch_size=batch_size,
                                                 shuffle=False)
        return train_loader,val_loader

# save——predicted——picture 用此dataloader
class EarthquakeDataLoader3dtest():
    def __init__(self,seismic_path,label_path,batch_size, test_number_of_pictures,test_start,dimension=0):
       self.test_loader,self.faults_dataset_test = self.test_dataset(seismic_path,batch_size, test_number_of_pictures,test_start,dimension)

    def test_dataset(self, seismic_path, batch_size, test_number_of_pictures, test_start,dimension):
        seis = np.load(seismic_path)

        def split3D(vol, l, step):
            smallVols = []
            r = int(l / 2)
            for i in range(r, vol.shape[0] - r + 1, step):
                for j in range(r, vol.shape[1] - r + 1, step):
                    for k in range(r, vol.shape[2] - r + 1, step):
                        smallVols.append(vol[i - r:i + r, j - r:j + r, k - r:k + r])
            return smallVols
        X_test = np.asarray(split3D(seis[test_start:test_start+test_number_of_pictures,:,:], 64, 48), dtype=np.float32)
        print(X_test.shape)
        X = np.expand_dims(X_test, axis=1)
        print(X.shape)

        class faultsDataset(torch.utils.data.Dataset):

            def __init__(self, preprocessed_images):

                self.images = preprocessed_images

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                image = self.images[idx]
                return image

        faults_dataset_test = faultsDataset(X)

        test_loader = torch.utils.data.DataLoader(dataset=faults_dataset_test,
                                                  batch_size=batch_size,
                                                  shuffle=False)

        return test_loader, faults_dataset_test

# gan 用此dataloader
class EarthquakeDataLoader_gan():
    def __init__(self, seismic_path, label_path, batch_size, train_number_of_pictures=1, val_number_of_pictures=1,
                 train_start=1, val_start=0, dimension=0):
        self.train_loader, self.val_loader = self.train_val_dataset(seismic_path, label_path, batch_size,
                                                                    train_number_of_pictures, val_number_of_pictures,
                                                                    train_start, val_start, dimension)

    def train_val_dataset(self, seismic_path, label_path, batch_size, train_number_of_pictures, val_number_of_pictures,
                          train_start, val_start, dimension):
        # dimension=1;

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only GPU 1 is visible to this code
        time1 = time.time()
        # print("RandomBrightnessContrast")
        # modelNo
        # Unet --> 0
        # Deeplab --> 1
        # HED --> 2
        # RCF --> 3
        # CED --> 4

        t_start = time.time()
        seismic = np.load(seismic_path)
        fault = np.load(label_path)
        # label=original high resolution
        fault = seismic
        img1, img2 = preprocess(seismic)
        fault = img2
        # input = low resolution
        seismic = filters.gaussian_filter(img1, 1)
        # seismic =  filters.gaussian_filter(seismic,1)

        # print("load in {} sec".format(time.time() - t_start))

        # In[4]:

        # print(seismic.shape, fault.shape)
        # seismic = np.moveaxis(seismic[4:1807],-2,-1)
        # print("seismic.shape, fault.shape",seismic.shape, fault.shape)
        # print("seismic.max(), seismic.min(), fault.max(), fault.min()",seismic.max(), seismic.min(), fault.max(), fault.min())
        # # reorder input data to same order IL, Z, XL

        # # In[5]:

        # seismic = (seismic-seismic.min(axis=(1,2), keepdims=True))/(seismic.max(axis=(1,2), keepdims=True)-seismic.min(axis=(1,2), keepdims=True))
        # print(seismic.shape)

        # # In[6]:

        IL, Z, XL = fault.shape
        if dimension == 0:
            im_height = Z
            im_width = XL
        elif dimension == 1:
            im_height = IL
            im_width = XL
        elif dimension == 2:
            im_height = IL
            im_width = Z
        # im_height = Z
        # im_width = XL
        splitsize = 96
        stepsize = 48
        overlapsize = splitsize - stepsize
        pixelThre = int(0.03 * splitsize * splitsize)
        # print("pixelThre ",pixelThre)

        # In[8]:

        horizontal_splits_number = int(np.ceil((im_width - overlapsize) / stepsize))
        # print("horizontal_splits_number", horizontal_splits_number)
        width_after_pad = stepsize * horizontal_splits_number + overlapsize
        # print("width_after_pad", width_after_pad)
        left_pad = int((width_after_pad - im_width) / 2)
        right_pad = width_after_pad - im_width - left_pad
        # print("left_pad,right_pad", left_pad, right_pad)

        vertical_splits_number = int(np.ceil((im_height - overlapsize) / stepsize))
        # print("vertical_splits_number", vertical_splits_number)
        height_after_pad = stepsize * vertical_splits_number + overlapsize
        # print("height_after_pad", height_after_pad)
        top_pad = int((height_after_pad - im_height) / 2)
        bottom_pad = height_after_pad - im_height - top_pad
        # print("top_pad,bottom_pad", top_pad, bottom_pad)

        # In[17]:
        # train_dataset

        t_start = time.time()
        X = []
        Y = []

        for i in range(train_start, train_number_of_pictures + train_start, 1):
            if dimension == 0:
                mask = fault[i]
            elif dimension == 1:
                mask = fault[:, i, :]
            elif dimension == 2:
                mask = fault[:, :, i]
            # mask = fault[i]
            # print("mask:         ",mask.shape)
            splits = split_Image(mask, True, top_pad, bottom_pad, left_pad, right_pad, splitsize, stepsize,
                                 vertical_splits_number, horizontal_splits_number)
            #     print(splits.shape)
            t = (splits.sum((1, 2)) < pixelThre)
            no_label_element_index = list(compress(range(len(t)), t))
            # get all the indexes of the no label pieces by adding elements in axis 2 and 3.
            splits = np.delete(splits, no_label_element_index, 0)  # delete element i along axis 0
            #     print("splits.shape", splits.shape)
            Y.extend(splits)

            if dimension == 0:
                img = seismic[i]
            elif dimension == 1:
                img = seismic[:, i, :]
            elif dimension == 2:
                img = seismic[:, :, i]
            # img = seismic[i]

            splits = split_Image(img, True, top_pad, bottom_pad, left_pad, right_pad, splitsize, stepsize,
                                 vertical_splits_number, horizontal_splits_number)
            splits = np.delete(splits, no_label_element_index, 0)  # delete element i along axis 0
            #     print("splits.shape", splits.shape)
            X.extend(splits)
        #     break

        # print("len(Y)",len(Y))
        # print("len(X)",len(X))
        # print("X[0].shape",X[0].shape)
        # print("read images in {} sec".format(time.time() - t_start))

        # In[20]:

        t_start = time.time()
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        # print(X.shape)
        # print(Y.shape)
        # print("read images in {} sec".format(time.time() - t_start))

        # In[22]:

        if len(Y.shape) == 3:
            Y = np.expand_dims(Y, axis=-1)
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=-1)
        # print(X.shape)
        # print(Y.shape)

        X_train = X
        Y_train = Y

        t_start = time.time()
        X = []
        Y = []
        for i in range(val_start, val_number_of_pictures + val_start, 1):
            if dimension == 0:
                mask = fault[i]
            elif dimension == 1:
                mask = fault[:, i, :]
            elif dimension == 2:
                mask = fault[:, :, i]
            # mask = fault[i]
            splits = split_Image(mask, True, top_pad, bottom_pad, left_pad, right_pad, splitsize, stepsize,
                                 vertical_splits_number, horizontal_splits_number)
            #     print(splits.shape)
            t = (splits.sum((1, 2)) < pixelThre)
            no_label_element_index = list(compress(range(len(t)), t))
            # get all the indexes of the no label pieces by adding elements in axis 2 and 3.
            splits = np.delete(splits, no_label_element_index, 0)  # delete element i along axis 0
            #     print("splits.shape", splits.shape)
            Y.extend(splits)

            if dimension == 0:
                img = seismic[i]
            elif dimension == 1:
                img = seismic[:, i, :]
            elif dimension == 2:
                img = seismic[:, :, i]
            # img = seismic[i]
            splits = split_Image(img, True, top_pad, bottom_pad, left_pad, right_pad, splitsize, stepsize,
                                 vertical_splits_number, horizontal_splits_number)
            splits = np.delete(splits, no_label_element_index, 0)  # delete element i along axis 0
            #     print("splits.shape", splits.shape)
            X.extend(splits)
        #     break

        # print(len(Y))
        # print(len(X))
        # print(X[0].shape)
        # print("read images in {} sec".format(time.time() - t_start))

        # In[20]:

        t_start = time.time()
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        # print(X.shape)
        # print(Y.shape)
        # print("read images in {} sec".format(time.time() - t_start))

        # In[22]:

        if len(Y.shape) == 3:
            Y = np.expand_dims(Y, axis=-1)
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=-1)
        # print(X.shape)
        # print(Y.shape)

        X_val = X
        Y_val = Y

        # print("X_train", X_train.shape)
        # print("X_val", X_val.shape)
        #
        # print("Y_train", Y_train.shape)
        # print("Y_val", Y_val.shape)

        # In[ ]:

        # aug_times = 1

        # t_start = time.time()
        # origin_train_size = len(X_train)
        # print(origin_train_size)
        # X_train_aug = np.zeros((origin_train_size*aug_times,splitsize,splitsize,1))
        # Y_train_aug = np.zeros((origin_train_size*aug_times,splitsize,splitsize,1))
        # for i in range(len(X_train)):
        #     for j in range(aug_times):
        #         aug = strong_aug(p=1)
        #         augmented = aug(image=X_train[i], mask=Y_train[i])
        #         X_train_aug[origin_train_size*j + i] = augmented['image']
        #         Y_train_aug[origin_train_size*j + i] = augmented['mask']
        # print("read images in {} sec".format(time.time()-t_start))

        # X_train_aug = X_train_aug.astype(np.float32)
        # Y_train_aug = Y_train_aug.astype(np.float32)
        # if len(X_train)==origin_train_size:
        #     X_train = np.append(X_train,X_train_aug, axis=0)
        # if len(Y_train)==origin_train_size:
        #     Y_train = np.append(Y_train, Y_train_aug, axis=0)
        # print("X_train after aug",X_train.shape)
        # print("Y_train after aug",Y_train.shape)
        # print("read images in {} sec".format(time.time()-t_start))
        # X_train = X_train.astype(np.float32)
        # Y_train = Y_train.astype(np.float32)
        # #-----------------------
        # t_start = time.time()
        # origin_val_size = len(X_val)
        # print(origin_val_size)
        # X_val_aug = np.zeros((origin_val_size*aug_times,splitsize,splitsize,1))
        # Y_val_aug = np.zeros((origin_val_size*aug_times,splitsize,splitsize,1))
        # for i in range(len(X_val)):
        #     for j in range(aug_times):
        #         aug = strong_aug(p=1)
        #         augmented = aug(image=X_train[i], mask=Y_train[i])
        #         X_val_aug[origin_val_size*j + i] = augmented['image']
        #         Y_val_aug[origin_val_size*j + i] = augmented['mask']
        # print("read images in {} sec".format(time.time()-t_start))

        # X_val_aug = X_val_aug.astype(np.float32)
        # Y_val_aug = Y_val_aug.astype(np.float32)
        # if len(X_val)==origin_val_size:
        #     X_val = np.append(X_val,X_val_aug, axis=0)
        # if len(Y_val)==origin_val_size:
        #     Y_val = np.append(Y_val, Y_val_aug, axis=0)
        # print("X_val after aug",X_val.shape)
        # print("Y_val after aug",Y_val.shape)
        # print("read images in {} sec".format(time.time()-t_start))
        # X_val = X_val.astype(np.float32)
        # Y_val = Y_val.astype(np.float32)
        # -----------------------
        print("X_train\n\n\n", X_train.shape)
        X_train = np.moveaxis(X_train, -1, 1)
        Y_train = np.moveaxis(Y_train, -1, 1)
        X_val = np.moveaxis(X_val, -1, 1)
        Y_val = np.moveaxis(Y_val, -1, 1)

        # print("X_train", X_train.shape)
        # print("X_val", X_val.shape)
        # print("Y_train", Y_train.shape)
        # print("Y_val", Y_val.shape)

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

        faults_dataset_train = faultsDataset(X_train, train=True, preprocessed_masks=Y_train)
        faults_dataset_val = faultsDataset(X_val, train=False, preprocessed_masks=Y_val)
        # faults_dataset_test = faultsDataset(X_test, train=False, preprocessed_masks=Y_test)

        train_loader = torch.utils.data.DataLoader(dataset=faults_dataset_train,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        val_loader = torch.utils.data.DataLoader(dataset=faults_dataset_val,
                                                 batch_size=batch_size,
                                                 shuffle=False)

        # test_loader = torch.utils.data.DataLoader(dataset=faults_dataset_test,
        #                                            batch_size=batch_size,
        #                                            shuffle=False)
        return train_loader, val_loader

def preprocess(image, scale=3):
    """
    Preprocess single image file
      (1) Read original image as YCbCr format (and grayscale as default)
      (2) Normalize
      (3) Apply image file with bicubic interpolation
    Args:
      path: file path of desired file
      input_: image applied bicubic interpolation (low-resolution)
      label_: image with original resolution (high-resolution)
    """
    # image = imread(path, is_grayscale=True)
    label_ = modcrop(image, scale)

    # # Must be normalized
    # image = image / 255.
    # label_ = label_ / 255.

    input_ = scipy.ndimage.interpolation.zoom(label_, (1. / scale), prefilter=False)
    input_ = scipy.ndimage.interpolation.zoom(input_, (scale / 1.), prefilter=False)

    return input_, label_

def modcrop(image, scale=3):
    """
    To scale down and up the original image, first thing to do is to have no remainder while scaling operation.

    We need to find modulo of height (and width) and scale factor.
    Then, subtract the modulo from height (and width) of original image size.
    There would be no remainder even after scaling operation.
    """
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image