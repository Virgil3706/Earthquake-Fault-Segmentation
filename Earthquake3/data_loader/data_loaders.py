from torchvision import datasets, transforms
from base import BaseDataLoader
from dataset.Dataset import faultsDataset
from functions import *
class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class EarthquakeDataLoader(BaseDataLoader):
    """

    """
    def __init__(self, data_dir, seismic_path,label_path,batch_size, shuffle=True, validation_split=0.0, num_workers=2,number_of_pictures=350, start=0,training=True):

        seismic = np.load(seismic_path)
        fault = np.load(label_path)
        # In[4]:
        # print(seismic.shape, fault.shape)
        # seismic = np.moveaxis(seismic[4:1807],-2,-1)
        print(seismic.shape, fault.shape)
        print(seismic.max(), seismic.min(), fault.max(), fault.min())
        # # reorder input data to same order IL, Z, XL

        # # In[5]:

        # seismic = (seismic-seismic.min(axis=(1,2), keepdims=True))/(seismic.max(axis=(1,2), keepdims=True)-seismic.min(axis=(1,2), keepdims=True))
        # print(seismic.shape)

        # # In[6]:

        IL, Z, XL = fault.shape

        # In[7]:

        best_model_fpath = 'hed_192_96_900200_seed.model'
        best_iou_threshold = 0.5
        epoches = 1
        patience = 20
        im_height = Z
        im_width = XL
        splitsize = 96
        stepsize = 48
        overlapsize = splitsize - stepsize
        pixelThre = int(0.03 * splitsize * splitsize)
        print(pixelThre)


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

        # In[17]:

        t_start = time.time()
        X = []
        Y = []
        print("start:",start)
        print("number_of_pictures: ",number_of_pictures)
        for i in range(start, number_of_pictures+start, 1):
            mask = fault[i]
            splits = split_Image(mask, True, top_pad, bottom_pad, left_pad, right_pad, splitsize, stepsize,
                                 vertical_splits_number, horizontal_splits_number)
            #     print(splits.shape)
            t = (splits.sum((1, 2)) < pixelThre)
            no_label_element_index = list(compress(range(len(t)), t))
            # get all the indexes of the no label pieces by adding elements in axis 2 and 3.
            splits = np.delete(splits, no_label_element_index, 0)  # delete element i along axis 0
            #     print("splits.shape", splits.shape)
            Y.extend(splits)

            img = seismic[i]
            splits = split_Image(img, True, top_pad, bottom_pad, left_pad, right_pad, splitsize, stepsize,
                                 vertical_splits_number, horizontal_splits_number)
            splits = np.delete(splits, no_label_element_index, 0)  # delete element i along axis 0
            #     print("splits.shape", splits.shape)
            X.extend(splits)
        #     break

        print(len(Y))
        print(len(X))
        print('X[0].shape', X[0].shape)
        print("read images in {} sec".format(time.time() - t_start))

        # In[20]:

        t_start = time.time()
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        print(X.shape)
        print(Y.shape)
        print("read images in {} sec".format(time.time() - t_start))

        # In[22]:

        if len(Y.shape) == 3:
            Y = np.expand_dims(Y, axis=-1)
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=-1)
        print(X.shape)
        print(Y.shape)

        X_train = X
        Y_train = Y

        X_train = np.moveaxis(X_train, -1, 1)
        Y_train = np.moveaxis(Y_train, -1, 1)
        print("X_train", X_train.shape)
        print("Y_train", Y_train.shape)

        # In[ ]:

        faults_dataset_train = faultsDataset(X_train, train=training, preprocessed_masks=Y_train)
        super().__init__(faults_dataset_train, batch_size, shuffle, validation_split, num_workers)


class EarthquakeDataLoader2(BaseDataLoader):
    """

    """
    def __init__(self, seismic_path,label_path,batch_size, shuffle=True, validation_split=0.0, num_workers=1,number_of_pictures=10,start=0,training=True):

        seismic = np.load(seismic_path)
        fault = np.load(label_path)
        # In[4]:
        # print(seismic.shape, fault.shape)
        # seismic = np.moveaxis(seismic[4:1807],-2,-1)
        print(seismic.shape, fault.shape)
        print(seismic.max(), seismic.min(), fault.max(), fault.min())
        # # reorder input data to same order IL, Z, XL

        # # In[5]:

        # seismic = (seismic-seismic.min(axis=(1,2), keepdims=True))/(seismic.max(axis=(1,2), keepdims=True)-seismic.min(axis=(1,2), keepdims=True))
        # print(seismic.shape)

        # # In[6]:

        IL, Z, XL = fault.shape

        # In[7]:

        best_model_fpath = 'hed_192_96_900200_seed.model'
        best_iou_threshold = 0.5
        epoches = 1
        patience = 20
        im_height = Z
        im_width = XL
        splitsize = 96
        stepsize = 48
        overlapsize = splitsize - stepsize
        pixelThre = int(0.03 * splitsize * splitsize)
        print(pixelThre)


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

        # In[17]:

        t_start = time.time()
        X = []
        Y = []
        print("start:",start)
        print("number_of_pictures: ",number_of_pictures)
        for i in range(start, number_of_pictures+start, 1):
            mask = fault[i]
            splits = split_Image(mask, True, top_pad, bottom_pad, left_pad, right_pad, splitsize, stepsize,
                                 vertical_splits_number, horizontal_splits_number)
            #     print(splits.shape)
            t = (splits.sum((1, 2)) < pixelThre)
            no_label_element_index = list(compress(range(len(t)), t))
            # get all the indexes of the no label pieces by adding elements in axis 2 and 3.
            splits = np.delete(splits, no_label_element_index, 0)  # delete element i along axis 0
            #     print("splits.shape", splits.shape)
            Y.extend(splits)

            img = seismic[i]
            splits = split_Image(img, True, top_pad, bottom_pad, left_pad, right_pad, splitsize, stepsize,
                                 vertical_splits_number, horizontal_splits_number)
            splits = np.delete(splits, no_label_element_index, 0)  # delete element i along axis 0
            #     print("splits.shape", splits.shape)
            X.extend(splits)
        #     break

        print(len(Y))
        print(len(X))
        print('X[0].shape', X[0].shape)
        print("read images in {} sec".format(time.time() - t_start))

        # In[20]:

        t_start = time.time()
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        print(X.shape)
        print(Y.shape)
        print("read images in {} sec".format(time.time() - t_start))

        # In[22]:

        if len(Y.shape) == 3:
            Y = np.expand_dims(Y, axis=-1)
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=-1)
        print(X.shape)
        print(Y.shape)



        X = np.moveaxis(X, -1, 1)
        Y = np.moveaxis(Y, -1, 1)
        print("X_train", X.shape)
        print("Y_train", Y.shape)

        # In[ ]:

        faults_dataset_train = faultsDataset(X, train=training, preprocessed_masks=Y)
        super().__init__(faults_dataset_train, batch_size, shuffle, validation_split, num_workers)

