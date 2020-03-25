import matplotlib;
import numpy;
from functions import *
def visu123(save_path,seismic_path,index,dimension=0):


    # matplotlib.use('TkAgg')
    # 存储路径
    # save_path = 'saved/picture/Unet/0309_182959'
    # label的第几张图
    # index=0
    # label路径
    # a = np.load("/Users/shanyuhai/PycharmProjects/Earthquake/data/FYP_data/fault_sub_350IL_500t_1200XL.npy")
    a=numpy.load(seismic_path)
    if dimension == 0:
        a = a[index]
    elif dimension == 1:
        a = a[:, index, :]
    elif dimension == 2:
        a = a[:, :, index]
        print(a.shape)
    # a = a[index]
    # b = np.load(os.path.join("/home/anyu/myproject/venv/an/pieces/HED/dropout/0.2/testGTs","0.npy"))
    import cv2
    import cmapy
    plt.figure(figsize=(10, 8))
    plt.imshow(a)
    # plt.colorbar(shrink=0.5)
    plt.axis('off')
    plt.show()
    plt.savefig('{}/123.png'.format(save_path, index))
    # return '{}/seismic_{}.png'.format(save_path, index)

visu123("saved","data/FYP_data/fault_sub_350IL_500t_1200XL.npy",51,2)