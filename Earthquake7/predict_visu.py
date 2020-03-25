from functions import *
import matplotlib; matplotlib.use('TkAgg')
save_path = 'saved/picture/Unet/0309_134059'

index= 2

a = np.load(os.path.join(save_path,str(index)+'.npy'))
a=np.load("/Users/shanyuhai/PycharmProjects/Earthquake/data/FYP_data/fault_sub_350IL_500t_1200XL.npy")
a=a[0]
# b = np.load(os.path.join("/home/anyu/myproject/venv/an/pieces/HED/dropout/0.2/testGTs","0.npy"))

import cv2
import cmapy

heatmap_img = cv2.applyColorMap((a * 255).astype(np.uint8), cmapy.cmap('jet_r'))
plt.figure(figsize=(10, 8))
plt.imshow(heatmap_img)
plt.colorbar(shrink=0.5)
# plt.axis('off')
# plt.show()
plt.savefig('{}/{}.png'.format(save_path,index))
