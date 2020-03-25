from functions import *
import matplotlib; matplotlib.use('TkAgg')
# 存储路径
save_path = 'saved/picture/Unet/test'
# label的第几张图
index= 222

# label路径
a=np.load("/Users/shanyuhai/PycharmProjects/Earthquake/data/FYP_data/seis_sub_350IL_500t_1200XL.npy")
a=a[index]
# b = np.load(os.path.join("/home/anyu/myproject/venv/an/pieces/HED/dropout/0.2/testGTs","0.npy"))
import cv2
import cmapy
heatmap_img = cv2.applyColorMap((a * 255).astype(np.uint8), cmapy.cmap('jet_r'))
plt.figure(figsize=(10, 8))

plt.imshow(a)
# plt.colorbar(shrink=0.5)
plt.axis('off')
plt.show()
# plt.savefig('{}/label_{}.png'.format(save_path,index))
