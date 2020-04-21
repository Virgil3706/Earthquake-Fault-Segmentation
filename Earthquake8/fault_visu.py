from functions import *
import matplotlib
# 存储路径
save_path = 'saved/picture'
# label的第几张图
index= 20
import pandas as pd
# label路径
# a=np.load("/Users/shanyuhai/PycharmProjects/Earthquake/data/FYP_data/seis_sub_350IL_500t_1200XL.npy")
a=np.load("/Users/shanyuhai/Downloads/subGSBxl_t_il.npy")
# a=a[index]
# a=np.load("/Users/shanyuhai/Downloads/_gsb_inl_1791.npy",allow_pickle=True)

print(a.shape)
# print(a)

# a = pd.read_csv("/Users/shanyuhai/Downloads/gsb_inl_1791.csv",delimiter=' ',header = None)
# b=np.save("data/gsb_inl_1791.npy", a)
# a=np.load('data/gsb_inl_1791.npy',allow_pickle=True)


# b = np.load(os.path.join("/home/anyu/myproject/venv/an/pieces/HED/dropout/0.2/testGTs","0.npy"))
import cv2
import cmapy
# heatmap_img = cv2.applyColorMap((a * 255).astype(np.uint8), cmapy.cmap('jet_r'))
plt.figure(figsize=(9, 4))
plt.axis('off')
plt.imshow(a)
# plt.colorbar(shrink=0.5)

plt.show()
plt.savefig('{}/label_{}.png'.format(save_path,index))
