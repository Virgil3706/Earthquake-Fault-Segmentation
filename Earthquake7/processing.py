from PIL import Image


def blendTwoImages(img1='saved/picture/Unet/0314_202109/0314_202109_100.png',img2='saved/picture/Unet/0314_202109/label_100.png',save_path='saved/a.png'):
    '''
    两幅图片叠加成为一张图片【使用Image.blend()接口】
    '''
    img1=Image.open(img1)
    img1=img1.convert('RGBA')
    img2=Image.open(img2)
    img2=img2.convert('RGBA')
    new_im=img1.resize(img2.size,Image.ANTIALIAS)
    img=Image.blend(new_im,img2,0.3)
    print(img)
    img.save(save_path)
    return save_path


a=blendTwoImages('/Users/shanyuhai/PycharmProjects/Earthquake7/saved/picture/DeepLab/0314_222912/label_0.png','/Users/shanyuhai/PycharmProjects/Earthquake7/saved/picture/DeepLab/0314_222912/seismic_0.png')
b=blendTwoImages(a,'/Users/shanyuhai/PycharmProjects/Earthquake7/saved/picture/DeepLab/0314_222912/0314_222912_0.png')
#
# ————————————————
# 版权声明：本文为CSDN博主「Together_CZ」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/Together_CZ/article/details/91809362