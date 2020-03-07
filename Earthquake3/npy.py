
import numpy as np
test=np.load("/Users/shanyuhai/PycharmProjects/Earthquake/data/FYP_data/seis_sub_350IL_500t_1200XL.npy",encoding = "latin1")  #加载文件
doc = open('2.txt', 'a')  #打开一个存储文件，并依次写入
print(test, file=doc)  #将打印内容写入文件中
