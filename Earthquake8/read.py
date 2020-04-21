
import numpy as np
input_data = np.load('data/FYP_data/seis_sub_350IL_500t_1200XL.npy')
print(input_data.shape)
data = input_data.reshape(1,-1)
print(data.shape)
print(data)
np.savetxt(r"seismic.txt",data,delimiter=',')

