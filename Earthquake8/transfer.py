import os
import sys
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im
import imageio

from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.filters import gaussian
from skimage import measure
from skimage import exposure
from sklearn.feature_extraction import image
from sklearn import svm
from sklearn.externals import joblib
from sklearn.utils import shuffle
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw


np.random.seed(1337)

seismic_name = "gsb"



# Gaussian smoothing with default std 1.5
# https://stackoverflow.com/a/45020378
def myGaussSmooth( data, std=1.5):
    return gaussian(data,sigma=std,truncate=2)
#Clipping
def clip(data,min=-6000,max=6000):
    data[data>max] = max
    data[data<min] = min
    return data


# Normalizes values of a matrix between -1 and 1
def myNormalization(data):
    max_val = np.max(data)
    min_val = np.min(data)

    return 2 * (data[:, :] - min_val) / (max_val - min_val) - 1





# Set configurations for format subvolume
data_file = 'data/gsb_crl_2800.csv'
data_file="/Users/shanyuhai/Downloads/data/gsb_inl_2011.csv"
# data_file='/Users/shanyuhai/Downloads/data/gsb_crl_2800.csv'
min_inl = 1565  # 1565
max_inl = 2531  # 2531
step_inl = 2
min_crl = 2800  # 2568
max_crl = 2800  # 3568
step_crl = 2
min_z = 1000
max_z = 1300
step_z = 4

# Define sections type
types = ['inline', 'crossline', 'z-slice', 'section_coords']
section_type = 'crossline'

output_dir = 'data/formatted/' + seismic_name + '/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read and format files
raw_data = np.genfromtxt(data_file, delimiter='\t')

if (section_type == "inline"):
    nb_crosslines = (max_crl - min_crl) / step_crl + 1

if (section_type == "crossline"):
    nb_crosslines = (max_inl - min_inl) / step_inl + 1

nb_sections = raw_data.shape[0] / nb_crosslines

for k in range(int(nb_sections)):
    # Output name
    if (section_type == "inline"):
        inl_nb = int(min_inl + k);
        name = 'inl' + str(inl_nb) + '_sc_formatted.csv'
    if (section_type == "crossline"):
        crl_nb = int(min_crl + k);
        name = 'crl' + str(crl_nb) + '_sc_formatted.csv'

    # Read
    section = raw_data[k * int(nb_crosslines): (k + 1) * int(nb_crosslines), :]
    section = np.transpose(section)
    section = np.fliplr(section)

    # Smooth
    section = myGaussSmooth(section)

    # Clip
    section = clip(section)

    # perc = np.percentile(section, [1,99])
    # section = exposure.rescale_intensity(section, in_range=(perc[0], perc[1]), out_range=(0,255))

    # Normalize between -1 and 1
    section = myNormalization(section)
    print(section.shape)

    # Write
    np.savetxt(output_dir + name, section, delimiter=" ")

    # Visualize
    plt.figure(figsize=(9, 4))
    # plt.title('Section')
    plt.axis('off')
    # ax = plt.gca()
    # im = plt.imshow(section, cmap="seismic", aspect='auto')
    im = plt.imshow(section)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(im, cax=cax)
    plt.show()
    print(section.shape)