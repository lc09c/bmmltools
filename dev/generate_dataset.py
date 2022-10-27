# Title: 'generate_dataset.py'
# Author: Curcuraci L.
# Date: 20/10/2022
#
# Scope: produce a dataset of 2d images extracted from the input data. These images form the dataset used to train an
# autoencoder

"""

"""

#################
#####   LIBRARIES
#################


import numpy as np
import matplotlib.pyplot as plt
import os

from bmmltools.core.data import Data
from bmmltools.utils.basic import manage_path


############
#####   MAIN
############


### inputs

# input image
data_path = r'S:\2022-1-active\MaxR\test_segment'
data_code = '0243'

# dataset setting
vol_frac_th = 0.2
sampling_patch_shape = (200,200)
n_dataset_points = 20000
dataset_saving_path = 'autoencoder\dataset'


### setup

dataset_saving_path = manage_path(dataset_saving_path)+os.sep+'dataset.npy'


### load input 3d binary data

data = Data()
data.link(working_folder = data_path,data_code = data_code)
data.use_dataset('1L_test')
img_shape = data.shape


### sampling
Z_bounds = [0,img_shape[0]-1]
Y_bounds = [sampling_patch_shape[0],img_shape[1]-sampling_patch_shape[0]-1]
X_bounds = [sampling_patch_shape[1],img_shape[2]-sampling_patch_shape[1]-1]
dataset = []
N = 0
while True:

    print('Samples: {}/{}'.format(N+1,n_dataset_points),end='\r')
    z = np.random.randint(Z_bounds[0],Z_bounds[1])
    y = np.random.randint(Y_bounds[0],Y_bounds[1])
    x = np.random.randint(X_bounds[0],X_bounds[1])
    patch = data[z,y:y+sampling_patch_shape[0],x:x+sampling_patch_shape[1]]
    if np.sum(patch)> np.prod(sampling_patch_shape)*vol_frac_th:

        dataset.append( patch )
        N += 1

    if N == n_dataset_points:

        print('Done!')
        break


###
dataset = np.array(dataset)
np.save(dataset_saving_path,dataset)