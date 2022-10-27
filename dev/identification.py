# Title: 'identification.py'
# Author: Curcuraci L.
# Date: 20/10/2022
#
# Scope: Feature extraction+clustering+reconstruction in patch space (with stride).

"""

"""


#################
#####   LIBRARIES
#################


import numpy as np
import matplotlib.pyplot as plt
import os
import hdbscan

from sklearn.decomposition import PCA

from bmmltools.utils.basic import manage_path
from bmmltools.core.data import Data

import tensorflow as tf

from tensorflow.keras.models import load_model


############
#####   MAIN
############


### Inputs

# paths
data_path = r'S:\2022-1-active\MaxR\test_segment'
dataset_path = r'dev/dataset/dataset.npy'
model_path = r'dev/models/encoder_7443.h5'
results_saving_path = r'dev/results'

# data setting
data_code = '0243'
dataset_to_use_name = '1L_test'

# patch space rendering setting
patch_shape = (200,200)
h = 100                         ## Chose a stride which is 2^n times smaller that the patch shape e.g patch shape = (300,300) -> h = 150
patch_vol_th = 0.1

# slice to labels
# slice_list= list(range(0,130,30))
slice_list = [50]

### Setup

# load data and model
dataset = np.load(dataset_path)
encoder = load_model(model_path)

# load data
data = Data()
data.link(working_folder=data_path,data_code=data_code)
data.use_dataset(dataset_to_use_name)

# compute necessary parameters
stride_factor = h/np.array(patch_shape)


### Extract features (train)

ls_repr = encoder.predict(dataset[:,:,:,None])


### Dimensional reduction (train)

ls_repr = ls_repr.reshape((ls_repr.shape[0],-1))    # flatten code representation
reducer = PCA(n_components=10)
dim_red_repr = reducer.fit_transform(ls_repr)


### Extract feature and project with trained PCA

coords = []
repr = []
for n,s in enumerate(slice_list):

    img = data[s,:,:]
    image_shape = img.shape
    patch_space_shape = tuple(np.array(image_shape)//np.array(patch_shape))
    for y in np.arange(1,image_shape[0]//patch_shape[0],stride_factor[0]):

        for x in np.arange(1,image_shape[1]//patch_shape[1],stride_factor[1]):

            print('slice {} | y: {}/{} | x: {}/{}'.format(s,2*y+1,2*image_shape[0]//(patch_shape[0]),
                                                            2*x+1,2*image_shape[1]//(patch_shape[1])))
            patch = img[int(y*patch_shape[0])-h:int((y+1)*patch_shape[0])-h,
                        int(x*patch_shape[1])-h:int((x+1)*patch_shape[1])-h]
            if np.sum(patch) > patch_vol_th*np.prod(patch_shape):

                lr_patch = encoder.predict(patch[None,:,:,None]).reshape((1,-1))
                dim_red_patch = reducer.transform(lr_patch)
                coords.append([n,int(y/stride_factor[0]),int(x/stride_factor[1])])
                repr.append(list(dim_red_patch[0,:]))

coords = np.array(coords)
repr = np.array(repr)


### Data standardization

repr_N = (repr-np.mean(repr,axis=1,keepdims=True))/np.std(repr,axis=1,keepdims=True)
repr_N = (repr_N-np.mean(repr_N,axis=0,keepdims=True))/np.std(repr_N,axis=0,keepdims=True)


### Clustering

cls = hdbscan.HDBSCAN(min_cluster_size=50,min_samples=50,metric='euclidean',prediction_data=True)
cls.fit(repr_N)
labels = np.argmax(hdbscan.all_points_membership_vectors(cls),axis=1)
print('N clusters found: {}'.format(len(list(set(labels)))))


### See result

# select the slice to label (slice should be present in the slice list)
if len(slice_list) > 1:

    print('Choose a slice among these here: ',slice_list)
    sel_slice = int(input('Insert slice number then press enter '))
    z_pos = slice_list.index(sel_slice)

else:

    z_pos = 0
    sel_slice = slice_list[0]

# fill the canvas
canvas = np.zeros((int(patch_space_shape[0]/stride_factor[0]), int(patch_space_shape[1]/stride_factor[1])))
for y in np.arange(1,image_shape[0]//patch_shape[0],stride_factor[0]):

    for x in np.arange(1,image_shape[1]//patch_shape[1],stride_factor[1]):

        filt = np.logical_and( coords[:,0] == z_pos, np.logical_and( coords[:,1] == int(y/stride_factor[0]) ,
                                                                         coords[:,2] == int(x/stride_factor[1])))
        if np.sum(filt) > 0:

            idx = np.where(filt == True)
            canvas[int(y/stride_factor[0]),int(x/stride_factor[1])] = labels[idx]+1


plt.figure()
plt.title('Original image: slice {}'.format(sel_slice))
plt.imshow(data[sel_slice,:,:])
plt.show()

plt.figure()
plt.title('Labeled image: slice {}'.format(sel_slice))
plt.imshow(canvas)
plt.show()

### Save result in patch space

patch_space_labelled = np.zeros((len(slice_list),
                                 int(patch_space_shape[0]/stride_factor[0]),
                                 int(patch_space_shape[1]/stride_factor[1])))
for n,zyx in enumerate(coords):

    z = zyx[0]
    y = zyx[1]
    x = zyx[2]
    patch_space_labelled[z,y,x] = labels[n]+1                      # label+1 is what is saved!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

results_saving_path = manage_path(results_saving_path)
np.save(results_saving_path+os.sep+'labelled_patch_space.npy',patch_space_labelled)