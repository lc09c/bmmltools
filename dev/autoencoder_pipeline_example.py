#
#
#
#
#

"""

"""

#################
#####   LIBRARIES
#################


import numpy as np
from bmmltools.core.data import Data
from bmmltools.core.tracer import Trace
from bmmltools.operations.feature import PatchTransform3D,DataStandardization,DimensionalReduction_PCA
from bmmltools.operations.clustering import Clusterer_HDBSCAN
from bmmltools.operations.io import Input

import tensorflow as tf
from tensorflow.keras.models import load_model


#################
#####   FUNCTIONS
#################


# custom patch transform function
def patch_transform_function(patch,model):

    # result = []
    # for z in range(patch.shape[0]):
    #
    #     features = model.predict(patch[z,:,:][None,:,:,None],verbose=0)
    #     result.append(features[0,:,:,:])
    #
    # return np.array(result).mean(axis=0,keepdims=True)

    # alternatively assuming a patch with 3 dimensions (to be tested)
    return model.predict(patch[:,:,:,None],verbose=0).mean(axis=0,keepdims=True)


############
#####   MAIN
############


### inputs

# input image
data_path = r'S:\2022-1-active\MaxR\test_segment'
data_code = '0243'

# encoder
model_path = r'autoencoder/models/encoder_7443.h5'


### Setup

# load data
data = Data()
data.link(working_folder=data_path,data_code=data_code)

# create/load trace
trace = Trace()
# trace.create(working_folder=r'S:\2022-1-active\curcuraci\bmmltools\mouse_bone\working_folder',group_name='segmenter')
trace.link(trace_folder=r'S:\2022-1-active\curcuraci\bmmltools\mouse_bone\working_folder\trace_5645',
           group_name='segmenter')

# load encoder for custom patch transform
encoder_model = load_model(model_path)
pt_func = lambda x: patch_transform_function(x,encoder_model)


### Model

# Uncomment lines below
Input(trace).i('1L_test').apply(data)
PatchTransform3D(trace,pt_func)\
    .io('input_dataset','post_pt3d_inference_dataset')\
    .apply(patch_shape=(30,200,200),stride=(30,100,100),vol_frac_th=0.1)
PatchTransform3D(trace,pt_func)\
    .io('input_dataset','post_pt3d_training_dataset')\
    .apply(patch_shape=(1,200,200),random_patches=True,n_random_patches=20000,vol_frac_th=0.2)
DimensionalReduction_PCA(trace)\
    .io(['post_pt3d_inference_dataset','post_pt3d_training_dataset'],'post_dm_pca_inference_dataset')\
    .apply(inference_key='transformed_patch',training_key='transformed_patch',save_model=False)
DataStandardization(trace).io('post_dm_pca_inference_dataset','post_ds_inference_dataset').apply(axis=(1,0))
Clusterer_HDBSCAN(trace)\
    .io(['post_ds_inference_dataset','post_pt3d_inference_dataset'],'raw_labels_dataset')\
    .apply(p=dict(min_cluster_size=40,prediction_data=True,metric='euclidean',min_samples=20),save_model=False)