# Title: 'feature.py'
# Author: Curcuraci L.
# Date: 17/10/2022
#
# Scope: Collect all the feature extraction operations in bmmltools

"""
Operations used to extract features from some dataset on a trace.
"""


#################
#####   LIBRARIES
#################


import numpy as np
import json
import os
import glob
import h5py
import pandas as pd
import inspect
import functools

from joblib import dump, load
from sklearn.decomposition import PCA,NMF

from bmmltools.utils.basic import get_capitals,manage_path,ParametersTracker
from bmmltools.utils.io_utils import write_rrn
from bmmltools.features.dft import periodic_smooth_decomposition_dft3,dft


###############
#####   CLASSES
###############


class Binarizer:
    """
    Operation used to binarize a dataset.
    """

    __name__ = 'Binarizer'
    def __init__(self,trace):
        """
        Initialize the operation. The Binarizer operation is used to binarize some dataset. The output dataset will take
        value in {0,1}. Binarization is performed by simple thresholding.

        :param trace: (bmmltools.core.tracer.Trace object) trace on which the operation act.
        """
        self.trace = trace
        self.output_dataset = ['post_'+get_capitals(Binarizer.__name__)+'_dataset']
        self.pt = ParametersTracker(self)

    def i(self,x):
        """
        Specify operation inputs.

        :param x: (str, list[str]) name of the input dataset used in this operation. If more than one input is required
                  a list with the name of the input dataset need to be specified following the specific order described
                  in the bmmltools documentation.
        """
        if type(x) is str:

            self.input_dataset = [x]

        elif type(x) in [tuple,list]:

            self.input_dataset = x

        else:

            raise TypeError

        return self

    def o(self,x):
        """
        Specify operation outputs.

        :param x: (str, list[str]) optional, name or list of names of the operations output. If more than one output
                  is present, in the list one needs to specify all the names of the output. If nothing is specified,
                  the default name is used. The default name is

                                'post_[Capital Letters + Numbers in the operation class name]_dataset'.

                  If more than one output is present, some additional words or numbers are added to the default name
                  in order to differentiate the different outputs.
        """
        if type(x) is str:

            self.output_dataset = [x]

        elif type(x) in [tuple, list]:

            self.output_dataset = x

        else:

            raise TypeError

        return self

    def io(self,x,y):
        """
        Specify operation inputs and outputs.

        :param x: (str, list[str]) name of the input dataset used in this operation. If more than one input is required
                  a list with the name of the input dataset need to be specified following the specific order described
                  in the bmmltools documentation.
        :param y: (str, list[str]) optional, name or list of names of the operations output. If more than one output
                  is present, in the list one needs to specify all the names of the output. If nothing is specified,
                  the default name is used. The default name is

                                'post_[Capital Letters + Numbers in the operation class name]_dataset'.

                  If more than one output is present, some additional words or numbers are added to the default name
                  in order to differentiate the different outputs.
        """
        self.i(x)
        self.o(y)
        return self

    def apply(self,threshold=0.5):
        """
        Apply the operation to the trace. The binarization is done by thresholding.

        :param threshold: (float) threshold above witch the binarization result will be 1 and below which will be 0.
        """
        with self.pt:

            self.threshold = threshold

        print('>----------------<')
        print('Binarization')
        binarized = self.trace.__getattribute__(self.input_dataset[0])>self.threshold
        self.trace.__setattr__(self.output_dataset[0],binarized.astype(np.byte))
        print('>----------------<',end='\r')

        self._update_graph_dict()
        self._update_params_dict()
        if self.trace.enable_operations_tracking:

            self.trace.add_operation(Binarizer.__name__,self,self.pt.get_current_tracked_params())

        return self.output_dataset

    def read(self,name=None,save_as='npy'):
        """
        Read operation results in the hdf5 file. For this plugin, the results are organized in the form of a pandas
        dataframe.

        :param save_as: (str) optional, format used to save the result. It can be 'npy' of 'rrn'.
        :param name: (str) optional, name of the result file.
        """
        tmp_array = self.trace.__getattribute__(self.output_dataset[0])
        trace_pos = self.trace.get_trace_pos()
        path_to_readings_folder = manage_path(
            self.trace.trace_readings_path()+os.sep+Binarizer.__name__+'_{}'.format(trace_pos))
        if name is None:

            name = self.output_dataset[0]

        if save_as == 'npy':

            np.save(path_to_readings_folder+os.sep+name+'.npy',tmp_array)

        elif save_as == 'rrn':

            write_rrn(tmp_array,path_to_readings_folder,name)

        else:

            pass

    def _update_graph_dict(self):
        """
        Update the graph dictionary in the trace json adding the inputs and outputs of this operation.
        """
        if self.trace.enable_trace_graph == True:

            tmp_dict = self.trace.trace_graph_dict
            link_list = []
            for i_name in self.input_dataset:

                for o_name in self.output_dataset:

                    link_list.append((i_name,o_name))

            key = len(tmp_dict)
            if tmp_dict[key-1]['node'] == self.output_dataset:

                key = key-1

            tmp_dict.update({key:{'node': self.output_dataset,
                                  'link': list(set(link_list)),
                                  'edge': Binarizer.__name__}})
            self.trace.__standard_setattr__('trace_graph_dict',tmp_dict)
            with open(self.trace.json_trace_path, 'r') as jfile:

                previous_content = json.load(jfile)

            previous_content[self.trace.group_name]['trace_graph'] = tmp_dict
            with open(self.trace.json_trace_path, 'w') as jfile:

                json.dump(previous_content, jfile, indent=4)

    def _update_params_dict(self):
        """
        Update the parameter dictionary in the trace json adding parameters of this operation.
        """
        with open(self.trace.json_trace_path, 'r') as jfile:

            previous_content = json.load(jfile)

        params_dict = previous_content[self.trace.group_name]['ops_parameters']
        key = len(params_dict)
        if key > 0 and \
                previous_content[self.trace.group_name]['trace_graph'][str(key-1)]['node'] == self.output_dataset:

            key = key-1
            del params_dict[str(key)]

        params_dict.update({key: {'op': Binarizer.__name__,
                                  'p': self.pt.get_current_tracked_params()}})
        previous_content[self.trace.group_name]['ops_parameters'] = params_dict
        with open(self.trace.json_trace_path, 'w') as jfile:

            json.dump(previous_content, jfile, indent=4)

class PatchTransform3D:
    """
    Operation used to apply a custom transformation patch-wise on some input dataset.
    """

    __name__ = 'PatchTransform3D'
    def __init__(self,trace,patch_transform_function):
        """
        Initialize the operation. The PatchTransform3D operation is used to apply a transformation function specified
        by the user patch-wise on some input dataset.

        :param trace: (bmmltools.core.tracer.Trace object) trace on which the operation act.
        :param patch_transform_function: (function) function used to extract information from the patch. It has to be a
                                         single input function. the only input have to be a patch of the dataset on
                                         which this operation is applied.
        """
        self.trace = trace
        self.output_dataset = ['post_'+get_capitals(PatchTransform3D.__name__)+'_dataset']
        self.pt = ParametersTracker(self)
        self._patch_transform = patch_transform_function

    def i(self,x):
        """
        Specify operation inputs.

        :param x: (str, list[str]) name of the input dataset used in this operation. If more than one input is required
                  a list with the name of the input dataset need to be specified following the specific order described
                  in the bmmltools documentation.
        """
        if type(x) is str:

            self.input_dataset = [x]

        elif type(x) in [tuple,list]:

            self.input_dataset = x

        else:

            raise TypeError

        return self

    def o(self,x):
        """
        Specify operation outputs.

        :param x: (str, list[str]) optional, name or list of names of the operations output. If more than one output
                  is present, in the list one needs to specify all the names of the output. If nothing is specified,
                  the default name is used. The default name is

                                'post_[Capital Letters + Numbers in the operation class name]_dataset'.

                  If more than one output is present, some additional words or numbers are added to the default name
                  in order to differentiate the different outputs.
        """
        if type(x) is str:

            self.output_dataset = [x]

        elif type(x) in [tuple, list]:

            self.output_dataset = x

        else:

            raise TypeError

        return self

    def io(self,x,y):
        """
        Specify operation inputs and outputs.

        :param x: (str, list[str]) name of the input dataset used in this operation. If more than one input is required
                  a list with the name of the input dataset need to be specified following the specific order described
                  in the bmmltools documentation.
        :param y: (str, list[str]) optional, name or list of names of the operations output. If more than one output
                  is present, in the list one needs to specify all the names of the output. If nothing is specified,
                  the default name is used. The default name is

                                'post_[Capital Letters + Numbers in the operation class name]_dataset'.

                  If more than one output is present, some additional words or numbers are added to the default name
                  in order to differentiate the different outputs.
        """
        self.i(x)
        self.o(y)
        return self

    def apply(self,patch_shape,stride=None,vol_frac_th = 0.05,transform_name = None,random_patches=False,
              n_random_patches=4000):
        """
        Apply the operation to the trace. The function 'transform_function' specified during the operation
        initialization is applied to various patches of the input data. The application can be done:

            * on the ZYX-grid, where the transformation for all the patches obtained dividing the sample into an regular
            ZYX grid which is constructed from the patch chosen;

            * at random, where patches are take in random position of the sample.

        :param patch_shape: (tuple[int]) shape of the input patch used in this operation.
        :param stride: (None or tuple[int]) optional, stride along each dimension (used when the transformation is
                       taken along a ZYX-grid).
        :param vol_frac_th: (float between 0 and 1) fraction of the patch that need to be occupied by 1s in order to
                            consider a patch suitable for the transformation (i.e. when there are too many zeros the
                            patch is disregarded).
        :param transform_name: (str) optional, name given to the transformation output (one of the two keys of the
                               output dictionary of this operation)
        :param random_patches: (bool) optional, if True the patch transform is applied in random points of the input
                               dataset, otherwise an ZYX-grid is used.
        :param n_random_patches: (int) optional, number of random patches taken (this variable is considered only when
                                 random patches are used).
        """
        #
        with self.pt:

            self.patch_shape = patch_shape
            self.stride = stride
            self.transform_name = transform_name
            self.vol_frac_th = vol_frac_th
            self.random_patches = random_patches
            self.n_random_patches = n_random_patches

        #
        if self.transform_name is None:

            self.transform_name = 'transformed_patch'

        with h5py.File(self.trace.hdf5_trace_path, 'r') as file:

            self.data_shape = file[self.trace._hdf5_inner_path+self.input_dataset[0]].shape

        self.transformation_output_shape = self._patch_transform(np.zeros(self.patch_shape)).shape
        self.patch_space_shape = tuple(np.array(self.data_shape) // np.array(self.patch_shape))
        if self.stride == None:

            self.stride = self.patch_shape

        print('>----------------<')
        print('Patch Transform 3d')
        print('({})'.format('using random patches' if self.random_patches else 'using zyx-grid'))
        if self.random_patches:

            self._random_patches_transform(self.transform_name)

        else:

            self._on_grid_patches_transform(self.transform_name)

        print('>----------------<',end='\r')
        self._update_graph_dict()
        self._update_params_dict()
        if self.trace.enable_operations_tracking:

            self.trace.add_operation(PatchTransform3D.__name__,self,self.pt.get_current_tracked_params())

        return self.output_dataset

    def _on_grid_patches_transform(self,name):
        """
        Compute the patch transform using a regular ZYX-grid.

        :param name: (str) name of the dataset (dictionary key in this case) where the transformed patches are saved.
        """
        # get coordinates
        patch_space_dz = self.stride[0]/self.patch_shape[0]
        z_start = patch_space_dz * int(self.stride[0]<self.patch_shape[0])
        patch_space_dy = self.stride[1]/self.patch_shape[1]
        y_start = patch_space_dy * int(self.stride[1]<self.patch_shape[1])
        patch_space_dx = self.stride[2]/self.patch_shape[2]
        x_start = patch_space_dx * int(self.stride[2]<self.patch_shape[2])
        Zs, Ys, Xs = np.meshgrid(np.arange(z_start,self.patch_space_shape[0],patch_space_dz),
                                 np.arange(y_start,self.patch_space_shape[1],patch_space_dy),
                                 np.arange(x_start,self.patch_space_shape[2],patch_space_dx))
        ZYXs = np.vstack([Zs.flatten(), Ys.flatten(), Xs.flatten()]).transpose()
        n_patches = len(ZYXs)

        # get necessary trace attributes before the hdf5 file is open
        hdf5_inner_path = self.trace._hdf5_inner_path
        group_name = self.trace.group_name
        traced_dict = self.trace.dict_variables_traced_on_hd
        traced_pandas = self.trace.pandas_variables_traced_on_hd
        traced_numpy = self.trace.numpy_variables_traced_on_hd

        #
        with h5py.File(self.trace.hdf5_trace_path,'a') as file:

            # delete dataset with the same name of the output if present
            try:

                del file[self.trace._hdf5_inner_path+self.output_dataset[0]]
                traced_dict = [elem for elem in traced_dict
                               if '{}.{}'.format(group_name,self.output_dataset[0]) not in elem]
                traced_numpy = [elem for elem in traced_numpy
                                if '{}.{}'.format(group_name,self.output_dataset[0]) not in elem]
                traced_pandas = [elem for elem in traced_pandas
                                 if '{}.{}'.format(group_name,self.output_dataset[0]) not in elem]

            except:

                pass

            group_db = file.create_group(name=hdf5_inner_path + self.output_dataset[0])
            tmp_db = group_db.create_dataset('{}_tmp'.format(name),
                                             shape=(n_patches,)+self.transformation_output_shape,
                                             dtype=np.float64,
                                             chunks=(1,)+self.transformation_output_shape)
            in_db = file[hdf5_inner_path+self.input_dataset[0]]
            nonnull_Zs = []
            nonnull_Ys = []
            nonnull_Xs = []
            m = 0
            for n in range(n_patches):

                # select patch
                z,y,x = ZYXs[n]
                zmin = int(z*self.patch_shape[0])
                zmax = zmin+self.patch_shape[0]
                ymin = int(y*self.patch_shape[1])
                ymax = ymin+self.patch_shape[1]
                xmin = int(x*self.patch_shape[2])
                xmax = xmin+self.patch_shape[2]
                patch = in_db[zmin:zmax,ymin:ymax,xmin:xmax]
                if np.sum(patch) > self.vol_frac_th*np.prod(self.patch_shape):

                    # compute transformed patch and coordinates in the output patch shape
                    tmp_db[m,:,:,:] = self._patch_transform(patch)
                    nonnull_Zs.append(z)
                    nonnull_Ys.append(y)
                    nonnull_Xs.append(x)
                    m += 1

                print('patch {}/{}'.format(n+1,n_patches), end='\r')

            group_db.create_dataset(name,data=tmp_db[:len(nonnull_Zs),:,:,:],
                                         dtype=np.float64,
                                         chunks=(1,) + self.transformation_output_shape)
            del group_db['{}_tmp'.format(name)]

            # update trace attributes
            traced_dict.append('{}.{}'.format(group_name, self.output_dataset[0]))
            file[hdf5_inner_path].attrs['dict_variables_traced_on_hd'] = list(set(traced_dict))

            traced_numpy.append('{}.{}.{}'.format(group_name, self.output_dataset[0],name))
            file[hdf5_inner_path].attrs['numpy_variables_traced_on_hd'] = \
                list(set([el for el in traced_numpy if group_name in el]))

        patch_space_coords_df = pd.DataFrame({'Z': nonnull_Zs,
                                              'Y': nonnull_Ys,
                                              'X': nonnull_Xs})
        patch_space_coords_df.to_hdf(self.trace.hdf5_trace_path,
                                     hdf5_inner_path+self.output_dataset[0]+'/patch_space_coordinates',
                                     mode='a', format='t', data_columns=True)
        with h5py.File(self.trace.hdf5_trace_path, 'a') as file:

            traced_pandas.append('{}.{}.{}'.format(group_name,self.output_dataset[0],'patch_space_coordinates'))
            file[hdf5_inner_path].attrs['pandas_variables_traced_on_hd'] = \
                list(set([el for el in traced_pandas if group_name in el]))

        self.trace.numpy_variables_traced_on_hd = list(set(traced_numpy))
        self.trace.pandas_variables_traced_on_hd = list(set(traced_pandas))
        self.trace.dict_variables_traced_on_hd = list(set(traced_dict))

    def _random_patches_transform(self,name):
        """
        Compute the patch transform using random patches.

        :param name: (str) name of the dataset (dictionary key in this case) where the transformed patches are saved.
        """
        # set seed if present
        if hasattr(self.trace,'seed'):

            np.random.seed(self.trace.seed)

        # get necessary trace attributes before the hdf5 file is open
        hdf5_inner_path = self.trace._hdf5_inner_path
        group_name = self.trace.group_name
        traced_dict = self.trace.dict_variables_traced_on_hd
        traced_pandas = self.trace.pandas_variables_traced_on_hd
        traced_numpy = self.trace.numpy_variables_traced_on_hd

        #
        with h5py.File(self.trace.hdf5_trace_path,'a') as file:

            # try:

            if self.output_dataset[0] in file[hdf5_inner_path].keys():

                del file[self.trace._hdf5_inner_path+self.output_dataset[0]]
                traced_dict = [elem for elem in traced_dict
                               if '{}.{}'.format(group_name,self.output_dataset[0]) not in elem]
                traced_numpy = [elem for elem in traced_numpy
                                if '{}.{}'.format(group_name,self.output_dataset[0]) not in elem]
                traced_pandas = [elem for elem in traced_pandas
                                 if '{}.{}'.format(group_name,self.output_dataset[0]) not in elem]

            #
            group_db = file.create_group(name=hdf5_inner_path + self.output_dataset[0])
            db = group_db.create_dataset('{}'.format(name),
                                    shape=(self.n_random_patches,)+self.transformation_output_shape,
                                    dtype=np.float64,
                                    chunks=(1,)+self.transformation_output_shape)
            in_db = file[hdf5_inner_path+self.input_dataset[0]]
            Zs = []
            Ys = []
            Xs = []
            while True:

                zmin = np.random.randint(self.patch_shape[0],self.data_shape[0]-self.patch_shape[0])
                zmax = zmin+self.patch_shape[0]
                ymin = np.random.randint(self.patch_shape[1],self.data_shape[1]-self.patch_shape[1])
                ymax = ymin+self.patch_shape[1]
                xmin = np.random.randint(self.patch_shape[2],self.data_shape[2]-self.patch_shape[2])
                xmax = xmin+self.patch_shape[2]
                patch = in_db[zmin:zmax,ymin:ymax,xmin:xmax]
                if np.sum(patch) > self.vol_frac_th*np.prod(self.patch_shape):

                    # compute transformed patch
                    db[len(Zs),:,:,:] = self._patch_transform(patch)
                    Zs.append(zmin/self.patch_shape[0])
                    Ys.append(ymin/self.patch_shape[1])
                    Xs.append(xmin/self.patch_shape[2])

                    print('patch: {}/{}'.format(len(Zs),self.n_random_patches),end='\r')
                    if len(Zs) == self.n_random_patches:

                        break

            # update trace attributes
            traced_dict.append('{}.{}'.format(group_name, self.output_dataset[0]))
            file[hdf5_inner_path].attrs['dict_variables_traced_on_hd'] = list(set(traced_dict))

            traced_numpy.append('{}.{}.{}'.format(group_name, self.output_dataset[0],name))
            file[hdf5_inner_path].attrs['numpy_variables_traced_on_hd'] = \
                list(set([el for el in traced_numpy if group_name in el]))

        patch_space_coords_df = pd.DataFrame({'Z': Zs,'Y': Ys,'X': Xs})
        patch_space_coords_df.to_hdf(self.trace.hdf5_trace_path,
                                     hdf5_inner_path + self.output_dataset[0] + '/patch_space_coordinates',
                                     mode='a', format='t', data_columns=True)
        with h5py.File(self.trace.hdf5_trace_path, 'a') as file:

            traced_pandas.append(
                '{}.{}.{}'.format(group_name, self.output_dataset[0], 'patch_space_coordinates'))
            file[hdf5_inner_path].attrs['pandas_variables_traced_on_hd'] = \
                list(set([el for el in traced_pandas if group_name in el]))

        self.trace.numpy_variables_traced_on_hd = list(set(traced_numpy))
        self.trace.pandas_variables_traced_on_hd = list(set(traced_pandas))
        self.trace.dict_variables_traced_on_hd = list(set(traced_dict))

    def read(self):
        """
        Read operation results in the hdf5 file. For this plugin, the results are organized in the form of a pandas
        dataframe (patch coordinates) and numpy array (transformation result). The first is saved as csv file, while the
        second as npy file.
        """
        trace_pos = self.trace.get_trace_pos()
        path_to_readings_folder = manage_path(
            self.trace.trace_readings_path() + os.sep + PatchTransform3D.__name__+'_{}'.format(trace_pos))
        res = self.trace.__getattribute__(self.output_dataset[0])
        for key in res.keys():

            if type(res[key]) == pd.DataFrame:

                res[key].to_csv(path_to_readings_folder + os.sep + key + '.csv')

            else:

                np.save(path_to_readings_folder + os.sep + key + '.npy', res[key])

    def _update_graph_dict(self):
        """
        Update the graph dictionary in the trace json adding the inputs and outputs of this operation.
        """
        if self.trace.enable_trace_graph == True:

            tmp_dict = self.trace.trace_graph_dict
            link_list = []
            for i_name in self.input_dataset:

                for o_name in self.output_dataset:

                    link_list.append((i_name,o_name))

            key = len(tmp_dict)
            if tmp_dict[key-1]['node'] == self.output_dataset:

                key = key-1

            tmp_dict.update({key:{'node': self.output_dataset,
                                   'link': list(set(link_list)),
                                   'edge': PatchTransform3D.__name__}})
            self.trace.__standard_setattr__('trace_graph_dict',tmp_dict)
            with open(self.trace.json_trace_path, 'r') as jfile:

                previous_content = json.load(jfile)

            previous_content[self.trace.group_name]['trace_graph'] = tmp_dict
            with open(self.trace.json_trace_path, 'w') as jfile:

                json.dump(previous_content, jfile, indent=4)

    def _update_params_dict(self):
        """
        Update the parameter dictionary in the trace json adding parameters of this operation.
        """
        with open(self.trace.json_trace_path, 'r') as jfile:

            previous_content = json.load(jfile)

        params_dict = previous_content[self.trace.group_name]['ops_parameters']
        key = len(params_dict)
        if key > 0 and \
                previous_content[self.trace.group_name]['trace_graph'][str(key-1)]['node'] == self.output_dataset:

            key = key-1
            del params_dict[str(key)]

        params_dict.update({key: {'op': PatchTransform3D.__name__,
                                  'p': self.pt.get_current_tracked_params()}})
        previous_content[self.trace.group_name]['ops_parameters'] = params_dict
        with open(self.trace.json_trace_path, 'w') as jfile:

            json.dump(previous_content, jfile, indent=4)

class PatchDiscreteFourierTransform3D:
    """
    Operation used to apply a  3d Discrete Fourier transformation patch-wise on some input dataset.
    """

    __name__ = 'PatchDiscreteFourierTransform3D'
    def __init__(self,trace):
        """
        Initialize the operation. The PatchDiscreteFourierTransform3D operation is used to apply a 3d Discrete Fourier
        transformation patch-wise on some input dataset.

        :param trace: (bmmltools.core.tracer.Trace object) trace on which the operation act.
        """
        self.trace = trace
        self.output_dataset = ['post_'+get_capitals(PatchDiscreteFourierTransform3D.__name__)+'_dataset']
        self.pt = ParametersTracker(self)

    def i(self,x):
        """
        Specify operation inputs.

        :param x: (str, list[str]) name of the input dataset used in this operation. If more than one input is required
                  a list with the name of the input dataset need to be specified following the specific order described
                  in the bmmltools documentation.
        """
        if type(x) is str:

            self.input_dataset = [x]

        elif type(x) in [tuple,list]:

            self.input_dataset = x

        else:

            raise TypeError

        return self

    def o(self,x):
        """
        Specify operation outputs.

        :param x: (str, list[str]) optional, name or list of names of the operations output. If more than one output
                  is present, in the list one needs to specify all the names of the output. If nothing is specified,
                  the default name is used. The default name is

                                'post_[Capital Letters + Numbers in the operation class name]_dataset'.

                  If more than one output is present, some additional words or numbers are added to the default name
                  in order to differentiate the different outputs.
        """
        if type(x) is str:

            self.output_dataset = [x]

        elif type(x) in [tuple, list]:

            self.output_dataset = x

        else:

            raise TypeError

        return self

    def io(self,x,y):
        """
        Specify operation inputs and outputs.

        :param x: (str, list[str]) name of the input dataset used in this operation. If more than one input is required
                  a list with the name of the input dataset need to be specified following the specific order described
                  in the bmmltools documentation.
        :param y: (str, list[str]) optional, name or list of names of the operations output. If more than one output
                  is present, in the list one needs to specify all the names of the output. If nothing is specified,
                  the default name is used. The default name is

                                'post_[Capital Letters + Numbers in the operation class name]_dataset'.

                  If more than one output is present, some additional words or numbers are added to the default name
                  in order to differentiate the different outputs.
        """
        self.i(x)
        self.o(y)
        return self

    def apply(self,patch_shape,representation = 'module_phase',vol_frac_th=0.05,random_patches=False,
              n_random_patches=4000,use_periodic_smooth_decomposition=True):
        """
        Apply the operation to the trace. The function 'transform_function' specified during the operation
        initialization is applied to various patches of the input data. The application can be done:

            * on the ZYX-grid, where the transformation for all the patches obtained dividing the sample into an regular
            ZYX grid which is constructed from the patch chosen;

            * at random, where patches are take in random position of the sample.

        :param patch_shape: (tuple[int]) shape of the patch used in this operation.
        :param representation: (str) optional, it can be 'module_phase' or 'real_imaginary' depending if one want to
                               save the module and phase of the DFT or the real and imaginary part of the DFT.
        :param vol_frac_th: (float between 0 and 1) fraction of the patch that need to be occupied by 1s in order to
                            consider a patch suitable for the transformation (i.e. when there are too many zeros the
                            patch is disregarded).
        :param random_patches: (bool) optional, if True the patch transform is applied in random points of the input
                               dataset, otherwise an ZYX-grid is used.
        :param n_random_patches: (int) optional, number of random patches taken (this variable is considered only when
                                 random patches are used).
        :param use_periodic_smooth_decomposition: (bool) optional, if True the periodic component of the DFT is used
                                                  instead of the simple DFT, for the patch transformation.
        """
        #
        with self.pt:

            self.patch_shape = patch_shape
            self.representation = representation
            self.vol_frac_th = vol_frac_th
            self.random_patches = random_patches
            self.n_random_patches = n_random_patches
            self.use_periodic_smooth_decomposition = use_periodic_smooth_decomposition

        #
        with h5py.File(self.trace.hdf5_trace_path, 'r') as file:

            self.data_shape = file[self.trace._hdf5_inner_path+self.input_dataset[0]].shape

        self.patch_space_shape = tuple(np.array(self.data_shape) // np.array(self.patch_shape))
        if self.representation == 'module_phase':

            c1_name = 'module'
            c2_name = 'phase'
            f1 = lambda x: np.abs(x)
            f2 = lambda x: np.angle(x)

        else:

            c1_name = 'real'
            c2_name = 'imaginary'
            f1 = lambda x: np.real(x)
            f2 = lambda x: np.imag(x)

        print('>----------------<')
        print('Patch DFT 3d')
        print('({})'.format('using random patches' if self.random_patches else 'using zyx-grid'))
        if self.random_patches:

            self._random_patches_transform(c1_name,f1,c2_name,f2)

        else:

            self._on_grid_patches_transform(c1_name,f1,c2_name,f2)

        print('>----------------<',end='\r')
        self._update_graph_dict()
        self._update_params_dict()
        if self.trace.enable_operations_tracking:

            self.trace.add_operation(PatchDiscreteFourierTransform3D.__name__,self,self.pt.get_current_tracked_params())

        return self.output_dataset

    def _on_grid_patches_transform(self,c1_name,f1,c2_name,f2):
        """
        Compute the patch-wise 3d DFT using a regular ZYX-grid.

        :param c1_name: (str) name of the dataset (dictionary key in this case) where the transformed patches are saved
                        (it can be  'module' or 'real' in this case).
        :param f1: (function) function used for the computation of the dataset saved with the c1_name.
        :param c2_name: (str) name of the dataset (dictionary key in this case) where the transformed patches are saved
                        (it can be  'phase' or 'imaginary' in this case).
        :param f2: (function) function used for the computation of the dataset saved with the c2_name.
        """
        # get coordinates
        Zs, Ys, Xs = np.meshgrid(np.arange(0,self.patch_space_shape[0]),
                                 np.arange(0,self.patch_space_shape[1]),
                                 np.arange(0,self.patch_space_shape[2]))
        ZYXs = np.vstack([Zs.flatten(), Ys.flatten(), Xs.flatten()]).transpose()
        n_patches = len(ZYXs)

        # get necessary trace attributes before the hdf5 file is open
        hdf5_inner_path = self.trace._hdf5_inner_path
        group_name = self.trace.group_name
        traced_dict = self.trace.dict_variables_traced_on_hd
        traced_pandas = self.trace.pandas_variables_traced_on_hd
        traced_numpy = self.trace.numpy_variables_traced_on_hd

        #
        with h5py.File(self.trace.hdf5_trace_path,'a') as file:

            # delete dataset with the same name of the output if present
            try:

                del file[self.trace._hdf5_inner_path+self.output_dataset[0]]
                traced_dict = [elem for elem in traced_dict
                               if '{}.{}'.format(group_name,self.output_dataset[0]) not in elem]
                traced_numpy = [elem for elem in traced_numpy
                                if '{}.{}'.format(group_name,self.output_dataset[0]) not in elem]
                traced_pandas = [elem for elem in traced_pandas
                                 if '{}.{}'.format(group_name,self.output_dataset[0]) not in elem]

            except:

                pass

            group_db = file.create_group(name=hdf5_inner_path+self.output_dataset[0])
            out_db_c1_tmp = group_db.create_dataset(c1_name+'_tmp',
                            shape=(n_patches,)+self.patch_shape,
                            dtype=np.float64,
                            chunks=(1,)+self.patch_shape)
            out_db_c2_tmp = group_db.create_dataset(c2_name+'_tmp',
                            shape=(n_patches,)+self.patch_shape,
                            dtype=np.float64,
                            chunks=(1,)+self.patch_shape)
            in_db = file[hdf5_inner_path+self.input_dataset[0]]
            nonnull_Zs = []
            nonnull_Ys = []
            nonnull_Xs = []
            m = 0
            for n in range(n_patches):

                # select patch
                z,y,x = ZYXs[n]
                zmin = int(z*self.patch_shape[0])
                zmax = zmin+self.patch_shape[0]
                ymin = int(y*self.patch_shape[1])
                ymax = ymin+self.patch_shape[1]
                xmin = int(x*self.patch_shape[2])
                xmax = xmin+self.patch_shape[2]
                patch = in_db[zmin:zmax,ymin:ymax,xmin:xmax]
                if np.sum(patch) > self.vol_frac_th*np.prod(self.patch_shape):

                    # compute transformed patch
                    t_patch = self._patch_transform(patch)
                    # store transformed patch
                    out_db_c1_tmp[m,:,:,:] = f1(t_patch)
                    out_db_c2_tmp[m,:,:,:] = f2(t_patch)
                    nonnull_Zs.append(z)
                    nonnull_Ys.append(y)
                    nonnull_Xs.append(x)
                    m += 1

                print('patch {}/{}'.format(n+1,n_patches), end='\r')

            group_db.create_dataset(c1_name,
                                    data=out_db_c1_tmp[:len(nonnull_Zs),:,:,:],
                                    dtype=np.float64,
                                    chunks=(1,) + self.patch_shape)
            del group_db[c1_name+'_tmp']
            group_db.create_dataset(c2_name,
                                    data=out_db_c2_tmp[:len(nonnull_Zs),:,:,:],
                                    dtype=np.float64,
                                    chunks=(1,) + self.patch_shape)
            del group_db[c2_name+'_tmp']


            # update trace attributes
            traced_dict.append('{}.{}'.format(group_name, self.output_dataset[0]))
            file[hdf5_inner_path].attrs['dict_variables_traced_on_hd'] = list(set(traced_dict))

            traced_numpy.append('{}.{}.{}'.format(group_name, self.output_dataset[0], c1_name))
            traced_numpy.append('{}.{}.{}'.format(group_name, self.output_dataset[0], c2_name))
            file[hdf5_inner_path].attrs['numpy_variables_traced_on_hd'] = \
                list(set([el for el in traced_numpy if group_name in el]))

        patch_space_coords_df = pd.DataFrame({'Z': nonnull_Zs,
                                              'Y': nonnull_Ys,
                                              'X': nonnull_Xs})
        patch_space_coords_df.to_hdf(self.trace.hdf5_trace_path,
                                     hdf5_inner_path+self.output_dataset[0]+'/patch_space_coordinates',
                                     mode='a', format='t', data_columns=True)
        with h5py.File(self.trace.hdf5_trace_path, 'a') as file:

            traced_pandas.append('{}.{}.{}'.format(group_name,self.output_dataset[0],'patch_space_coordinates'))
            file[hdf5_inner_path].attrs['pandas_variables_traced_on_hd'] = \
                list(set([el for el in traced_pandas if group_name in el]))

        self.trace.numpy_variables_traced_on_hd = list(set(traced_numpy))
        self.trace.pandas_variables_traced_on_hd = list(set(traced_pandas))
        self.trace.dict_variables_traced_on_hd = list(set(traced_dict))

    def _random_patches_transform(self,c1_name,f1,c2_name,f2):
        """
        Compute the patch-wise 3d DFT using random patches.

        :param c1_name: (str) name of the dataset (dictionary key in this case) where the transformed patches are saved
                        (it can be  'module' or 'real' in this case).
        :param f1: (function) function used for the computation of the dataset saved with the c1_name.
        :param c2_name: (str) name of the dataset (dictionary key in this case) where the transformed patches are saved
                        (it can be  'phase' or 'imaginary' in this case).
        :param f2: (function) function used for the computation of the dataset saved with the c2_name.
        """
        # set seed if present
        if hasattr(self.trace,'seed'):

            np.random.seed(self.trace.seed)

        # get necessary trace attributes before the hdf5 file is open
        hdf5_inner_path = self.trace._hdf5_inner_path
        group_name = self.trace.group_name
        traced_dict = self.trace.dict_variables_traced_on_hd
        traced_pandas = self.trace.pandas_variables_traced_on_hd
        traced_numpy = self.trace.numpy_variables_traced_on_hd

        #
        with h5py.File(self.trace.hdf5_trace_path,'a') as file:

            # try:

            if self.output_dataset[0] in file[hdf5_inner_path].keys():

                del file[self.trace._hdf5_inner_path+self.output_dataset[0]]
                traced_dict = [elem for elem in traced_dict
                               if '{}.{}'.format(group_name,self.output_dataset[0]) not in elem]
                traced_numpy = [elem for elem in traced_numpy
                                if '{}.{}'.format(group_name,self.output_dataset[0]) not in elem]
                traced_pandas = [elem for elem in traced_pandas
                                 if '{}.{}'.format(group_name,self.output_dataset[0]) not in elem]

            # except:
            #
            #     pass

            #
            in_db = file[hdf5_inner_path+self.input_dataset[0]]
            group_db = file.create_group(name=hdf5_inner_path+self.output_dataset[0])
            out_db_c1 = group_db.create_dataset(hdf5_inner_path+self.output_dataset[0]+'/'+c1_name,
                                                shape=(self.n_random_patches,)+self.patch_shape,
                                                dtype=np.float64,
                                                chunks=(1,)+self.patch_shape)
            out_db_c2 = group_db.create_dataset(hdf5_inner_path+self.output_dataset[0]+'/'+c2_name,
                                                shape=(self.n_random_patches,)+self.patch_shape,
                                                dtype=np.float64,
                                                chunks=(1,)+self.patch_shape)

            Zs = []
            Ys = []
            Xs = []
            while True:

                zmin = np.random.randint(self.patch_shape[0],self.data_shape[0]-self.patch_shape[0])
                zmax = zmin+self.patch_shape[0]
                ymin = np.random.randint(self.patch_shape[1],self.data_shape[1]-self.patch_shape[1])
                ymax = ymin+self.patch_shape[1]
                xmin = np.random.randint(self.patch_shape[2],self.data_shape[2]-self.patch_shape[2])
                xmax = xmin+self.patch_shape[2]
                patch = in_db[zmin:zmax,ymin:ymax,xmin:xmax]
                if np.sum(patch) > self.vol_frac_th*np.prod(self.patch_shape):

                    # compute transformed patch
                    t_patch = self._patch_transform(patch)

                    # store transformed patch
                    out_db_c1[len(Zs),:,:,:] = f1(t_patch)
                    out_db_c2[len(Zs),:,:,:] = f2(t_patch)

                    Zs.append(zmin/self.patch_shape[0])
                    Ys.append(ymin/self.patch_shape[1])
                    Xs.append(xmin/self.patch_shape[2])

                    print('patch: {}/{}'.format(len(Zs),self.n_random_patches),end='\r')
                    if len(Zs) == self.n_random_patches:

                        break

            # update trace attributes
            traced_dict.append('{}.{}'.format(group_name, self.output_dataset[0]))
            file[hdf5_inner_path].attrs['dict_variables_traced_on_hd'] = list(set(traced_dict))

            traced_numpy.append('{}.{}.{}'.format(group_name, self.output_dataset[0], c1_name))
            traced_numpy.append('{}.{}.{}'.format(group_name, self.output_dataset[0], c2_name))
            file[hdf5_inner_path].attrs['numpy_variables_traced_on_hd'] = \
                list(set([el for el in traced_numpy if group_name in el]))

        patch_space_coords_df = pd.DataFrame({'Z': Zs,'Y': Ys,'X': Xs})
        patch_space_coords_df.to_hdf(self.trace.hdf5_trace_path,
                                     hdf5_inner_path + self.output_dataset[0] + '/patch_space_coordinates',
                                     mode='a', format='t', data_columns=True)
        with h5py.File(self.trace.hdf5_trace_path, 'a') as file:

            traced_pandas.append(
                '{}.{}.{}'.format(group_name, self.output_dataset[0], 'patch_space_coordinates'))
            file[hdf5_inner_path].attrs['pandas_variables_traced_on_hd'] = \
                list(set([el for el in traced_pandas if group_name in el]))

        self.trace.numpy_variables_traced_on_hd = list(set(traced_numpy))
        self.trace.pandas_variables_traced_on_hd = list(set(traced_pandas))
        self.trace.dict_variables_traced_on_hd = list(set(traced_dict))

    def _patch_transform(self,x):
        """
        Compute the 3d DFT (or the periodic component of the 3d DFT, depending on the user inputs) of a patch

        :param x: (ndarray) patch to transform
        """
        if self.use_periodic_smooth_decomposition:

            tmp_dft,_ = periodic_smooth_decomposition_dft3(x)

        else:

            tmp_dft = dft(x)

        return np.fft.fftshift(tmp_dft)

    def read(self):
        """
        Read operation results in the hdf5 file. For this plugin, the results are organized in the form of a pandas
        dataframe (patch coordinates) and numpy array (transformation result). The first is saved as csv file, while the
        second as npy file.
        """
        trace_pos = self.trace.get_trace_pos()
        path_to_readings_folder = manage_path(
            self.trace.trace_readings_path() + os.sep + PatchDiscreteFourierTransform3D.__name__+'_{}'.format(trace_pos))
        res = self.trace.__getattribute__(self.output_dataset[0])
        for key in res.keys():

            if type(res[key]) == pd.DataFrame:

                res[key].to_csv(path_to_readings_folder + os.sep + key + '.csv')

            else:

                np.save(path_to_readings_folder + os.sep + key + '.npy', res[key])

    def _update_graph_dict(self):
        """
        Update the graph dictionary in the trace json adding the inputs and outputs of this operation.
        """
        if self.trace.enable_trace_graph == True:

            tmp_dict = self.trace.trace_graph_dict
            link_list = []
            for i_name in self.input_dataset:

                for o_name in self.output_dataset:

                    link_list.append((i_name,o_name))

            key = len(tmp_dict)
            if tmp_dict[key-1]['node'] == self.output_dataset:

                key = key-1

            tmp_dict.update({key:{'node': self.output_dataset,
                                   'link': list(set(link_list)),
                                   'edge': PatchDiscreteFourierTransform3D.__name__}})
            self.trace.__standard_setattr__('trace_graph_dict',tmp_dict)
            with open(self.trace.json_trace_path, 'r') as jfile:

                previous_content = json.load(jfile)

            previous_content[self.trace.group_name]['trace_graph'] = tmp_dict
            with open(self.trace.json_trace_path, 'w') as jfile:

                json.dump(previous_content, jfile, indent=4)

    def _update_params_dict(self):
        """
        Update the parameter dictionary in the trace json adding parameters of this operation.
        """
        with open(self.trace.json_trace_path, 'r') as jfile:

            previous_content = json.load(jfile)

        params_dict = previous_content[self.trace.group_name]['ops_parameters']
        key = len(params_dict)
        if key > 0 and \
                previous_content[self.trace.group_name]['trace_graph'][str(key-1)]['node'] == self.output_dataset:

            key = key-1
            del params_dict[str(key)]

        params_dict.update({key: {'op': PatchDiscreteFourierTransform3D.__name__,
                                  'p': self.pt.get_current_tracked_params()}})
        previous_content[self.trace.group_name]['ops_parameters'] = params_dict
        with open(self.trace.json_trace_path, 'w') as jfile:

            json.dump(previous_content, jfile, indent=4)

class DimensionalReduction:
    """
    Operation used to apply a dimensional reduction technique on some input dataset.
    """

    __name__ = 'DimensionalReduction'
    def __init__(self,trace,skl_dim_red_alg):
        """
        Initialize the operation. The PatchTransform3D operation is used to apply a transformation function specified
        by the user patch-wise on some input dataset.

        :param trace: (bmmltools.core.tracer.Trace object) trace on which the operation act.
        :param skl_dim_red_alg: (class) class containing the implementation of the dimensional reduction technique
                                which is sklearn compatible.
        """
        self.trace = trace
        self.skl_dim_red_alg = skl_dim_red_alg
        if hasattr(skl_dim_red_alg,'__name__'):

            if skl_dim_red_alg.__name__ not in DimensionalReduction.__name__:

                DimensionalReduction.__name__ = DimensionalReduction.__name__ + '_' + skl_dim_red_alg.__name__

        print(self.skl_dim_red_alg.__name__)
        self.output_dataset = ['post_' + get_capitals(DimensionalReduction.__name__) + '_dataset']
        self.pt = ParametersTracker(self)

    def i(self,x):
        """
        Specify operation inputs.

        :param x: (str, list[str]) name of the input dataset used in this operation. If more than one input is required
                  a list with the name of the input dataset need to be specified following the specific order described
                  in the bmmltools documentation.
        """
        if type(x) is str:

            self.input_dataset = [x]

        elif type(x) in [tuple,list]:

            self.input_dataset = x

        else:

            raise TypeError

        if len(self.input_dataset)>1:

            self.output_dataset = ['post_'+get_capitals(DimensionalReduction.__name__)+'_i{}_dataset'.format(n)
                                   for n in range(len(self.input_dataset)-1)]

        return self

    def o(self,x):
        """
        Specify operation outputs.

        :param x: (str, list[str]) optional, name or list of names of the operations output. If more than one output
                  is present, in the list one needs to specify all the names of the output. If nothing is specified,
                  the default name is used. The default name is

                                'post_[Capital Letters + Numbers in the operation class name]_dataset'.

                  If more than one output is present, some additional words or numbers are added to the default name
                  in order to differentiate the different outputs.
        """
        if type(x) is str:

            self.output_dataset = [x]

        elif type(x) in [tuple, list]:

            self.output_dataset = x

        else:

            raise TypeError

        return self

    def io(self,x,y):
        """
        Specify operation inputs and outputs.

        :param x: (str, list[str]) name of the input dataset used in this operation. If more than one input is required
                  a list with the name of the input dataset need to be specified following the specific order described
                  in the bmmltools documentation.
        :param y: (str, list[str]) optional, name or list of names of the operations output. If more than one output
                  is present, in the list one needs to specify all the names of the output. If nothing is specified,
                  the default name is used. The default name is

                                'post_[Capital Letters + Numbers in the operation class name]_dataset'.

                  If more than one output is present, some additional words or numbers are added to the default name
                  in order to differentiate the different outputs.
        """
        self.i(x)
        self.o(y)
        return self

    def apply(self,inference_key = None,training_key = None,n_components = 10,p = {},save_model=False,
              trained_model_path=None):
        """
        Apply the operation to the trace. This operation is used to apply a dimensional reduction techniques to a given
        input dataset.

        :param inference_key: (str or None) optional, if not None, the operation assumes that the inference datasets are
                              dictionaries and this is the key where the dataset is located.
        :param training_key: (str or None) optional, if not None, the operation assumes that the training dataset is a
                              dictionary and this is the key where the dataset is located.
        :param n_components: (int) optional, number of component to keep in order to perform the dimensional reduction.
        :param save_model: (bool) optional, if True the dimensional reduction is saved using joblib.
        :param trained_model_path: (raw str or None) optional, if is not None, this is the path of the model used for
                                   dimensional reduction. The model is assumed already trained, therefore the training
                                   phase is skipped.
        """
        with self.pt:

            self.inference_key = inference_key
            self.training_key = training_key
            self.n_components = n_components
            self.p = p
            self.save_model = save_model
            self.trained_model_path = trained_model_path

        print('>----------------<')
        print('Dimensional reduction')
        print('({})'.format('via {}'.format(self.skl_dim_red_alg.__name__)
                          if hasattr(self.skl_dim_red_alg,'__name__') else ''))

        if self.trained_model_path is not None:

            self.dm_model = self._load_model(self.trained_model_path)
            inference_datasets = self.input_dataset
            self.save_model = False

        else:

            if len(self.input_dataset) > 1:

                inference_datasets = self.input_dataset[:-1]
                training_dataset = self.input_dataset[-1]

            else:

                training_dataset = self.input_dataset[0]
                inference_datasets = [training_dataset]

            self._fit(training_dataset)

        for n,inference_dataset in enumerate(inference_datasets):

            self._transform(n,inference_dataset)

        if self.save_model:

            self._save_model()

        print('>----------------<',end='\r')
        self._update_graph_dict()
        self._update_params_dict()
        if self.trace.enable_operations_tracking:

            self.trace.add_operation(DimensionalReduction.__name__,self,self.pt.get_current_tracked_params())

        return self.output_dataset

    def _fit(self,training_dataset):
        """
        Fit a dimensional reduction model.

        :param training_dataset: (str) name of the data saved on the trace to be used for the training (it can be a
                                  numpy-like dataset or a dictionary containing a numpy-like object).
        """
        # set seed if present
        if hasattr(self.trace,'seed') and 'random_state' in list(inspect.signature(self.skl_dim_red_alg).parameters):

            self.skl_dim_red_alg = functools.partial(self.skl_dim_red_alg,random_state=self.trace.seed)

        if self.training_key is None:

            dataset = self.trace.__getattribute__(training_dataset)

        else:

            dataset = self.trace.read_dictionary_key(training_dataset,self.training_key)

        if type(dataset) is pd.DataFrame:

            X = dataset.to_numpy()

        else:

            X = dataset

        X = X.reshape((X.shape[0], -1))
        if 'n_components' in list(inspect.signature(self.skl_dim_red_alg).parameters):

            self.dm_model = self.skl_dim_red_alg(n_components=self.n_components, **self.p)

        else:

            self.dm_model = self.skl_dim_red_alg(**self.p)

        self.dm_model.fit(X)

    def _transform(self,n,inference_dataset):
        """
        Apply a trained dimensional reduction model.

        :param n: (int) number of components to keep.
        :param inference_dataset: (str) name of the data saved on the trace to be used for the inference (it can be a
                                  numpy-like dataset or a dictionary containing a numpy-like object).
        """
        if self.inference_key is None:

            dataset = self.trace.__getattribute__(inference_dataset)

        else:

            dataset = self.trace.read_dictionary_key(inference_dataset, self.inference_key)

        if type(dataset) is pd.DataFrame:

            X = dataset.to_numpy()

        else:

            X = dataset

        X = X.reshape((X.shape[0], -1))
        transformed_X = self.dm_model.transform(X)
        self.trace.__setattr__(self.output_dataset[n], transformed_X)

    def _save_model(self,name = None):
        """
        Save dimensional reduction model using joblib.

        :param name: (str) name of the file where the dimensional reduction model is saved.
        """
        if name == None:

            try:

                name = self.skl_dim_red_alg.func.__name__

            except:

                name = self.skl_dim_red_alg.__name__

        trace_pos = self.trace.get_trace_pos()
        saving_path = manage_path(self.trace.trace_file_path()+os.sep+
                                  DimensionalReduction.__name__+'_{}'.format(trace_pos))
        dump(self.dm_model,saving_path+os.sep+name+'.joblib')

    def _load_model(self,path):
        """
        Load dimensional reduction model using joblib

        :param path: (raw str) path to the dimensional reduction model saved.
        """
        return load(path)

    def read(self, name=None, save_as='npy'):
        """
        Read operation results in the hdf5 file. For this plugin, the results are organized in the form of a pandas
        dataframe.

        :param save_as: (str) optional, format used to save the result. It can be 'npy' of 'rrn'.
        :param name: (str) optional, name of the result file.
        """
        tmp_array = self.trace.__getattribute__(self.output_dataset[0])
        trace_pos = self.trace.get_trace_pos()
        path_to_readings_folder = manage_path(
            self.trace.trace_readings_path() + os.sep + DimensionalReduction.__name__+'_{}'.format(trace_pos))
        if name is None:

            name = self.output_dataset[0]

        if save_as == 'npy':

            np.save(path_to_readings_folder + os.sep + name + '.npy', tmp_array)

        elif save_as == 'rrn':

            write_rrn(tmp_array, path_to_readings_folder, name)

        else:

            pass

    def _update_graph_dict(self):
        """
        Update the graph dictionary in the trace json adding the inputs and outputs of this operation.
        """
        if self.trace.enable_trace_graph == True:

            tmp_dict = self.trace.trace_graph_dict
            link_list = []
            for i_name in self.input_dataset:

                for o_name in self.output_dataset:

                    link_list.append((i_name,o_name))

            key = len(tmp_dict)
            if tmp_dict[key-1]['node'] == self.output_dataset:

                key = key-1

            tmp_dict.update({key:{'node': self.output_dataset,
                                  'link': list(set(link_list)),
                                  'edge': DimensionalReduction.__name__}})
            self.trace.__standard_setattr__('trace_graph_dict',tmp_dict)
            with open(self.trace.json_trace_path, 'r') as jfile:

                previous_content = json.load(jfile)

            previous_content[self.trace.group_name]['trace_graph'] = tmp_dict
            with open(self.trace.json_trace_path, 'w') as jfile:

                json.dump(previous_content, jfile, indent=4)

    def _update_params_dict(self):
        """
        Update the parameter dictionary in the trace json adding parameters of this operation.
        """
        with open(self.trace.json_trace_path, 'r') as jfile:

            previous_content = json.load(jfile)

        params_dict = previous_content[self.trace.group_name]['ops_parameters']
        key = len(params_dict)
        if key > 0 and \
                previous_content[self.trace.group_name]['trace_graph'][str(key-1)]['node'] == self.output_dataset:

            key = key-1
            del params_dict[str(key)]

        params_dict.update({key: {'op': DimensionalReduction.__name__,
                                  'p': self.pt.get_current_tracked_params()}})
        previous_content[self.trace.group_name]['ops_parameters'] = params_dict
        with open(self.trace.json_trace_path, 'w') as jfile:

            json.dump(previous_content, jfile, indent=4)

DimensionalReduction_PCA = functools.partial(DimensionalReduction,skl_dim_red_alg=PCA)
DimensionalReduction_NMF = functools.partial(DimensionalReduction,skl_dim_red_alg=NMF)

class DataStandardization:
    """
    Operation used to standardize a dataset after the dimensional reduction step.
    """

    __name__ = 'DataStandardization'
    def __init__(self, trace):
        """
        Initialize the operation. The DataStandardization operation is used to standardize the dataset coming from the
        DimensionalReduction operation.

        :param trace: (bmmltools.core.tracer.Trace object) trace on which the operation act.
        """
        self.trace = trace
        self.output_dataset = ['post_' + get_capitals(DataStandardization.__name__) + '_dataset']
        self.pt = ParametersTracker(self)

    def i(self, x):
        """
        Specify operation inputs.

        :param x: (str, list[str]) name of the input dataset used in this operation. If more than one input is required
                  a list with the name of the input dataset need to be specified following the specific order described
                  in the bmmltools documentation.
        """
        if type(x) is str:

            self.input_dataset = [x]

        elif type(x) in [tuple, list]:

            self.input_dataset = x

        else:

            raise TypeError

        if len(self.input_dataset) > 1:
            self.output_dataset = ['post_' + get_capitals(DataStandardization.__name__) + '_i{}_dataset'.format(n)
                                   for n in range(len(self.input_dataset)-1)]

        return self

    def o(self, x):
        """
        Specify operation outputs.

        :param x: (str, list[str]) optional, name or list of names of the operations output. If more than one output
                  is present, in the list one needs to specify all the names of the output. If nothing is specified,
                  the default name is used. The default name is

                                'post_[Capital Letters + Numbers in the operation class name]_dataset'.

                  If more than one output is present, some additional words or numbers are added to the default name
                  in order to differentiate the different outputs.
        """
        if type(x) is str:

            self.output_dataset = [x]

        elif type(x) in [tuple, list]:

            self.output_dataset = x

        else:

            raise TypeError

        return self

    def io(self, x, y):
        """
        Specify operation inputs and outputs.

        :param x: (str, list[str]) name of the input dataset used in this operation. If more than one input is required
                  a list with the name of the input dataset need to be specified following the specific order described
                  in the bmmltools documentation.
        :param y: (str, list[str]) optional, name or list of names of the operations output. If more than one output
                  is present, in the list one needs to specify all the names of the output. If nothing is specified,
                  the default name is used. The default name is

                                'post_[Capital Letters + Numbers in the operation class name]_dataset'.

                  If more than one output is present, some additional words or numbers are added to the default name
                  in order to differentiate the different outputs.
        """
        self.i(x)
        self.o(y)
        return self

    def apply(self, axis=(), save_parameters=True, load_parameters=False,parameters_path = '',inference_key = None,
              training_key = None):
        """
        Apply the operation to the trace.

        :param axis: (tuple[int]) axis along which the standardization is performed.
        :param save_parameters: (bool) optional, if True the standardization parameter (mean and std) are saved.
        :param load_parameters: (bool) optional, if True the standardization parameter (mean and std) are loaded.
        :param parameters_path: (raw str) optional, path to the parameters to load (this field is considered only if
                                load_parameters = True).
        :param inference_key: (str or None) optional, if not None, the operation assumes that the inference datasets are
                              dictionaries and this is the key where the dataset is located.
        :param training_key: (str or None) optional, if not None, the operation assumes that the training dataset is a
                              dictionary and this is the key where the dataset is located.
        """
        #
        with self.pt:

            self.axis = axis
            self.save_parameters = save_parameters
            self.load_parameters = load_parameters
            self.parameters_path = parameters_path
            self.inference_key = inference_key
            self.training_key = training_key

        print('>----------------<')
        print('Data standardization')
        print('(along axis {})'.format(self.axis))

        M0 = None
        S0 = None
        if 1 in self.axis:

            if self.load_parameters:

                M0,S0 = self._load_parameters(self.parameters_path)
                inference_datasets = self.input_dataset
                self.save_parameters = False

            else:

                if len(self.input_dataset) > 1:

                    inference_datasets = self.input_dataset[:-1]
                    training_dataset = self.input_dataset[-1]

                else:

                    training_dataset = self.input_dataset[0]
                    inference_datasets = [training_dataset]

                M0,S0 = self._fit(training_dataset)

        for n,inference_dataset in enumerate(inference_datasets):

            self._transform(n,inference_dataset,M0,S0)

        if self.save_parameters and M0 is not None and S0 is not None:

            self._save_parameters(M0,S0)

        print('>----------------<',end='\r')

        self._update_graph_dict()
        self._update_params_dict()
        if self.trace.enable_operations_tracking:

            self.trace.add_operation(DataStandardization.__name__,self,self.pt.get_current_tracked_params())

        return self.output_dataset

    def _fit(self,training_dataset):
        """
        Compute the some standardization parameters.

        :param training_dataset: (str) name of the dataset used for the computation of the standardization parameters.
        """
        if self.training_key is None:

            x = self.trace.__getattribute__(training_dataset)

        else:

            x = self.trace.read_dictionary_key(training_dataset, self.training_key)

        M = np.mean(x,axis=0)
        S = np.std(x,axis=0)
        return M,S

    def _transform(self,n,inference_dataset,M0,S0):
        """
        Compute the some standardization parameters.

        :param n: (int) index of the output dataset.
        :param inference_dataset: (str) name of the dataset to standardize.
        :param M0: (float) mean along 0 axis.
        :param S0: (float) std along 0 axis.
        """
        if self.inference_key is None:

            x = self.trace.__getattribute__(inference_dataset)

        else:

            x = self.trace.read_dictionary_key(inference_dataset, self.inference_key)

        for ax in self.axis:

            if ax == 1:

                M = np.mean(x,axis=1,keepdims=True)
                S = np.std(x,axis=1,keepdims=True)

            elif ax == 0:

                M = M0[None,:]
                S = S0[None,:]

            x = (x-M)/S

        self.trace.__setattr__(self.output_dataset[n],x)

    def _save_parameters(self,M0,S0):
        """
        Save parameters.

        :param M0: (float) mean parameter to save.
        :param S0: (float) std paramaeter to save.
        """
        if M0 is not None and S0 is not None:

            trace_pos = self.trace.get_trace_pos()
            parameters_file_path = manage_path(self.trace.trace_file_path() + os.sep +
                                               DataStandardization.__name__ + '_{}'.format(trace_pos))
            order = str(self.axis).replace('(', '').replace(')', '').replace(', ', '')
            p_df = pd.DataFrame({'mean_axis_0': M0,'standard_deviation_axis_0': S0})
            p_df.to_csv(parameters_file_path+os.sep+'standardization_parameters_order_{}.csv'.format(order))

    def _load_parameters(self,path):
        """
        Load parameters.

        :param path: (raw std) path to the csv file containing the parameters to load.
        """
        sp_path = glob.glob(path + os.sep + '*.csv')[0]
        fn = os.path.basename(sp_path)
        fn = fn.split('.')[0]
        order = fn[fn.find('order_') + len('order_'):]
        self.axis = [int(i) for i in order]
        tmp = pd.read_csv(sp_path)
        M0 = tmp.loc[:,'mean_axis_0'].to_numpy(),
        S0 = tmp.loc[:,'standard_deviation_axis_0'].to_numpy()
        return M0,S0

    def read(self, name=None, save_as='npy'):
        """
        Read operation results in the hdf5 file. For this plugin, the results are organized in the form of a pandas
        dataframe.

        :param save_as: (str) optional, format used to save the result. It can be 'npy' of 'rrn'.
        :param name: (str) optional, name of the result file.
        """
        tmp_array = self.trace.__getattribute__(self.output_dataset[0])
        trace_pos = self.trace.get_trace_pos()
        path_to_readings_folder = manage_path(
            self.trace.trace_readings_path() + os.sep + DataStandardization.__name__+'_{}'.format(trace_pos))
        if name is None:
            name = self.output_dataset[0]

        if save_as == 'npy':

            np.save(path_to_readings_folder + os.sep + name + '.npy', tmp_array)

        elif save_as == 'rrn':

            write_rrn(tmp_array, path_to_readings_folder, name)

        else:

            pass

    def _update_graph_dict(self):
        """
        Update the graph dictionary in the trace json adding the inputs and outputs of this operation.
        """
        if self.trace.enable_trace_graph == True:

            tmp_dict = self.trace.trace_graph_dict
            link_list = []
            for i_name in self.input_dataset:

                for o_name in self.output_dataset:
                    link_list.append((i_name, o_name))

            key = len(tmp_dict)
            if tmp_dict[key - 1]['node'] == self.output_dataset:
                key = key - 1

            tmp_dict.update({key: {'node': self.output_dataset,
                                   'link': list(set(link_list)),
                                   'edge': DataStandardization.__name__}})
            self.trace.__standard_setattr__('trace_graph_dict', tmp_dict)
            with open(self.trace.json_trace_path, 'r') as jfile:

                previous_content = json.load(jfile)

            previous_content[self.trace.group_name]['trace_graph'] = tmp_dict
            with open(self.trace.json_trace_path, 'w') as jfile:

                json.dump(previous_content, jfile, indent=4)

    def _update_params_dict(self):
        """
        Update the parameter dictionary in the trace json adding parameters of this operation.
        """
        with open(self.trace.json_trace_path, 'r') as jfile:

            previous_content = json.load(jfile)

        params_dict = previous_content[self.trace.group_name]['ops_parameters']
        key = len(params_dict)
        if key > 0 and \
                previous_content[self.trace.group_name]['trace_graph'][str(key-1)]['node'] == self.output_dataset:

            key = key-1
            del params_dict[str(key)]

        params_dict.update({key: {'op': DataStandardization.__name__,
                                  'p': self.pt.get_current_tracked_params()}})
        previous_content[self.trace.group_name]['ops_parameters'] = params_dict
        with open(self.trace.json_trace_path, 'w') as jfile:

            json.dump(previous_content, jfile, indent=4)