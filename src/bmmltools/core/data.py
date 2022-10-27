# Title: 'data.py'
# Author: Curcuraci L.
# Date: 03/08/2022
#
# Scope: Data objects in bmmltools.
"""
Data object in bmmltools are used to read data coming from external sources and make them compatible with the other
bmmltools objects.
"""


#################
#####   LIBRARIES
#################


import numpy as np
import pandas as pd
import os
import glob
import imageio
import h5py
import warnings

from tqdm import tqdm
from bmmltools.utils.basic import standard_number,manage_path


#############
#####   CLASS
#############


############
#####   MAIN
############


class Data:

    def __init__(self,working_folder=None,chunks=True):
        """
        Data is used to store in a RAM efficient way the datasets used in bmmltools.

        :param working_folder: (raw str) optional, path to the folder where the hdf5 file is saved. When is not None
                               an hdf5 file is created automatically.
        :param chunks: (bool) optional, if True the dataset is saved in chunked format.
        """
        self._dataset_info = {'array': [], 'dataframe': []}
        self._dataset_in_use = None
        self.chunks = chunks
        if working_folder is not None:

            self.new(working_folder)

    def new(self,working_folder):
        """
        Create a new hdf5 data file where the dataset are saved.

        :param working_folder: (raw str) path to the folder where the hdf5 file is saved.
        """
        working_folder = os.path.abspath(os.path.normpath(working_folder))
        # if not os.path.isabs(working_folder):
        #
        #     working_folder = os.getcwd()+os.sep+working_folder

        self.working_folder = manage_path(working_folder)
        self.data_code = standard_number(np.random.randint(0,9999),4)
        self.hdf5_file_path = manage_path(self.working_folder)\
                              +os.sep+'data_{}.hdf5'.format(self.data_code)
        self._dataset_in_use = None
        with h5py.File(self.hdf5_file_path,'a') as file:

            file['_dataset_info'] = np.nan
            for key in self._dataset_info.keys():

                file['_dataset_info'].attrs[key] = self._dataset_info[key]

    def link(self,working_folder,data_code):
        """
        Link the current data to an existing hdf5 data file.

        :param working_folder: (raw str) path to the folder where the hdf5 file is saved.
        :param data_code: (int or str) data code coresponding to the hdf5 file.
        """
        if type(data_code) is int:

            data_code = standard_number(data_code,4)

        working_folder = os.path.abspath(os.path.normpath(working_folder))
        # working_folder = os.path.normpath(working_folder)
        # if not os.path.isabs(working_folder):
        #
        #     working_folder = os.getcwd()+os.sep+working_folder

        self.working_folder = working_folder
        self.data_code = data_code
        self._dataset_in_use = None
        self.hdf5_file_path = glob.glob(working_folder+os.sep+'*_{}.hdf5'.format(self.data_code))[0]
        self._get_dataset_info()

    def _get_dataset_info(self):

        tmp = {}
        with h5py.File(self.hdf5_file_path,'r') as file:

            for key in file['_dataset_info'].attrs.keys():

                tmp.update({key: list(file['_dataset_info'].attrs[key])})

        self._dataset_info = tmp

    @staticmethod
    def _is_array(value):
        """
        Check if the input is in an hdf5 compatible format.

        :param value: (variable) variable to check.
        :return: True if is compatible, otherwise False.
        """
        return type(value) in [int, float, complex] or \
               (type(value) in [list, tuple] and np.all([type(e) in [int, float, complex] for e in value])) or \
               (type(value) is np.ndarray and (np.issubdtype(value.dtype,np.integer) or \
                np.issubdtype(value.dtype,np.floating) or np.issubdtype(value.dtype,np.complexfloating)))

    @staticmethod
    def _is_df_format_compatible(df,cols):

        try:

            list(df.loc[:,cols].dtypes).index(np.object_)
            return False

        except:

            return True

    def infodata(self):
        """
        Info about the current data object
        """
        print('Data code: {}'.format(self.data_code))
        print('Current linked hdf5 file:')
        print(self.hdf5_file_path)
        print('Available datasets:')
        print(self._dataset_info['array']+self._dataset_info['dataframe'])
        print('Current dataset in use: {}'.format(self._dataset_in_use if self._dataset_in_use is not None else 'None'))

    def from_array(self,x,dataset_name):
        """
        Add a dataset starting from numpy-array-like hdf5 compatible object.

        :param x: (array-like object) dataset to add to the hdf5 file.
        :param dataset_name: (str) name of the new dataset created to store the loaded dataset.
        """
        if self._is_array(x):

            with h5py.File(self.hdf5_file_path,'a') as file:

                file.create_dataset(name=dataset_name,data=x,chunks=self.chunks)
                self._dataset_info['array'].append(dataset_name)
                self._dataset_info['array'] = list(set(self._dataset_info['array']))
                file['_dataset_info'].attrs['array'] = self._dataset_info['array']

        else:

            return -1

    def from_pandas_df(self,x,dataset_name):
        """
        Add a dataset starting from pandas dataframe.

        :param x: (pandas.Dataframe) dataset to add to the hdf5 file.
        :param dataset_name: (str) name of the new dataset created to store the loaded dataset.
        """
        if self._is_df_format_compatible(x,None):

            x.to_hdf(self.hdf5_file_path,dataset_name,'a',format='t',data_columns=True)
            self._dataset_info['dataframe'].append(dataset_name)
            self._dataset_info['dataframe'] = list(set(self._dataset_info['dataframe']))
            with h5py.File(self.hdf5_file_path,'a') as file:

                file['_dataset_info'].attrs['dataframe'] = self._dataset_info['dataframe']
                file[dataset_name].attrs['shape'] = x.shape

        else:

            warnings.warn('Input dataset has some columns which is not in a compatible format (only numeric '
                          'data type is fully supported). Nothing has been saved in the linked hdf5 file.')

    def load_stack(self,path,dataset_name):
        """
        Load dataset from a multitiff file.

        :param path: (raw str) path to the multitiff file.
        :param dataset_name: (str) name of the new dataset created to store the loaded dataset.
        """
        reader = imageio.get_reader(path)
        n_slices = reader.get_length()
        data = np.array(reader.get_data(0))
        shape = (n_slices,)+data.shape
        with h5py.File(self.hdf5_file_path, 'a') as file:

            file.create_dataset(dataset_name,shape=shape,chunks=self.chunks,dtype=data.dtype)
            file[dataset_name][0,...] = data
            for slice_index in tqdm(range(1,n_slices)):

                file[dataset_name][slice_index,...] = np.array(reader.get_data(slice_index))

        reader.close()
        self._dataset_info['array'].append(dataset_name)
        self._dataset_info['array'] = list(set(self._dataset_info['array']))
        with h5py.File(self.hdf5_file_path, 'a') as file:

            file['_dataset_info'].attrs['array'] = self._dataset_info['array']

    def load_stack_from_folder(self,path,dataset_name,extension='tiff',sorting_rule=None):
        """
        Load dataset from a stack saved slice by slice in a folder.

        :param path: (raw str) path to the folders containing the slices.
        :param dataset_name: (str) name of the new dataset created to store the loaded dataset.
        :param extension: (str) optional, file extension of the images containing the slices of the stack.
        :param sorting_rule: (function) optional, function performing the sorting of the list of file names according
                             to a specific order. The default sorting function is the simple alphabetic order sort.
        """
        if sorting_rule is None:

            sort_fn = sorted

        slice_paths = sort_fn(glob.glob(path+os.sep+'*.'+extension))
        n_slices = len(slice_paths)
        with h5py.File(self.hdf5_file_path,'a') as file:

            for slice_index,slice_path in tqdm(enumerate(slice_paths)):

                reader = imageio.get_reader(slice_path)
                data = np.array(reader.get_data(0))
                reader.close()
                if slice_index == 0:

                    shape = (n_slices,) + data.shape
                    file.create_dataset(dataset_name,shape=shape,chunks=self.chunks,dtype=data.dtype)

                file[dataset_name][slice_index, ...] = data

        self._dataset_info['array'].append(dataset_name)
        self._dataset_info['array'] = list(set(self._dataset_info['array']))
        with h5py.File(self.hdf5_file_path, 'a') as file:

            file['_dataset_info'].attrs['array'] = self._dataset_info['array']

    def load_pandas_df_from_json(self,path,dataset_name,drop_columns=[]):
        """
        Load a dataset from a pandas dataframe saved in json format.

        :param path: (raw str) path to the json file.
        :param dataset_name: (str) name of the new dataset created to store the loaded dataset.
        :param drop_columns: (list[str]) optional, list with the name of the column to drop from the loaded dataset.
        """
        loaded_df = pd.read_json(path)
        if len(drop_columns) > 0:

            loaded_df.drop(columns=drop_columns,inplace=True)

        self.from_pandas_df(loaded_df,dataset_name)

    def load_pandas_df_from_csv(self,path,dataset_name,drop_columns=[]):
        """
        Load a dataset from a pandas dataframe saved in csv format.

        :param path: (raw str) path to the csv file.
        :param dataset_name: (str) name of the new dataset created to store the loaded dataset.
        :param drop_columns: (list[str]) optional, list with the name of the column to drop from the loaded dataset.
        """
        loaded_df = pd.read_csv(path)
        if len(drop_columns) > 0:

            loaded_df.drop(columns=drop_columns, inplace=True)

        self.from_pandas_df(loaded_df, dataset_name)

    def load_npy(self,path,dataset_name):
        """
        Load a dataset from a numpy array saved in an npy file.

        :param path: (raw str) path to the npy file.
        :param dataset_name: (str) name of the new dataset created to store the loaded dataset.
        """
        x = np.load(path)
        self.from_array(x,dataset_name)

    def use_dataset(self,name):
        """
        Set the dataset to use in the i/o operations. If None is given, in the i/o operations the dataset
        need to be specified.

        :param name: (str) existing dataframe name in the hdf5 file.
        """
        self._dataset_in_use = name
        if name is not None:

            try:

                with h5py.File(self.hdf5_file_path,'r') as file:

                    if name in self._dataset_info['array']:

                        self.shape = file[name].shape
                        if hasattr(self,'columns'):

                            del self.columns

                    elif name in self._dataset_info['dataframe']:

                        self.shape = tuple(file[name].attrs['shape'])
                        self.columns = list(file[name].keys())

            except:

                pass

        else:

            if hasattr(self,'shape'):

                del self.shape

            if hasattr(self,'columns'):

                del self.columns

    def __getitem__(self,item):

        if self._dataset_in_use is None:

            if item in self._dataset_info['array']:

                with h5py.File(self.hdf5_file_path, 'r') as file:

                    loaded_data = file[item][()]

                return loaded_data

            elif item in self._dataset_info['dataframe']:

                return pd.read_hdf(self.hdf5_file_path,item)

            else:

                pass

        else:

            if self._dataset_in_use in self._dataset_info['array']:

                with h5py.File(self.hdf5_file_path, 'r') as file:

                    loaded_data = file[self._dataset_in_use][item]

                return loaded_data

            elif self._dataset_in_use in self._dataset_info['dataframe']:

                column_names, row_index = item
                if type(row_index) is slice:

                    idx_start = row_index.start
                    idx_stop = row_index.stop

                elif type(row_index) is int:

                    idx_start = row_index
                    idx_stop = idx_start+1

                if column_names is not None:

                    column_names = list(column_names)

                return pd.read_hdf(self.hdf5_file_path,self._dataset_in_use,
                                   start=idx_start,
                                   stop=idx_stop,
                                   columns=column_names)

            else:

                pass

    def __setitem__(self, key, value):

        if self._dataset_in_use is None:

            if self._is_array(value):

                if key in self._dataset_info['array']:

                    with h5py.File(self.hdf5_file_path,'a') as file:

                        del file[key]

                self.from_array(value,key)

            elif type(value) in [pd.DataFrame]:

                if key in self._dataset_info['dataframe']:

                    with h5py.File(self.hdf5_file_path, 'a') as file:

                        del file[key]

                self.from_pandas_df(value,key)

        else:

            if self._dataset_in_use in self._dataset_info['array']:

                with h5py.File(self.hdf5_file_path,'a') as file:

                    file[self._dataset_in_use][key] = value

            elif self._dataset_in_use in self._dataset_info['dataframe']:

                cols,sl = key
                df = pd.read_hdf(self.hdf5_file_path,self._dataset_in_use)
                if self._is_df_format_compatible(df,cols):

                    value.columns = cols
                    value = value.set_index(df.loc[sl,:].index)
                    df.update(value)
                    with h5py.File(self.hdf5_file_path,'a') as file:

                        del file[self._dataset_in_use]

                    self.from_pandas_df(df,self._dataset_in_use)

                else:

                    warnings.warn('Input dataset has some columns which is not in a compatible format (only numeric '
                                  'data type is fully supported). The saved dataset has been left unchanged.')

            elif self._is_array(value):

                self.from_array(value,key)
                self.use_dataset(key)

            elif type(value) in [pd.DataFrame]:

                self.from_pandas_df(value,key)
                self.use_dataset(key)