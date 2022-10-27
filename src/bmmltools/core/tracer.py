# title: 'trace.py'
# author: Curcuraci L.
# date: 02/08/2022
#
# Scope: contain all the tracing related objects of bmmltools.

"""
Trace objects in bmmltools are used to track all the intermediate results (i.e. the results produced at the end of all
the intermediate operations of a pipeline) on the hard disk in an hdf5 file, without the need to keep them in RAM.
Variables that are tracked on a given trace, can be used in similar manner on variables that are in RAM.
"""


#################
#####   LIBRARIES
#################


import pandas as pd
import numpy as np
import h5py
import os
import glob
import json
import dill

from bmmltools.board.backend.tracegraph import inspect_graph
from bmmltools.utils.basic import manage_path,standard_number,delete_dict_key


###############
#####   CLASSES
###############


class Trace:
    """
    Class used to track variables on the hard disk in bmmltools.
    """

    numpy_variables_traced_on_hd = []
    pandas_variables_traced_on_hd = []
    external_hdf5_files_linked = []
    dict_variables_traced_on_hd = []
    group_name = '/'
    trace_graph_dict = {}
    def __init__(self,enable_trace_graph = True,enable_operations_tracking = True, seed = None):
        """
        Trace objects are used to track variables without the need to keep them in RAM, but saving them in an hdf5
        file, i.e. on the hard disk.

        :param enable_trace_graph: (bool) optional, if True the graph of all the operations applied on this trace is
                                   stored in the trace json file.
        :param enable_operations_tracking: (bool) optional, if True the initialized class of each operation of acting
                                           on this trace and the parameters given in the apply method of them are
                                           stored, and saved if 'trace.save_operations_dict()' method is called.
        :param seed: (int) optional, if given is the seed used for all the random parts of the operation acting on
                     this trace.
        """
        self.seed = seed
        self.enable_trace_graph = enable_trace_graph
        self.enable_operations_tracking = enable_operations_tracking
        if self.enable_operations_tracking:

            self.__standard_setattr__('operations_dict',{})

    def create(self, working_folder, group_name=None):
        """
        Create a new trace.

        :param working_folder: (raw str) folder where the tracing files are saved.
        :param group_name: (str) optional, name of the group ("directory") inside the hdf5 file used to store tracing
                           data.
        """
        self.trace_code = standard_number(np.random.randint(0, 9999), 4)
        self.trace_path = manage_path(working_folder+os.sep+'trace_{}'.format(self.trace_code))
        self.hdf5_trace_path = self.trace_path+os.sep+'trace_{}.hdf5'.format(self.trace_code)

        file = h5py.File(self.hdf5_trace_path,'w')
        if group_name is not None:

            file.create_group(group_name)
            self._hdf5_inner_path = '/' + group_name + '/'
            Trace.group_name = group_name

        else:

            self._hdf5_inner_path = '/'

        file.close()

        self.json_trace_path = self.trace_path+os.sep+'trace_{}.json'.format(self.trace_code)
        json_trace_dict = {Trace.group_name: {'trace_graph': Trace.trace_graph_dict,
                                              'ops_parameters': {}}}
        with open(self.json_trace_path,'w') as jfile:

            json.dump(json_trace_dict,jfile,indent=4)

    def link(self, trace_folder, group_name=None):
        """
        Link an existing trace to the initialized trace object.

        :param trace_folder: (raw str) folder where the tracing files are saved.
        :param group_name: (str) optional, name of the group ("directory") inside the hdf5 file used to store tracing
                           data.
        """
        trace_code = os.path.basename(os.path.normpath(trace_folder)).split('_')[-1]
        self.trace_path = trace_folder
        self.hdf5_trace_path = glob.glob(self.trace_path + os.sep + '*_{}.hdf5'.format(trace_code))[0]
        self.json_trace_path = glob.glob(self.trace_path + os.sep + '*_{}.json'.format(trace_code))[0]
        self.trace_code = trace_code

        file = h5py.File(self.hdf5_trace_path, 'r')
        if group_name is not None:

            self._hdf5_inner_path = '/' + group_name + '/'
            Trace.group_name = group_name

        else:

            self._hdf5_inner_path = '/'

        self._get_trace_lists(file)
        file.close()

        # load graph dict
        with open(self.json_trace_path, 'r') as file:

            content = json.load(file)

        tmp_dict = {}
        trace_dict_on_json = content[Trace.group_name]['trace_graph']
        for k in trace_dict_on_json:

            tmp_dict.update({int(k): trace_dict_on_json[k]})

        Trace.trace_graph_dict = tmp_dict
        try:

            self.load_operations_dict()

        except:

            pass
        # except:
        #
        #     raise FileExistsError

    def change_group(self, group_name):
        """
        Change the group used for tracking inside the tracking file. The group need to exist.

        :param group_name: (str) group name to use.
        """
        Trace.group_name = group_name
        self._hdf5_inner_path = '/' + group_name + ('/' if group_name is not None else '')

    def create_group(self, group_name):
        """
        Create a new group inside the tracking file and use it for tracking after calling this function.

        :param group_name: (str) new group name
        """
        try:

            with h5py.File(self.hdf5_trace_path, 'a') as file:

                file.create_group(group_name)

            Trace.group_name = group_name
            self._hdf5_inner_path = '/' + group_name + ('/' if group_name is not None else '')
            # new_trace_graph_dict = {Trace.group_name: {'trace_graph':{}}}
            Trace.trace_graph_dict.clear()
            with open(self.json_trace_path,'r') as file:

                new_trace_dict = json.loads(file.read())

            new_trace_dict.update({Trace.group_name: {'trace_graph':{},'ops_parameters':{}}})
            with open(self.json_trace_path,'w') as file:

                json.dump(new_trace_dict,file,indent=4)

        except:

            raise FileExistsError('No hdf5 file linked to this trace')

    def infotrace(self):
        """
        Get information on the current tracking status.
        """

        self._update_trace_lists()

        print('Trace code: {}'.format(self.trace_code))
        print('Current variables tracked in this trace:')
        print(Trace.numpy_variables_traced_on_hd +
              Trace.pandas_variables_traced_on_hd +
              Trace.dict_variables_traced_on_hd)
        print('Current linked external hdf5 files in this trace:')
        print(Trace.external_hdf5_files_linked)
        with h5py.File(self.hdf5_trace_path, 'r') as file:

            all_groups = list(file.keys())

        print('Groups available in this trace:')
        if np.all([g in all_groups for g in Trace.numpy_variables_traced_on_hd +
                                            Trace.pandas_variables_traced_on_hd +
                                            Trace.dict_variables_traced_on_hd +
                                            Trace.external_hdf5_files_linked]):

            print(['/'])

        else:

            print(all_groups)

        current_group = self.group_name if self.group_name is not None else '/'
        print('Tracking done in group: \n\'{}\''.format(current_group))

    def trace_file_path(self):
        """
        Get the trace file folder path.
        """
        path = os.path.dirname(self.hdf5_trace_path)
        return manage_path(path + os.sep + 'trace_{}_files'.format(self.trace_code) + os.sep + Trace.group_name)

    def trace_readings_path(self):
        """
        Get the trace readings folder path.
        """
        path = os.path.dirname(self.hdf5_trace_path)
        return manage_path(path + os.sep + 'trace_{}_readings'.format(self.trace_code) + os.sep + Trace.group_name)

    def trace_outputs_path(self):
        """
        Get the trace output folder path.
        """
        path = os.path.dirname(self.hdf5_trace_path)
        return manage_path(path + os.sep + 'trace_{}_outputs'.format(self.trace_code) + os.sep + Trace.group_name)

    def get_trace_pos(self):
        """
        Return the current position of the operation in the trace graph.
        """
        with open(self.json_trace_path, 'r') as file:

            pos = len(json.load(file)[Trace.group_name]['trace_graph'])

        return pos

    def _get_trace_lists(self, hdf5file):
        """
        Get from an hdf5 file the numpy, pandas, and dictionary variable tracking lists and external file list.
        This function needs to be used with a currently open hdf5 file.

        :param hdf5file: (file) hdf5 file to use
        """
        available_groups = list(hdf5file['/'].keys())
        numpy_traced_list = []
        pandas_traced_list = []
        external_links_list = []
        dictionary_traced_list = []
        for group in available_groups:

            for attr_key in hdf5file[group].attrs.keys():

                if 'numpy' in attr_key:

                    for attr in hdf5file[group].attrs[attr_key]:

                        numpy_traced_list.append(attr)

                if 'pandas' in attr_key:

                    for attr in hdf5file[group].attrs[attr_key]:

                        pandas_traced_list.append(attr)

                if 'external' in attr_key:

                    external_links_list.append(hdf5file[group].attrs[attr_key][0])

                if 'dict' in attr_key:

                    for attr in hdf5file[group].attrs[attr_key]:

                        dictionary_traced_list.append(attr)

        Trace.numpy_variables_traced_on_hd.clear()
        Trace.numpy_variables_traced_on_hd += numpy_traced_list

        Trace.pandas_variables_traced_on_hd.clear()
        Trace.pandas_variables_traced_on_hd += pandas_traced_list

        Trace.external_hdf5_files_linked.clear()
        Trace.external_hdf5_files_linked += external_links_list

        Trace.dict_variables_traced_on_hd.clear()
        Trace.dict_variables_traced_on_hd += dictionary_traced_list

    def _update_trace_lists(self):
        """
        Update the all the trace lists.
        """
        with h5py.File(self.hdf5_trace_path,'r') as file:

            self._get_trace_lists(file)

    def _filter_traced_list(self, traced_list):
        """
        Filter the tracked variable according to the current group used in the trace.

        :param traced_list: (list[str]) tracked variable list.
        :return: (list[str])
        """
        return [el for el in traced_list if Trace.group_name in el]

    def __filter_setmethods__(self, item, method1, method2, method3, method4, method5):
        """
        utility function: based on the kind of variable of item, one of the three methods is selected.

        :param item: (variable) object to analyze
        :param method1: (function)
        :param method2: (function)
        :param method3: (function)
        :param method4: (function)
        :param method5: (function)
        :return: (function) the selected method.
        """
        if type(item) in [int, float, complex] \
                or type(item) == np.integer \
                or type(item) == np.floating \
                or type(item) == np.complexfloating:

            return method1

        elif type(item) == np.ndarray:

            if np.issubdtype(item.dtype, np.integer) \
                    or np.issubdtype(item.dtype, np.floating) \
                    or np.issubdtype(item.dtype, np.complexfloating):

                return method1

            else:

                return method2

        elif type(item) in [list, tuple]:

            if np.all([type(e) in [int, float, complex] for e in item]):

                return method1

            elif len(item) == 2 and np.all([type(e) is str for e in item]) and '.hdf5' in item[0]:

                return method4

            else:

                return method2

        elif type(item) in [pd.DataFrame]:

            return method3

        elif type(item) is dict:

            return method5

        else:

            return method2

    def __standard_setattr__(self, key, value):

        self.__dict__[key] = value

    def __hdf5_setattr__(self, key, value):

        with h5py.File(self.hdf5_trace_path, 'a') as trace:

            try:

                del trace[self._hdf5_inner_path + key]

            except:

                pass

            fullkey = '{}.{}'.format(Trace.group_name, key)
            if not (fullkey in Trace.numpy_variables_traced_on_hd):

                trace.create_dataset(name=self._hdf5_inner_path + key, data=np.array(value))
                Trace.numpy_variables_traced_on_hd.append(fullkey)
                trace[self._hdf5_inner_path].attrs['numpy_variables_traced_on_hd'] = \
                    self._filter_traced_list(Trace.numpy_variables_traced_on_hd)

            else:

                trace.create_dataset(name=self._hdf5_inner_path + key, data=np.array(value))

    def __pandas_setattr__(self, key, value):

        if '{}.{}'.format(Trace.group_name,key) in Trace.pandas_variables_traced_on_hd:

            with h5py.File(self.hdf5_trace_path, 'a') as trace:

                del trace[self._hdf5_inner_path+key]

        value.to_hdf(self.hdf5_trace_path, self._hdf5_inner_path + key, 'a', format='t', data_columns=True)
        fullkey = '{}.{}'.format(Trace.group_name, key)
        if not (fullkey in Trace.pandas_variables_traced_on_hd):

            Trace.pandas_variables_traced_on_hd.append(fullkey)
            with h5py.File(self.hdf5_trace_path, 'a') as trace:

                trace[self._hdf5_inner_path].attrs['pandas_variables_traced_on_hd'] = \
                    self._filter_traced_list(Trace.pandas_variables_traced_on_hd)

    def __hdf5_external_setattr__(self, key, value):

        (external_db_link, external_db_key) = value
        with h5py.File(self.hdf5_trace_path, 'a') as trace:

            fullkey = '{}.{}'.format(Trace.group_name, key)
            if not (fullkey in Trace.external_hdf5_files_linked):

                trace[self._hdf5_inner_path + key] = h5py.ExternalLink(external_db_link, external_db_key)
                Trace.external_hdf5_files_linked.append(fullkey)
                trace[self._hdf5_inner_path].attrs['external_hdf5_files_linked'] = \
                    self._filter_traced_list(Trace.external_hdf5_files_linked)

            else:

                del trace[self._hdf5_inner_path]
                trace[self._hdf5_inner_path + key] = h5py.ExternalLink(external_db_link, external_db_key)

    def __dict_setattr__(self, key, value):

        fullkey = '{}.{}'.format(Trace.group_name, key)
        if not (fullkey in Trace.dict_variables_traced_on_hd):

            for k in value.keys():

                if type(value[k]) == pd.DataFrame:

                    value[k].to_hdf(self.hdf5_trace_path,self._hdf5_inner_path+key+'/'+k,'a',format='t',
                                    data_columns=True)
                    Trace.pandas_variables_traced_on_hd.append('{}.{}.{}'.format(Trace.group_name,key,k))
                    with h5py.File(self.hdf5_trace_path, 'a') as trace:

                        trace[self._hdf5_inner_path].attrs['pandas_variables_traced_on_hd'] = \
                            self._filter_traced_list(Trace.pandas_variables_traced_on_hd)

                else:

                    Trace.numpy_variables_traced_on_hd.append('{}.{}.{}'.format(Trace.group_name, key, k))
                    with h5py.File(self.hdf5_trace_path, 'a') as trace:

                        trace[self._hdf5_inner_path + key + '/' + k] = value[k]
                        trace[self._hdf5_inner_path].attrs['numpy_variables_traced_on_hd'] = \
                            self._filter_traced_list(Trace.numpy_variables_traced_on_hd)

            Trace.dict_variables_traced_on_hd.append(fullkey)
            with h5py.File(self.hdf5_trace_path, 'a') as trace:

                trace[self._hdf5_inner_path].attrs['dict_variables_traced_on_hd'] = \
                    self._filter_traced_list(Trace.dict_variables_traced_on_hd)

        else:

            with h5py.File(self.hdf5_trace_path, 'a') as trace:

                del trace[self._hdf5_inner_path+key]

            new_pandas_trace_list = [elem for elem in Trace.pandas_variables_traced_on_hd if fullkey+'.' not in elem]
            Trace.pandas_variables_traced_on_hd = self._filter_traced_list(new_pandas_trace_list)
            for k in value.keys():

                if type(value[k]) == pd.DataFrame:

                    value[k].to_hdf(self.hdf5_trace_path,self._hdf5_inner_path+key+'/'+k,'a',format='t',
                                    data_columns=True)
                    Trace.pandas_variables_traced_on_hd.append('{}.{}.{}'.format(Trace.group_name,key,k))

                else:

                    with h5py.File(self.hdf5_trace_path, 'a') as trace:

                        Trace.numpy_variables_traced_on_hd.append('{}.{}.{}'.format(Trace.group_name, key, k))
                        trace[self._hdf5_inner_path + key + '/'+k] = value[k]

            with h5py.File(self.hdf5_trace_path, 'a') as trace:

                trace[self._hdf5_inner_path].attrs['pandas_variables_traced_on_hd'] = \
                    self._filter_traced_list(Trace.pandas_variables_traced_on_hd)


    def __setattr__(self, key, value):

        setattr_method = self.__filter_setmethods__(value,
                                                    self.__hdf5_setattr__,
                                                    self.__standard_setattr__,
                                                    self.__pandas_setattr__,
                                                    self.__hdf5_external_setattr__,
                                                    self.__dict_setattr__)
        setattr_method(key, value)

    def __hdf5__getattribute__(self, item):

        with h5py.File(self.hdf5_trace_path, 'r') as trace:
            tmp = trace[self._hdf5_inner_path + item][()]

        return np.squeeze(tmp)

    def __pandas_getattribute__(self, item):

        return pd.read_hdf(self.hdf5_trace_path, self._hdf5_inner_path + item, 'r')

    def __dict_getattribute__(self, item):

        tmp = {}
        fullkey = '{}.{}.'.format(Trace.group_name,item)
        dict_keys = [elem.split('.')[-1] for elem in Trace.numpy_variables_traced_on_hd+Trace.pandas_variables_traced_on_hd
                     if fullkey in elem]
        for k in dict_keys:

            if '{}.{}.{}'.format(Trace.group_name,item,k) in Trace.pandas_variables_traced_on_hd:

                tmp.update({k: pd.read_hdf(self.hdf5_trace_path,self._hdf5_inner_path+item+'/'+k,'r')})

            else:

                with h5py.File(self.hdf5_trace_path, 'r') as trace:

                    tmp.update({k: trace[self._hdf5_inner_path + item + '/' + k][()]})

        return tmp

    def __getattribute__(self, item):

        fullitem = Trace.group_name + '.' + item
        if fullitem in Trace.numpy_variables_traced_on_hd or fullitem in Trace.external_hdf5_files_linked:

            return self.__hdf5__getattribute__(item)

        elif fullitem in Trace.pandas_variables_traced_on_hd:

            return self.__pandas_getattribute__(item)

        elif fullitem in Trace.dict_variables_traced_on_hd:

            return self.__dict_getattribute__(item)

        else:

            return super().__getattribute__(item)

    def read_dictionary_key(self,item,key):
        """
        Read the value of a key of a dictionary stored on the trace.

        :param item: (str) dictionary variable name.
        :parma key: (str) dictionary key name.
        """
        self._update_trace_lists()
        fullitemkey = '{}.{}.{}'.format(Trace.group_name,item,key)
        if fullitemkey in Trace.numpy_variables_traced_on_hd:

            with h5py.File(self.hdf5_trace_path,'r') as file:

                content = file[self._hdf5_inner_path+item+'/'+key][()]

        elif fullitemkey in Trace.pandas_variables_traced_on_hd:

            content = pd.read_hdf(self.hdf5_trace_path,self._hdf5_inner_path+item+'/'+key)

        else:

            content = None

        return content

    def write_dictionary_key(self,item, key, value):
        """
        Write the value of a key in a dictionary stored on the trace.

        :param item: (str) dictionary variable name.
        :parma key: (str) dictionary key name.
        """
        fullitem = '{}.{}'.format(Trace.group_name,item)
        fullitemkey = '{}.{}'.format(fullitem, key)
        with h5py.File(self.hdf5_trace_path, 'a') as file:

            try:

                group = file[self._hdf5_inner_path+item]

            except:

                group = file.create_group(self._hdf5_inner_path + item)
                Trace.dict_variables_traced_on_hd.append(fullitem)

            if type(value) is pd.DataFrame:

                value.to_hdf(self.hdf5_trace_path,self._hdf5_inner_path+key)
                if fullitemkey not in Trace.pandas_variables_traced_on_hd:

                    Trace.pandas_variables_traced_on_hd.append(fullitemkey)

            else:

                if fullitemkey not in Trace.numpy_variables_traced_on_hd:

                    group[key] = value
                    Trace.numpy_variables_traced_on_hd.append(fullitemkey)

                else:

                    group[key][()] = value

    def __standard_delattr__(self, item):

        del self.__dict__[item]

    def __hdf5_delattr__(self, item, fullitem):

        del Trace.numpy_variables_traced_on_hd[Trace.numpy_variables_traced_on_hd.index(fullitem)]
        with h5py.File(self.hdf5_trace_path, 'r+') as file:
            tmp = list(file[self._hdf5_inner_path].attrs['numpy_variables_traced_on_hd'])
            del tmp[tmp.index(fullitem)]
            file[self._hdf5_inner_path].attrs['numpy_variables_traced_on_hd'] = self._filter_traced_list(tmp)
            del file[self._hdf5_inner_path + item]

    def __pandas_delattr__(self, item, fullitem):

        del Trace.pandas_variables_traced_on_hd[Trace.pandas_variables_traced_on_hd.index(fullitem)]
        with h5py.File(self.hdf5_trace_path, 'r+') as file:
            tmp = list(file[self._hdf5_inner_path].attrs['pandas_variables_traced_on_hd'])
            del tmp[tmp.index(fullitem)]
            file[self._hdf5_inner_path].attrs['pandas_variables_traced_on_hd'] = self._filter_traced_list(tmp)
            del file[self._hdf5_inner_path + item]

    def __hdf5_external_delattr__(self, item, fullitem):

        del Trace.external_hdf5_files_linked[Trace.external_hdf5_files_linked.index(fullitem)]
        with h5py.File(self.hdf5_trace_path, 'r+') as file:
            tmp = list(file[self._hdf5_inner_path].attrs['external_hdf5_files_linked'])
            del tmp[tmp.index(fullitem)]
            file[self._hdf5_inner_path].attrs['external_hdf5_files_linked'] = self._filter_traced_list(tmp)
            del file[self._hdf5_inner_path + item]

    def __dict_delattr__(self, item, fullitem):

        del Trace.dict_variables_traced_on_hd[Trace.dict_variables_traced_on_hd.index(fullitem)]
        new_numpy_trace_list = [elem for elem in Trace.numpy_variables_traced_on_hd if fullitem + '.' not in elem]
        Trace.numpy_variables_traced_on_hd.clear()
        Trace.numpy_variables_traced_on_hd += self._filter_traced_list(new_numpy_trace_list)
        new_pandas_trace_list = [elem for elem in Trace.pandas_variables_traced_on_hd if fullitem + '.' not in elem]
        Trace.pandas_variables_traced_on_hd.clear()
        Trace.pandas_variables_traced_on_hd += self._filter_traced_list(new_pandas_trace_list)
        with h5py.File(self.hdf5_trace_path, 'r+') as file:

            tmp = list(file[self._hdf5_inner_path].attrs['dict_variables_traced_on_hd'])
            del tmp[tmp.index(fullitem)]
            file[self._hdf5_inner_path].attrs['dict_variables_traced_on_hd'] = self._filter_traced_list(tmp)
            file[self._hdf5_inner_path].attrs['numpy_variables_traced_on_hd'] = Trace.numpy_variables_traced_on_hd
            file[self._hdf5_inner_path].attrs['pandas_variables_traced_on_hd'] = Trace.pandas_variables_traced_on_hd
            del file[self._hdf5_inner_path + item]

    def __delattr__(self, item):

        delattr_method = self.__standard_delattr__
        fullitem = '{}.{}'.format(Trace.group_name, item)
        if fullitem in Trace.numpy_variables_traced_on_hd:

            delattr_method = lambda x: self.__hdf5_delattr__(x, fullitem)

        if fullitem in Trace.pandas_variables_traced_on_hd:

            delattr_method = lambda x: self.__pandas_delattr__(x, fullitem)

        if fullitem in Trace.external_hdf5_files_linked:

            delattr_method = lambda x: self.__hdf5_external_delattr__(x, fullitem)

        if fullitem in Trace.dict_variables_traced_on_hd:

            delattr_method = lambda x: self.__dict_delattr__(x, fullitem)

        delattr_method(item)
        self._delete_from_trace_and_operation_json(item)
        self._update_trace_lists()

    def _delete_from_trace_and_operation_json(self, item):
        """
        Delete an operation already applied to the trace from it in the proper manner.

        :param item: (str) name of the operation to delete.
        """
        if self.enable_trace_graph:

            with open(self.json_trace_path, 'r') as file:

                group_dict = json.load(file)
                graph_dict = group_dict[Trace.group_name]['trace_graph']
                param_dict = group_dict[Trace.group_name]['ops_parameters']

            all_group_names = list(group_dict.keys())
            key_to_remove = list(inspect_graph(item,graph_dict).keys())[0]
            graph_dict = delete_dict_key(graph_dict,key_to_remove)
            Trace.trace_graph_dict = graph_dict
            param_dict = delete_dict_key(param_dict,key_to_remove)
            if self.enable_operations_tracking:

                tmp_dict = delete_dict_key(self.operations_dict,int(key_to_remove))
                self.operations_dict.clear()
                self.operations_dict.update(tmp_dict)

            to_save = {}
            for key in all_group_names:

                if key == Trace.group_name:

                    to_save.update({key: {'trace_graph': graph_dict,
                                          'ops_parameters': param_dict}})

                else:

                    to_save.update({key: group_dict[key]})

            with open(self.json_trace_path, 'w') as file:

                json.dump(to_save, file,indent=4)

    def add_operation(self, name, cls, apply_params):
        """
        Add an operation to the operation dict of the trace.

        :param name: (str) operation name.
        :param cls: (object) initialized operation class.
        :param apply_params: (dict) dictionary containing the parameters used in the apply method of the initialized
                             operation.
        """
        N = len(self.operations_dict)
        if N > 0:

            if name == self.operations_dict[N-1][0] and \
                    cls.output_dataset == self.operations_dict[N-1][1].output_dataset:

                self.del_operation(N-1)

        self.__dict__['operations_dict'].update({len(self.operations_dict):(name,cls,apply_params)})

    def del_operation(self, key):
        """
        Delete an operation from the operation dict of the trace.

        :param key: (int) operation number (key in the operation dictionary).
        """
        del self.__dict__['operations_dict'][key]

    def save_operations_dict(self, path = None):
        """
        Save the operation dictionary in a dill file.

        :param path: (raw str) optional, path where the trace dill file is saved. If nothing is given, the dill file
                     will be created in the trace folder.
        """
        if hasattr(self,'operations_dict'):

            if path is None:

                path = self.trace_path + os.sep + 'trace_{}_operations.dill'.format(self.trace_code)

            with open(path, 'wb') as file:

                dill.dump(self.__dict__['operations_dict'], file)

    def load_operations_dict(self, path = None):
        """
        Load an existing operation dictionary.

        P.A.: Loading the operation dictionary does not mean to load the series of operation applied to the trace.

        :param path: (raw str) optional, path where the trace dill file is located. If nothing is given, the dill file
                     is assumed to be found in the trace folder.
        """
        if hasattr(self,'operations_dict'):

            if path is None:

                path = self.trace_path + os.sep + 'trace_{}_operations.dill'.format(self.trace_code)

            with open(path, 'rb') as file:

                self.__dict__['operations_dict'] = dill.load(file)