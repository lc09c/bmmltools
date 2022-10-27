# title: 'io.py'
# author: Curcuraci L.
# date: 04/08/22
#
# Scope: Collect all the I/O operations

"""
Input/Output operations in bmmltools.
"""


#################
#####   LIBRARIES
#################


import numpy as np
import pandas as pd
import h5py
import seaborn as sns
import imageio
import os
import json

from bmmltools.utils.basic import standard_number,manage_path,ParametersTracker


###############
#####   CLASSES
###############


class Input:
    """
    Input operation used to create from a Data object an input dataset on the trace.
    """

    __name__ = 'Input'
    def __init__(self,trace):
        """
        Initialize the operation. The Input operation is used to bring on the trace some dataset initially
        stored in a Data object. This operation can be used to create an input dataset on which other bmmltools
        operations may act.

        :param trace: (bmmltools.core.tracer.Trace object) trace on which the operation act.
        """
        self.trace = trace
        self.output_dataset = ['input_dataset']
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

        :param x: (str, list[str]) optional, name or list of names of the operations output. If nothing is specified,
                  the default name is used. The default name is 'input_dataset'.
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
        :param y: (str, list[str]) optional, name or list of names of the operations output. If nothing is specified,
                  the default name is used. The default name is 'input_dataset'.
        """
        self.i(x)
        self.o(y)
        return self

    def apply(self,data):
        """
        Apply the operation to the trace. This operation generate an external link to a dataset present in a Data
        object (if the is not possible the dataset is simply copied on the trace).

        :param data: (bmmltools.core.data.Data object) data object from which the input dataset is taken. The dataset
                     name is the one specified in the operation input.
        """
        data.use_dataset(None)
        data_hdf5_path = data.hdf5_file_path
        if self.input_dataset[0] in data._dataset_info['array']:

            self.trace.__setattr__(self.output_dataset[0],(data_hdf5_path,self.input_dataset[0]))

        else:

            tmp_df = pd.read_hdf(data_hdf5_path,key=self.input_dataset[0])
            self.trace.__setattr__(self.output_dataset[0],tmp_df)

        self._update_graph_dict()
        self._update_params_dict()
        if self.trace.enable_operations_tracking:

            self.trace.add_operation(Input.__name__,self,self.pt.get_current_tracked_params())

        return self.output_dataset

    def _update_graph_dict(self):
        """
        Update the graph dictionary in the trace json adding the inputs and outputs of this operation.
        """
        if self.trace.enable_trace_graph == True:

            tmp_dict = self.trace.trace_graph_dict
            key = len(tmp_dict)
            if key > 0:

                if tmp_dict[key-1]['node'] == self.output_dataset:

                    key = key - 1

            tmp_dict.update({key:{'node':self.output_dataset,'link':[],'edge':''}})
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
        if key > 0:

            if previous_content[self.trace.group_name]['trace_graph'][str(key-1)]['node'] == self.output_dataset:

                key = key-1

        params_dict.update({str(key): {'op': Input.__name__,
                                       'p': self.pt.get_current_tracked_params()}})
        previous_content[self.trace.group_name]['ops_parameters'] = params_dict
        with open(self.trace.json_trace_path, 'w') as jfile:

            json.dump(previous_content, jfile, indent=4)

class InputFromTrace:
    """
    Input operation used to specify as input dataset a dataset already present in the trace but in a different group.
    """

    __name__ = 'InputFromTrace'
    def __init__(self,trace):
        """
        Initialize the operation. The InputFromTrace operation is used to link a dataset present in a working
        group present on the trace, which is different from the one currently in use, the specified dataset. In this
        whay this datset can be used also in the current working group.

        :param trace: (bmmltools.core.tracer.Trace object) trace on which the operation act.
        """
        self.trace = trace
        self.output_dataset = ['input_from_trace_dataset']
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

        :param x: (str, list[str]) optional, name or list of names of the operations output. If nothing is specified,
                  the default name is used. The default name is 'input_dataset'.
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
        :param y: (str, list[str]) optional, name or list of names of the operations output. If nothing is specified,
                  the default name is used. The default name is 'input_dataset'.
        """
        self.i(x)
        self.o(y)
        return self

    def apply(self,dataset_name,dataset_group):
        """
        Apply the operation to the trace. This operation generate a link to a dataset present in a different group of
        the same trace. In this way a dataset present in a group different from the one on which one works can be
        used in the current working group.

        :param dataset_name: (str)  dataset name.
        :param dataset_group: (str) name of the group where the dataset is located.
        """
        with self.pt:

            self.dataset_name = dataset_name
            self.dataset_group = dataset_group

        with h5py.File(self.trace.hdf5_trace_path,'a') as file:

            try:

                del file[self.trace.group_name+'/'+self.output_dataset[0]]

            except:

                pass

            file[self.trace.group_name+'/'+self.output_dataset[0]] = file[dataset_group+'/'+dataset_name]
            fullitem = '{}.{}'.format(dataset_group,dataset_name)
            if fullitem in self.trace.numpy_variables_traced_on_hd:

                numpy_variables_traced_on_hd = self.trace.numpy_variables_traced_on_hd+ \
                                               ['{}.{}'.format(self.trace.group_name,self.output_dataset[0]),]
                file[self.trace._hdf5_inner_path].attrs['numpy_variables_traced_on_hd'] = \
                    self.trace._filter_traced_list(numpy_variables_traced_on_hd)

            elif fullitem in self.trace.pandas_variables_traced_on_hd:

                pandas_variables_traced_on_hd = self.trace.pandas_variables_traced_on_hd + \
                                               ['{}.{}'.format(self.trace.group_name, self.output_dataset[0]), ]
                file[self.trace._hdf5_inner_path].attrs['pandas_variables_traced_on_hd'] = \
                    self.trace._filter_traced_list(pandas_variables_traced_on_hd)

            elif fullitem in self.trace.dict_variables_traced_on_hd:

                dict_variables_traced_on_hd = self.trace.dict_variables_traced_on_hd + \
                                                ['{}.{}'.format(self.trace.group_name, self.output_dataset[0]), ]
                file[self.trace._hdf5_inner_path].attrs['dict_variables_traced_on_hd'] = \
                    self.trace._filter_traced_list(dict_variables_traced_on_hd)

            elif fullitem in self.trace.external_hdf5_files_linked:

                external_hdf5_files_linked = self.trace.external_hdf5_files_linked + \
                                              ['{}.{}'.format(self.trace.group_name, self.output_dataset[0]), ]
                file[self.trace._hdf5_inner_path].attrs['external_hdf5_files_linked'] = \
                    self.trace._filter_traced_list(external_hdf5_files_linked)

        self.trace._update_trace_lists()

        self._update_graph_dict()
        self._update_params_dict()
        if self.trace.enable_operations_tracking:

            self.trace.add_operation(InputFromTrace.__name__,self,self.pt.get_current_tracked_params())

        return self.output_dataset

    def _update_graph_dict(self):
        """
        Update the graph dictionary in the trace json adding the inputs and outputs of this operation.
        """
        if self.trace.enable_trace_graph == True:

            tmp_dict = self.trace.trace_graph_dict

            tmp_dict.update({len(tmp_dict):{'node':self.output_dataset,'link':[],'edge':''}})
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

        params_dict.update({key: {'op': InputFromTrace.__name__,
                                  'p': self.pt.get_current_tracked_params()}})
        previous_content[self.trace.group_name]['ops_parameters'] = params_dict
        with open(self.trace.json_trace_path, 'w') as jfile:

            json.dump(previous_content, jfile, indent=4)

class OutputRawLabels:
    """
    Output operation used to obtain a segmentation based on the labels produced by the clustering algorithm.
    """

    __name__ = 'OutputRawLabels'
    def __init__(self, trace):
        """
        Initialize the operation. The OutputRawLabels operation is used to output the segmentation constructed by
        applying the labeling obtained from a clustering algorithm directly on the input dataset (or its binarized
        version, in the input dataset is not a binary data taking value in {0,1}).

        :param trace: (bmmltools.core.tracer.Trace object) trace on which the operation act.
        """
        self.trace = trace
        self.pt = ParametersTracker(self)
        self.output_dataset = ['valid_labels_output']

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

        return self

    def o(self, x):
        """
        Specify operation outputs.

        :param x: (str, list[str]) optional, name or list of names of the operations output. If nothing is specified,
                  the default name is used. The default name is 'input_dataset'.
        """
        return self

    def io(self, x, y):
        """
        Specify operation inputs and outputs.

        :param x: (str, list[str]) name of the input dataset used in this operation. If more than one input is required
                  a list with the name of the input dataset need to be specified following the specific order described
                  in the bmmltools documentation.
        :param y: (str, list[str]) optional, name or list of names of the operations output. If nothing is specified,
                  the default name is used. The default name is 'input_dataset'.
        """
        self.i(x)
        self.o(y)
        return self

    def apply(self, patch_shape, save_separate_masks=False,save_in_patch_space = False):
        """
        Apply the operation to the trace. Produce the segmentation result, by applying on some input image (which is
        specified in the input of this operation and have to take only values only in {0,1}) the result of the
        clustering algorithm. The segmentation is produced by associating to a given voxel of the input image the same
        label of the patch (taken on an ZYX-grid) to which this voxel belongs.

        :param patch_shape: (tuple[int]) shape of the patch used to produce the clustering.
        :param save_separate_masks: (bool) optional, if True each label is saved as binary mask in a different folder.
                                    Otherwise the result is saved in a single image where different colors correspond
                                    to different clusters.
        :param save_in_patch_space: (bool) optional, if True the labels are saved in patch space (i.e. they are
                                    not superimposed to the sample).
        """
        with self.pt:

            self.patch_shape = patch_shape
            self.save_separate_maks = save_separate_masks
            self.save_in_patch_space = save_in_patch_space

        print('>----------------<')
        print('Output: raw labels')
        label_df = self.trace.__getattribute__(self.input_dataset[1])
        saving_path = manage_path(self.trace.trace_outputs_path() + os.sep + OutputRawLabels.__name__)
        if self.save_in_patch_space:

            self._save_labels_in_patch_space(label_df,saving_path)

        elif self.save_separate_maks:

            self._save_labels_in_separate_masks(label_df, saving_path)

        else:

            self._save_labels_on_input(label_df, saving_path)

        print('Raw labels saved!')
        print('>----------------<')

        self._update_graph_dict()
        self._update_params_dict()
        if self.trace.enable_operations_tracking:

            self.trace.add_operation(OutputRawLabels.__name__, self, self.pt.get_current_tracked_params())

    def _save_labels_in_separate_masks(self, label_df, saving_path):
        """
        Save the result as a series of different masks (saved as stacks of binary images) on for each label specified
        in the input datasets.

        :param label_df: (pandas.DataFrame) dataframe containing the ZYX position in patch space of each patch (in
                         columns called 'Z','Y', and 'X')  and the corresponding label (in a column called 'Label').
        :param saving_path: (str) path where the masks are saved. Each mask is saved in folders having name
                            'Label_[Label number]' created inside the trace output folder.
        """
        available_labels = list(set(label_df['label'].tolist()))
        with h5py.File(self.trace.hdf5_trace_path, 'a') as file:

            data = file[self.trace._hdf5_inner_path + self.input_dataset[0]]
            data_shape = data.shape
            for L in available_labels:

                print('Preparing mask for label {}...'.format(L),end='')
                label_l_df = label_df[label_df['label'] == L]
                label_l_df.reset_index(inplace=True, drop=True)
                canvas = np.zeros(data_shape)
                for i in range(len(label_l_df)):

                    Z = label_l_df.loc[i, 'Z']
                    zmin = Z * self.patch_shape[0]
                    zmax = (Z + 1) * self.patch_shape[0]

                    Y = label_l_df.loc[i, 'Y']
                    ymin = Y * self.patch_shape[1]
                    ymax = (Y + 1) * self.patch_shape[1]

                    X = label_l_df.loc[i, 'X']
                    xmin = X * self.patch_shape[2]
                    xmax = (X + 1) * self.patch_shape[2]

                    canvas[zmin:zmax, ymin:ymax, xmin:xmax] = np.ones(self.patch_shape)

                saving_path_L = manage_path(saving_path + os.sep + 'label_{}'.format(L))
                print('[OK]')
                for z in range(data_shape[0]):

                    print('Saving slice {}/{}'.format(z + 1, data_shape[0]), end='\r')
                    labelled_slice = canvas[z, :, :] * data[z, :, :]
                    imageio.imwrite(
                        uri=saving_path_L + os.sep + 'label_{}_-_slice_{}.tiff'.format(L, standard_number(z)),
                        im=(255 * labelled_slice).astype(np.uint8),
                        format='tiff')

                print('Mask for label {} saved'.format(L))

    def _save_labels_on_input(self, label_df, saving_path):
        """
        Save the result as single colored image, where each color corresponds to a label specified in the input
        datasets.

        :param label_df: (pandas.DataFrame) dataframe containing the ZYX position in patch space of each patch (in
                         columns called 'Z','Y', and 'X')  and the corresponding label (in a column called 'Label').
        :param saving_path: (str) path where the masks are saved. Each mask is saved in folders having name
                            'Label_[Label number]' created inside the trace output folder.
        """
        Ymin = min(label_df['Y'])
        Ymax = max(label_df['Y'])
        Xmin = min(label_df['X'])
        Xmax = max(label_df['X'])
        available_labels = list(set(label_df['label'].tolist()))
        N_labels = max(available_labels)
        cpal = sns.color_palette('hls', N_labels+1)
        with h5py.File(self.trace.hdf5_trace_path, 'a') as file:

            data = file[self.trace._hdf5_inner_path + self.input_dataset[0]]
            data_shape = data.shape
            for z in range(data_shape[0]):

                print('slice {}/{}'.format(z + 1, data_shape[0]), end='\r')
                canvas = np.ones(data_shape[1:] + (3,))
                for Y in range(Ymin, Ymax + 1):

                    ymin = Y * self.patch_shape[1]
                    ymax = (Y + 1) * self.patch_shape[1]
                    for X in range(Xmin, Xmax + 1):

                        xmin = X * self.patch_shape[2]
                        xmax = (X + 1) * self.patch_shape[2]
                        try:

                            l = label_df[(label_df['Z'] == z // self.patch_shape[0]) &
                                         (label_df['Y'] == Y) &
                                         (label_df['X'] == X)].loc[:, 'label'].tolist()[0]
                            canvas[ymin:ymax, xmin:xmax] = np.ones(self.patch_shape[1:] + (3,)) * \
                                                                    np.array(cpal[l])[None, None, :]

                        except:

                            continue

                labelled_slice = canvas * data[z, :, :][:, :, None]
                imageio.imwrite(uri=saving_path + os.sep + 'slice_{}.tiff'.format(standard_number(z)),
                                im=(255 * labelled_slice).astype(np.uint8),
                                format='tiff')

    def _save_labels_in_patch_space(self, label_df, saving_path):
        """
        Save the result as single multitiff image representing the valid clusters in the patch space.

        :param label_df: (pandas.DataFrame) dataframe containing the ZYX position in patch space of each patch (in
                         columns called 'Z','Y', and 'X')  and the corresponding label (in a column called 'Label').
        :param saving_path: (raw str) path where image is saved.
        """
        with h5py.File(self.trace.hdf5_trace_path, 'a') as file:
            data_shape = file[self.trace._hdf5_inner_path + self.input_dataset[0]].shape

        patch_space_shape = tuple(np.array(data_shape) // np.array(self.patch_shape))
        canvas = np.zeros(patch_space_shape)
        ZYXL = label_df[['Z', 'Y', 'X', 'label']].to_numpy()
        for zyxl in ZYXL:

            coord = zyxl[:3]
            label = zyxl[-1]
            canvas[coord[0], coord[1], coord[2]] = label + 1

        imageio.mimsave(os.path.join(saving_path, 'raw_clusters_in_patch_space.tiff'), canvas.astype(np.uint8))

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
            if tmp_dict[key - 1]['node'] == self.output_dataset[0]:
                key = key - 1

            tmp_dict.update({key: {'node': self.output_dataset,
                                   'link': list(set(link_list)),
                                   'edge': OutputRawLabels.__name__}})
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
                previous_content[self.trace.group_name]['trace_graph'][str(key - 1)]['node'] == self.output_dataset[0]:
            key = key - 1
            del params_dict[str(key)]

        params_dict.update({key: {'op': OutputRawLabels.__name__,
                                  'p': self.pt.get_current_tracked_params()}})
        previous_content[self.trace.group_name]['ops_parameters'] = params_dict
        with open(self.trace.json_trace_path, 'w') as jfile:
            json.dump(previous_content, jfile, indent=4)

class OutputValidLabels:
    """
    Output operation used to obtain a segmentation based on the valid clusters, possibly also with labels obtained
    after identification because similar if rotated.
    """

    __name__ = 'OutputValidLabels'
    def __init__(self, trace):
        """
        Initialize the operation. The OutputValidLabels operation is used to output the segmentation constructed by
        applying the labeling obtained from the valid clusters directly on the input dataset (or its binarized version,
        in the input dataset is not a binary data taking value in {0,1}).

        :param trace: (bmmltools.core.tracer.Trace object) trace on which the operation act.
        """
        self.trace = trace
        self.pt = ParametersTracker(self)
        self.output_dataset = ['valid_labels_output']

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

        return self

    def o(self, x):
        """
        Specify operation outputs.

        :param x: (str, list[str]) optional, name or list of names of the operations output. If nothing is specified,
                  the default name is used. The default name is 'input_dataset'.
        """
        return self

    def io(self, x, y):
        """
        Specify operation inputs and outputs.

        :param x: (str, list[str]) name of the input dataset used in this operation. If more than one input is required
                  a list with the name of the input dataset need to be specified following the specific order described
                  in the bmmltools documentation.
        :param y: (str, list[str]) optional, name or list of names of the operations output. If nothing is specified,
                  the default name is used. The default name is 'input_dataset'.
        """
        self.i(x)
        self.o(y)
        return self

    def apply(self, patch_shape, label_kind='label', point_kind='all', save_separate_masks=False,
              save_in_patch_space = False):
        """
        Apply the operation to the trace. Produce the segmentation result, by applying on some input image (which is
        specified in the input of this operation and have to take only values only in {0,1}) the clusters surviving the
        ClusterValidator operation, i.e. the valid clusters. The segmentation is produced by associating to a given
        voxel of the input image the same valid label (if any) of the patch (taken on an ZYX-grid) to which this voxel
        belongs.

        :param patch_shape: (tuple[int]) shape of the patch used to produce the clustering.
        :param label_kind: (str) optional, it can be 'label' or 'label_RS' depending if the ordinary labels or the one
                           that are not equal under rotation. In the last case the 'label_RS' column need to be present
                           since it is not produced by the ClusterValidator operation alone, but is added by the
                           RotationalSimilarityIdentifier operation. The default value is 'label'
        :param point_kind: (str) optional, it can be 'all', 'core', 'bilayer', 'boundary' based on the point
                           classification performed by the ClusterValidator operation. The default value is 'all'.
        :param save_separate_masks: (bool) optional, if True each label is saved as binary mask in a different folder.
                                    Otherwise the result is saved in a single image where different colors correspond
                                    to different clusters.
        :param save_in_patch_space: (bool) optional, if True the valid labels are saved in patch space (i.e. they are
                                    not superimposed to the sample).
        """
        with self.pt:

            self.label_kind = label_kind
            self.point_kind = point_kind
            self.patch_shape = patch_shape
            self.save_separate_maks = save_separate_masks
            self.save_in_patch_space = save_in_patch_space

        print('>----------------<')
        print('Output: valid labels')
        if self.label_kind not in ['label', 'label_RS']:

            self.label_kind = 'label'

        label_df = self.trace.__getattribute__(self.input_dataset[1])
        if self.point_kind in ['core', 'bilayer', 'boundary']:

            label_df = label_df[label_df['{}_point'.format(self.point_kind)] == 1]

        else:

            self.point_kind = 'all'

        label_df = label_df[['Z', 'Y', 'X', self.label_kind]]
        saving_path = manage_path(self.trace.trace_outputs_path() + os.sep + OutputValidLabels.__name__)
        if self.save_in_patch_space:

            self._save_labels_in_patch_space(label_df, saving_path)

        elif self.save_separate_maks:

            self._save_labels_in_separate_masks(label_df, saving_path)

        else:

            self._save_labels_on_input(label_df, saving_path)

        print('Valid labels saved!')
        print('>----------------<')

        self._update_graph_dict()
        self._update_params_dict()
        if self.trace.enable_operations_tracking:

            self.trace.add_operation(OutputValidLabels.__name__, self, self.pt.get_current_tracked_params())

    def _save_labels_in_separate_masks(self, label_df, saving_path):
        """
        Save the result as a series of different masks (saved as stacks of binary images) on for each label specified
        in the input datasets.

        :param label_df: (pandas.DataFrame) dataframe containing the ZYX position in patch space of each patch (in
                         columns called 'Z','Y', and 'X')  and the corresponding label (in a column called 'Label' or
                         'Label_RS' based on the user setting).
        :param saving_path: (raw str) path where the masks are saved. Each mask is saved in folders having name
                            'Label_[Label number]' created inside the trace output folder.
        """
        available_labels = list(set(label_df[self.label_kind].tolist()))
        with h5py.File(self.trace.hdf5_trace_path, 'a') as file:

            data = file[self.trace._hdf5_inner_path + self.input_dataset[0]]
            data_shape = data.shape
            for L in available_labels:

                print('Preparing mask for label {}...'.format(L), end='')
                label_l_df = label_df[label_df[self.label_kind] == L]
                label_l_df.reset_index(inplace=True, drop=True)
                canvas = np.zeros(data_shape)
                for i in range(len(label_l_df)):

                    Z = label_l_df.loc[i, 'Z']
                    zmin = Z * self.patch_shape[0]
                    zmax = (Z + 1) * self.patch_shape[0]

                    Y = label_l_df.loc[i, 'Y']
                    ymin = Y * self.patch_shape[1]
                    ymax = (Y + 1) * self.patch_shape[1]

                    X = label_l_df.loc[i, 'X']
                    xmin = X * self.patch_shape[2]
                    xmax = (X + 1) * self.patch_shape[2]

                    canvas[zmin:zmax, ymin:ymax, xmin:xmax] = np.ones(self.patch_shape)

                saving_path_L = manage_path(saving_path + os.sep + '{}_{}_{}_points'.format(self.label_kind,
                                                                                            L, self.point_kind))
                print('[OK]')
                for z in range(data_shape[0]):

                    print('slice {}/{}'.format(z + 1, data_shape[0]), end='\r')
                    labelled_slice = canvas[z, :, :] * data[z, :, :]
                    imageio.imwrite(
                        uri=saving_path_L + os.sep + '{}_{}_{}_points_-_slice_{}.tiff'.format(self.label_kind, L,
                                                                                              self.point_kind,
                                                                                              standard_number(z)),
                        im=(255 * labelled_slice).astype(np.uint8),
                        format='tiff')

                print('Mask for label {} saved'.format(L), end='')

    def _save_labels_on_input(self, label_df, saving_path):
        """
        Save the result as single colored image, where each color corresponds to a label specified in the input
        datasets.

        :param label_df: (pandas.DataFrame) dataframe containing the ZYX position in patch space of each patch (in
                         columns called 'Z','Y', and 'X')  and the corresponding label (in a column called 'Label' or
                         'Label_RS' based on the user setting).
        :param saving_path: (raw str) path where the images are saved.
        """
        Ymin = min(label_df['Y'])
        Ymax = max(label_df['Y'])
        Xmin = min(label_df['X'])
        Xmax = max(label_df['X'])
        available_labels = list(set(label_df[self.label_kind].tolist()))
        N_labels = max(available_labels)

        # ltr = {e:n for n,e in enumerate(available_labels)}

        cpal = sns.color_palette('hls', N_labels+1)
        with h5py.File(self.trace.hdf5_trace_path, 'a') as file:

            data = file[self.trace._hdf5_inner_path + self.input_dataset[0]]
            data_shape = data.shape
            for z in range(data_shape[0]):

                print('slice {}/{}'.format(z + 1, data_shape[0]), end='\r')
                canvas = np.ones(data_shape[1:] + (3,))
                for Y in range(Ymin, Ymax + 1):

                    ymin = Y * self.patch_shape[1]
                    ymax = (Y + 1) * self.patch_shape[1]
                    for X in range(Xmin, Xmax + 1):

                        xmin = X * self.patch_shape[2]
                        xmax = (X + 1) * self.patch_shape[2]
                        try:

                            l = label_df[(label_df['Z'] == z // self.patch_shape[0]) & (label_df['Y'] == Y) &
                                         (label_df['X'] == X)]
                            l = l.loc[:, self.label_kind].tolist()[0]
                            canvas[ymin:ymax, xmin:xmax] = np.ones(self.patch_shape[1:] + (3,)) * \
                                                                np.array(cpal[l])[None,None, :]

                        except:

                            continue

                labelled_slice = canvas * data[z, :, :][:, :, None]
                imageio.imwrite(uri=saving_path + os.sep + 'slice_{}.tiff'.format(standard_number(z)),
                                im=(255 * labelled_slice).astype(np.uint8),
                                format='tiff')

    def _save_labels_in_patch_space(self,label_df, saving_path):
        """
        Save the result as single multitiff image representing the valid clusters in the patch space.

        :param label_df: (pandas.DataFrame) dataframe containing the ZYX position in patch space of each patch (in
                         columns called 'Z','Y', and 'X')  and the corresponding label (in a column called 'Label' or
                         'Label_RS' based on the user setting).
        :param saving_path: (raw str) path where image is saved.
        """
        with h5py.File(self.trace.hdf5_trace_path, 'a') as file:

            data_shape = file[self.trace._hdf5_inner_path + self.input_dataset[0]].shape

        patch_space_shape = tuple(np.array(data_shape) // np.array(self.patch_shape))
        canvas = np.zeros(patch_space_shape)
        ZYXL = label_df[['Z', 'Y', 'X', self.label_kind]].to_numpy()
        for zyxl in ZYXL:

            coord = zyxl[:3]
            label = zyxl[-1]
            canvas[coord[0], coord[1], coord[2]] = label + 1

        imageio.mimsave(os.path.join(saving_path,'valid_clusters_in_patch_space.tiff'), canvas.astype(np.uint8))

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
            if tmp_dict[key - 1]['node'] == self.output_dataset[0]:
                key = key - 1

            tmp_dict.update({key: {'node': self.output_dataset,
                                   'link': list(set(link_list)),
                                   'edge': OutputValidLabels.__name__}})
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
                previous_content[self.trace.group_name]['trace_graph'][str(key - 1)]['node'] == self.output_dataset[0]:
            key = key - 1
            del params_dict[str(key)]

        params_dict.update({key: {'op': OutputValidLabels.__name__,
                                  'p': self.pt.get_current_tracked_params()}})
        previous_content[self.trace.group_name]['ops_parameters'] = params_dict
        with open(self.trace.json_trace_path, 'w') as jfile:
            json.dump(previous_content, jfile, indent=4)

class OutputSegmentation:
    """
    Output operation used to obtain a refined segmentation based on the weakly supervised training of a segmentation
    algorithm, where the weak label are derived from the core part of the valid labels.
    """

    __name__ = 'OutputSegmentation'
    def __init__(self, trace):
        """
        Initialize the operation. The OutputSegmentation operation is used to output the segmentation produced using
        a refinement segmentation step (RandomForestClassifier in particular) on some input dataset (or its binarized
        version, in the input dataset is not a binary data taking value in {0,1}).

        :param trace: (bmmltools.core.tracer.Trace object) trace on which the operation act.
        """
        self.trace = trace
        self.pt = ParametersTracker(self)
        self.output_dataset = ['segmentation_output']

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

        return self

    def o(self, x):
        """
        Specify operation outputs.

        :param x: (str, list[str]) optional, name or list of names of the operations output. If nothing is specified,
                  the default name is used. The default name is 'input_dataset'.
        """
        return self

    def io(self, x, y):
        """
        Specify operation inputs and outputs.

        :param x: (str, list[str]) name of the input dataset used in this operation. If more than one input is required
                  a list with the name of the input dataset need to be specified following the specific order described
                  in the bmmltools documentation.
        :param y: (str, list[str]) optional, name or list of names of the operations output. If nothing is specified,
                  the default name is used. The default name is 'input_dataset'.
        """
        self.i(x)
        self.o(y)
        return self

    def apply(self, use_RS_labels=False, save_separate_masks=False, use_rgb = False):
        """
         Apply the operation to the trace. Produce the segmentation result, by multiplying some input image (which is
        specified in the input of this operation with the refined segmentation result, which is the output of the
        RandomForestSegmenter. In this way the segmentation is restricted just to the correct parts of the input data,
        producing the correct segmentation result.

        :param use_RS_labels: (bool) optional, if True the labels identified because rotationally similar are used,
                              rather than the ordinary labels.
        :param save_separate_masks: (bool) optional, if True for each label the mask is saved in a folder called
                                    'Label_[Label number]' created in the trace output folder.
        :param use_rgb: (bool) optional, if True the output will be a stack of RGB images.
        """
        with self.pt:

            self.use_RS_labels = use_RS_labels
            self.save_separate_masks = save_separate_masks
            self.use_rgb = use_rgb

        print('>----------------<')
        print('Output: segmentation')
        valid_df = self.trace.__getattribute__(self.input_dataset[2])
        self.labels_list = list(set(valid_df['label'].tolist()))
        if self.use_RS_labels:

            self.RS_l = {}
            for l in self.labels_list:

                tmp = valid_df[valid_df['label'] == l].reset_index()
                self.RS_l.update({l: tmp.loc[0, 'RS_label']})

        saving_path = manage_path(self.trace.trace_outputs_path() + os.sep + OutputSegmentation.__name__ +
                                  os.sep + '{}segmentation_result'.format('RS_' if self.use_RS_labels else ''))
        with h5py.File(self.trace.hdf5_trace_path, 'r') as file:

            slices_result = file[self.trace._hdf5_inner_path + self.input_dataset[0]]
            slices_input = file[self.trace._hdf5_inner_path + self.input_dataset[1]]
            if self.save_separate_masks:

                self._save_final_segmentation_masks(slices_result, slices_input, saving_path)

            else:

                if self.use_rgb:

                    color_palette = sns.color_palette('hls', np.max(self.labels_list) + 1)

                else:

                    color_palette = None

                self._save_final_segmentation(slices_result, slices_input, saving_path, color_palette)

        print('Segmentation saved!')
        print('>----------------<')

        self._update_graph_dict()
        self._update_params_dict()
        if self.trace.enable_operations_tracking:

            self.trace.add_operation(OutputSegmentation.__name__, self, self.pt.get_current_tracked_params())

    def _save_final_segmentation(self, slices_result, slices_input, saving_path, color_palette):
        """
        Save the final segmentation in a single colored image.

        :param slices_result: (ndarray) slice of the segmentation result.
        :param slices_input: (ndarray) slice of the input image one which the segmentation is applied.
        :param saving_path: (raw str) path where the slices of the result are saved.
        :param color_palette: (list[tuple[float]]) color palette produced using the seaborn library.
        """
        for z in range(slices_result.shape[0]):

            print('saving slice...{}/{}'.format(z, slices_result.shape[0]), end='\r')
            tmp_slice = slices_result[z, :, :] * slices_input[z, :, :]
            if self.use_rgb:

                final_slice = np.zeros(shape=slices_result.shape[1:] + (3,))
                for l in self.labels_list:

                    mask = (tmp_slice == l + 1)
                    l_c = l
                    if self.use_RS_labels:

                        l_c = self.RS_l[l]

                    final_slice[:, :, 0][mask] = color_palette[l_c][0]
                    final_slice[:, :, 1][mask] = color_palette[l_c][1]
                    final_slice[:, :, 2][mask] = color_palette[l_c][2]

                imageio.imsave(saving_path + os.sep + 'slice_{}.tiff'.format(standard_number(z)),
                               (255 * final_slice).astype(np.uint8), format='tiff')

            else:

                imageio.imsave(saving_path + os.sep + 'slice_{}.tiff'.format(standard_number(z)),
                               (final_slice).astype(np.uint8), format='tiff')

        print('saving slice...[ok]')

    def _save_final_segmentation_masks(self, slices_result, slices_input, saving_path):
        """
        Save the final segmentation in separate masks, one fore each label.

        :param slices_result: (ndarray) slice of the segmentation result.
        :param slices_input: (ndarray) slice of the input image one which the segmentation is applied.
        :param saving_path: (str) path where the masks are saved. Each mask is saved in folders having name
                            'Label_[Label number]' created inside the trace output folder.
        """
        for l in self.labels_list:

            saving_path_L = manage_path(saving_path + os.sep + 'Label_{}'.format(l))
            for z in range(slices_result.shape[0]):

                print('saving slice for label {}...{}/{}'.format(l,z,slices_result.shape[0]),end='\r')
                tmp_slice = slices_result[z, :, :] * slices_input[z, :, :]
                l_sel = l
                if self.use_RS_labels:

                    l_sel = self.RS_l[l]

                mask = 255 * (tmp_slice == l_sel + 1).astype(np.uint8)
                imageio.imsave(saving_path_L + os.sep + 'label_{}_slice_{}.tiff'.format(l, standard_number(z)),
                               mask, format='tiff')

            print('saving slice for label {}...[ok]'.format(l))

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
            if tmp_dict[key - 1]['node'] == self.output_dataset[0]:
                key = key - 1

            tmp_dict.update({key: {'node': self.output_dataset,
                                   'link': list(set(link_list)),
                                   'edge': OutputSegmentation.__name__}})
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
                previous_content[self.trace.group_name]['trace_graph'][str(key - 1)]['node'] == self.output_dataset[0]:
            key = key - 1
            del params_dict[str(key)]

        params_dict.update({key: {'op': OutputSegmentation.__name__,
                                  'p': self.pt.get_current_tracked_params()}})
        previous_content[self.trace.group_name]['ops_parameters'] = params_dict
        with open(self.trace.json_trace_path, 'w') as jfile:
            json.dump(previous_content, jfile, indent=4)