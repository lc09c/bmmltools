# Title: 'clustering.py'
# Author: Curcuraci L.
# Date: 17/10/2022
#
# Scope: collect the clustering related operations.

"""
Operations used to do clustering and study some of their properties.
"""


#################
#####   LIBRARIES
#################


import numpy as np
import json
import os
import functools
import hdbscan as hdbs
import h5py
import pandas as pd
import copy
import inspect

from joblib import dump, load
from sklearn.cluster import DBSCAN,KMeans
from scipy.ndimage import distance_transform_edt,binary_erosion,binary_dilation,binary_fill_holes,gaussian_filter
from skimage.feature import match_template
from scipy.stats import f as f_distribution
from scipy.stats import ncf as non_centred_f_distribution

from bmmltools.utils.basic import get_capitals,ParametersTracker,manage_path
from bmmltools.features.dft import periodic_smooth_decomposition_dft3
from bmmltools.utils.geometric import ChangeCoordinateSystem,spherical_to_cartesian,grad_to_rad


###############
#####   CLASSES
###############


class Clusterer:
    """
    Operation used to perform clustering.
    """

    __name__ = 'Clusterer'
    def __init__(self,trace,skl_cluster_alg):
        """
        Initialize the operation. The Clusterer operation is used to perform clustering on the dataset given as input.

        :param trace: (bmmltools.core.tracer.Trace object) trace on which the operation act.
        :param skl_cluster_alg: (class) class containing the implementation of a clustering technique which is sklearn
                                compatible.
        """
        self.trace = trace
        self.skl_cluster_alg = skl_cluster_alg
        if hasattr(skl_cluster_alg,'__name__'):

            if skl_cluster_alg.__name__ not in Clusterer.__name__:

                Clusterer.__name__ = Clusterer.__name__+'_'+skl_cluster_alg.__name__

        self.output_dataset = ['post_'+get_capitals(Clusterer.__name__)+'_dataset']
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

        self.output_dataset = ['post_'+get_capitals(Clusterer.__name__)+'_i{}_dataset'.format(k)
                               for k in range(len(self.input_dataset)-1)]
        if len(self.output_dataset)>2:

            self.output_dataset = self.output_dataset[:-1]

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

    def apply(self,p={}):
        """
        Apply the operation to the trace. Apply the clustering algorithm to some input dataset.

        :param p: (dict) dictionary containing the parameters name (as key) and values (as value) of the clustering
                  algorithm.
        """
        if hasattr(self.trace,'seed') and 'random_state' in list(inspect.signature(self.skl_dim_red_alg).parameters):

            self.skl_dim_red_alg = functools.partial(self.skl_dim_red_alg,random_state=self.trace.seed)

        with self.pt:

            self.p = p

        print('>----------------<')
        print('Clustering')
        inference_datasets = self.input_dataset[:-1]
        patch_shape_coords_df = self.trace.read_dictionary_key(self.input_dataset[-1],'patch_space_coordinates')
        for n,inference_dataset in enumerate(inference_datasets):

            self._cluster(n,inference_dataset,patch_shape_coords_df)
            if hasattr(self,'_N_clusters_found'):

                print('Found {} cluster in dataset given as input {}'-format(self._N_clusters_found,n))


        print('>----------------<',end='\r')

        self._update_graph_dict()
        self._update_params_dict()
        if self.trace.enable_operations_tracking:

            self.trace.add_operation(Clusterer.__name__,self,self.pt.get_current_tracked_params())

        return self.output_dataset

    def _cluster(self,n,data,df):
        """
        Core clustering algorithm.

        :param n: (int) index of the output dataset.
        :param data: (str) name of the dataset to cluster.
        :param df: (pandas.DataFrame) dataframe where the clustering result will be stored in a colunm.
        """
        X = self.trace.__getattribute__(data)
        cluster_model = self.skl_cluster_alg(**self.p)
        cluster_model.fit(X)
        labels = cluster_model.labels_
        df.loc[:,'label'] = labels
        self._N_clusters_found = len(list(set(labels)))
        self.trace.__setattr__(self.output_dataset[n],df)

    def read(self,name=None,save_as='csv'):
        """
        Read operation results in the hdf5 file. For this plugin, the results are organized in the form of a pandas
        dataframe.

        :param save_as: (str) optional, format used to save the result. It can be 'npy' of 'rrn'.
        :param name: (str) optional, name of the result file.
        """
        df = self.trace.__getattribute__(self.output_dataset[0])
        if name == None:

            name = self.output_dataset[0]

        trace_pos = self.trace.get_trace_pos()
        saving_path = manage_path(self.trace.trace_readings_path() + os.sep +
                                  Clusterer.__name__ + '_{}'.format(trace_pos))
        if save_as == 'csv':

            df.to_csv(saving_path+os.sep+name+'.'+save_as)

        elif save_as == 'json':

            df.to_json(saving_path+os.sep+name+'.'+save_as)

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
                                  'edge': Clusterer.__name__}})
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

        params_dict.update({key: {'op': Clusterer.__name__,
                                  'p': self.pt.get_current_tracked_params()}})
        previous_content[self.trace.group_name]['ops_parameters'] = params_dict
        with open(self.trace.json_trace_path, 'w') as jfile:

            json.dump(previous_content, jfile, indent=4)

Clusterer_KMeans = functools.partial(Clusterer,skl_cluster_alg = KMeans)
Clusterer_DBSCAN = functools.partial(Clusterer,skl_cluster_alg = DBSCAN)

class Clusterer_HDBSCAN:
    """
    Operation used to perform clustering with HDBSCAN.
    """

    __name__ = 'Clusterer_HDBSCAN'
    def __init__(self, trace):
        """
        Initialize the operation. The Clusterer operation is used to perform clustering on the dataset given as input.

        :param trace: (bmmltools.core.tracer.Trace object) trace on which the operation act.
        """
        self.trace = trace
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

        self.output_dataset = ['post_' + get_capitals(Clusterer_HDBSCAN.__name__) + '_i{}_dataset'.format(k)
                               for k in range(len(self.input_dataset) - 1)]
        if len(self.output_dataset) > 2:
            self.output_dataset = self.output_dataset[:-1]

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

    def apply(self, p={}, save_model=True, trained_model_path=None):
        """
        Apply the operation to the trace. Apply the HDBSCAN clustering algorithm to some input dataset.

        :param p: (dict) optional, dictionary containing the parameters name (as key) and values (as value) of the
                  clustering algorithm.
        :param save_model: (bool) optional, if True the HDBSCAN model trained is saved.
        :param trained_model_path: (str or None) optional, if a path is given, this is the pat used to load a trained
                                   HDBSCAN model.

                                   **P.A.**: Not all the combinations of parameters works when the inference dataset is
                                   different from the training dataset.
        """
        with self.pt:

            self.p = p
            self.save_model = save_model
            self.trained_model_path = trained_model_path

        print('>----------------<')
        print('Clustering')
        if self.p == {}:

            self.p = hdbs.HDBSCAN.get_params()

        if self.trained_model_path is not None:

            self.cluster_model = self._load_model(self.trained_model_path)
            inference_datasets = self.input_dataset[:-1]
            self.save_model = False

        else:

            if len(self.input_dataset) == 2:

                training_dataset = self.input_dataset[0]
                inference_datasets = [training_dataset]

            else:

                inference_datasets = self.input_dataset[:-2]
                training_dataset = self.inference_datasets[-2]

            self._fit(training_dataset)

        patch_shape_coords_df = self.trace.read_dictionary_key(self.input_dataset[-1], 'patch_space_coordinates')
        for n, inference_dataset in enumerate(inference_datasets):

            self._cluster(n, inference_dataset, patch_shape_coords_df)
            if hasattr(self,'_N_clusters_found'):

                print('Found {} cluster in dataset given as input {}'-format(self._N_clusters_found,n))

        if self.save_model:

            self._save_model()

        print('>----------------<', end='\r')

        self._update_graph_dict()
        self._update_params_dict()
        if self.trace.enable_operations_tracking:

            self.trace.add_operation(Clusterer_HDBSCAN.__name__,self,self.pt.get_current_tracked_params())

        return self.output_dataset

    def _fit(self, training_data):
        """
        Fit the clustering algorithm.

        :param training_data: (str) name of the dataset used for training.
        """
        X = self.trace.__getattribute__(training_data)
        self.cluster_model = hdbs.HDBSCAN(**self.p)
        self.cluster_model.fit(X)

    def _cluster(self, n, data, df):
        """
        Core clustering algorithm.

        :param n: (int) index of the output dataset.
        :param data: (str) name of the dataset to cluster.
        :param df: (pandas.DataFrame) dataframe where the clustering result will be stored in a colunm.
        """
        X = self.trace.__getattribute__(data)
        if len(self.input_dataset) == 2:

            soft_clusters = hdbs.all_points_membership_vectors(self.cluster_model)

        else:

            soft_clusters = hdbs.membership_vector(self.cluster_model, X)

        labels = [np.argmax(x) for x in soft_clusters]
        df.loc[:,'label'] = labels
        self._N_clusters_found = len(list(set(labels)))
        self.trace.__setattr__(self.output_dataset[n], df)

    def _save_model(self):
        """
        Save HDBSCAN model using joblib.

        :param name: (str) name of the file where the dimensional reduction model is saved.
        """
        trace_pos = self.trace.get_trace_pos()
        saving_path = manage_path(self.trace.trace_file_path() + os.sep +
                                  Clusterer_HDBSCAN.__name__ + '_{}'.format(trace_pos))
        dump(self.cluster_model, saving_path + os.sep + 'hdbscan_model.joblib')

    def _load_model(self, path):
        """
        Load HDBSCAN clustering using joblib.

        :param path: (raw str) path to the dimensional reduction model saved.
        """
        return load(path)

    def read(self, name=None, save_as='csv'):
        """
        Read operation results in the hdf5 file. For this plugin, the results are organized in the form of a pandas
        dataframe.

        :param save_as: (str) optional, format used to save the result. It can be 'npy' of 'rrn'.
        :param name: (str) optional, name of the result file.
        """
        df = self.trace.__getattribute__(self.output_dataset[0])
        if name == None:

            name = self.output_dataset[0]

        trace_pos = self.trace.get_trace_pos()
        saving_path = manage_path(self.trace.trace_readings_path() + os.sep +
                                  Clusterer_HDBSCAN.__name__ + '_{}'.format(trace_pos))
        if save_as == 'csv':

            df.to_csv(saving_path + os.sep + name + '.' + save_as)

        elif save_as == 'json':

            df.to_json(saving_path + os.sep + name + '.' + save_as)

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
                                   'edge': Clusterer_HDBSCAN.__name__}})
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

        params_dict.update({key: {'op': Clusterer_HDBSCAN.__name__,
                                  'p': self.pt.get_current_tracked_params()}})
        previous_content[self.trace.group_name]['ops_parameters'] = params_dict
        with open(self.trace.json_trace_path, 'w') as jfile:

            json.dump(previous_content, jfile, indent=4)

class ClusterValidator:
    """
    Operation used to validate from a geometric point of view the cluster found by a clustering algorithm.
    """

    __name__ = 'ClusterValidator'
    def __init__(self, trace):
        """
        Initialize the operation. The ClustererValidator operation is used to evaluate if the clusters found are
        reasonable or not using purely geometrical considerations. What is checked is:

        * sufficient degree of continuity in patch space;

        * sufficient volume in patch space.

        :param trace: (bmmltools.core.tracer.Trace object) trace on which the operation act.
        """
        self.trace = trace
        self.output_dataset = ['post_' + get_capitals(ClusterValidator.__name__) + '_dataset']
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

    def apply(self,patch_space_volume_th = 1,patch_space_core_volume_th = 1):
        """
        Apply the operation to the trace. Apply the validation procedure to a set of clusters.

        :param patch_space_volume_th: (int) optional, threshold on the volume in patch space below witch a cluster is
                                      eliminated.
        :param patch_space_core_volume_th: (int) optional, threshold on the volume of the core part of a cluster in
                                           patch space below witch a cluster is eliminated.
        """
        with self.pt:

            self.patch_space_volume_th = patch_space_volume_th
            self.patch_space_core_volume_th = patch_space_core_volume_th

        print('>----------------<')
        print('Valid cluster identification')

        #
        result_df = self.trace.__getattribute__(self.input_dataset[0])

        #
        labels = list(set(result_df.loc[:, 'label'].to_list()))
        Zmax = max(result_df.loc[:, 'Z'])
        Ymax = max(result_df.loc[:, 'Y'])
        Xmax = max(result_df.loc[:, 'X'])
        labels_in_patch_space = -2 * np.ones((Zmax + 1, Ymax + 1, Xmax + 1))
        for i in range(len(result_df)):

            z, y, x, l = result_df.iloc[i].to_list()
            labels_in_patch_space[z, y, x] = l

        valid_labels_df = pd.DataFrame()
        for l in labels:

            l_in_patch_space = (labels_in_patch_space == l)

            # check label spatial continuity and define core part
            l_in_patch_space = binary_erosion(l_in_patch_space, border_value=True)
            l_in_patch_space = binary_dilation(l_in_patch_space, border_value=False)
            l_in_patch_space = binary_fill_holes(l_in_patch_space).astype(np.uint8)
            l_core_in_patch_space = binary_erosion(l_in_patch_space, border_value=True).astype(np.uint8)
            l_boundary_point_in_patch_space = l_in_patch_space - l_core_in_patch_space

            # define bilayer part
            l_z_bilayer_in_patch_space = self._find_bilayer_points(l_boundary_point_in_patch_space, 0)
            l_y_bilayer_in_patch_space = self._find_bilayer_points(l_boundary_point_in_patch_space, 1)
            l_x_bilayer_in_patch_space = self._find_bilayer_points(l_boundary_point_in_patch_space, 2)

            # check validity
            is_valid = True
            if patch_space_volume_th > 0:
                is_valid &= (np.sum(l_core_in_patch_space) + len(l_z_bilayer_in_patch_space) +
                             len(l_y_bilayer_in_patch_space) + len(l_x_bilayer_in_patch_space)) >= patch_space_volume_th

            if patch_space_core_volume_th > 0:
                is_valid &= np.sum(l_core_in_patch_space) >= patch_space_core_volume_th

            # get and classify valid label coordinates
            l_coords_in_patch_space = np.vstack(np.where(l_in_patch_space == 1)).astype(np.uint8).T.tolist()
            l_core_coords_in_patch_space = np.vstack(np.where(l_core_in_patch_space == 1)).astype(np.uint8).T.tolist()

            #
            for i in range(len(l_coords_in_patch_space)):

                coord = l_coords_in_patch_space[i]
                is_core = (coord in l_core_coords_in_patch_space)
                is_bilayer = (coord in l_z_bilayer_in_patch_space +
                                       l_y_bilayer_in_patch_space +
                                       l_x_bilayer_in_patch_space)
                row = {'Z': coord[0],
                       'Y': coord[1],
                       'X': coord[2],
                       'label': int(l),
                       'core_point': is_core,
                       'bilayer_point': is_bilayer,
                       'z_bilayer': (coord in l_z_bilayer_in_patch_space),
                       'y_bilayer': (coord in l_y_bilayer_in_patch_space),
                       'x_bilayer': (coord in l_x_bilayer_in_patch_space),
                       'boundary_point': not (is_core or is_bilayer)}
                valid_labels_df = valid_labels_df.append(row, ignore_index=True)

            valid_labels_df = valid_labels_df.astype(int)

        # remove uncertain labeling of points in patch space
        for z in range(Zmax + 1):

            for y in range(Ymax + 1):

                for x in range(Xmax + 1):

                    tmp = valid_labels_df[
                        (valid_labels_df['Z'] == z) & (valid_labels_df['Y'] == y) & (valid_labels_df['X'] == x)]
                    if len(tmp) > 1:
                        valid_labels_df.drop(valid_labels_df[(valid_labels_df['Z'] == z) &
                                                           (valid_labels_df['Y'] == y) &
                                                           (valid_labels_df['X'] == x)].index, inplace=True)

        valid_labels_df.reset_index(inplace=True,drop=True)
        self.trace.__setattr__(self.output_dataset[0],valid_labels_df)
        print('>----------------<', end='\r')

        self._update_graph_dict()
        self._update_params_dict()
        if self.trace.enable_operations_tracking:

            self.trace.add_operation(ClusterValidator.__name__, self, self.pt.get_current_tracked_params())

        return self.output_dataset

    @staticmethod
    def _npidx(idx, axis, pos=None, n_dims=3):
        """
        Utility method. Produce the correct list of slice method in order to take the desired portion of an array
        by specifying axis, extrema and kind of interval required.

        :param idx: (list[int]) list defining the interval.
        :param axis: (int) axis along which the interval are defined.
        :param pos: (str) how the interval is defined of interval. It can be 'start','stop','step', 'start_stop',
                    'start_step','stop_step' depending on how the
        :param n_dims: (int) optional, number of dimension of the array to slice
        """
        index = n_dims * [slice(None, None, None)]
        if type(idx) is int:

            if pos is None:
                index[axis] = idx

            if pos == 'start':

                index[axis] = slice(idx, None, None)

            elif pos == 'stop':

                index[axis] = slice(None, idx, None)

            elif pos == 'step':

                index[axis] = slice(None, None, idx)

        elif type(idx) in [tuple, list]:

            if len(idx) == 2:

                if pos == 'start_stop' or pos is None:

                    index[axis] = slice(idx[0], idx[1], None)

                elif pos == 'start_step':

                    index[axis] = slice(idx[0], None, idx[1])

                elif pos == 'stop_step':

                    index[axis] = slice(None, idx[0], idx[1])

            elif len(idx) == 3:

                index[axis] = slice(idx[0], idx[1], idx[2])

        return tuple(index)

    def _find_bilayer_points(self,object_in_path_space, ax=0):
        """
        Find bilayer points.

        :param object_in_path_space: (ndarray) 3d array containing the cluster in patch space
        :param ax: (int) axis with respect to which the bilayer points are computed.
        """
        W1 = np.zeros(object_in_path_space.shape)
        W1[self._npidx(-1,ax,pos='stop')] = \
            (np.diff(object_in_path_space,axis=ax)==0)*object_in_path_space[self._npidx(-1,ax,pos='stop')]
        W2 = np.zeros(object_in_path_space.shape)
        W2[self._npidx(1,ax,pos='start')] = \
            (np.diff(object_in_path_space[self._npidx(-1,ax,pos='step')],axis=ax)[self._npidx(-1,ax,pos='step')]==0) * \
            object_in_path_space[self._npidx(1,ax,pos='start')]
        W = (W1 + W2 > 0).astype(np.uint8)
        bilayer_points_ax = []
        for i in range(object_in_path_space.shape[ax]):

            bilayer_points_ax.append(binary_erosion(W[self._npidx(i, ax)], border_value=1))
            # bilayer_points_ax.append(binary_erosion(object_in_path_space[self._npidx(i, ax)], border_value=1))

        bilayer_points_ax = np.array(bilayer_points_ax)
        return np.vstack(np.where(bilayer_points_ax == True)).T.tolist()

    def read(self, name=None, save_as='csv'):
        """
        Read operation results in the hdf5 file. For this plugin, the results are organized in the form of a pandas
        dataframe.

        :param save_as: (str) optional, format used to save the result. It can be 'csv' of 'json'.
        :param name: (str) optional, name of the result file.
        """
        tmp_df = self.trace.__getattribute__(self.output_dataset[0])
        trace_pos = self.trace.get_trace_pos()
        path_to_readings_folder = manage_path(self.trace.trace_readings_path() + os.sep +
                                              ClusterValidator.__name__ + '_{}'.format(trace_pos))
        if name is None:

            name = self.output_dataset[0]

        if save_as == 'csv':

            tmp_df.to_csv(path_to_readings_folder+os.sep+name+'.csv')

        elif save_as == 'json':

            tmp_df.to_json(path_to_readings_folder+os.sep+name+'.json')

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
                                   'edge': ClusterValidator.__name__}})
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
                previous_content[self.trace.group_name]['trace_graph'][str(key - 1)]['node'] == self.output_dataset:

            key = key - 1
            del params_dict[str(key)]

        params_dict.update({key: {'op': ClusterValidator.__name__,
                                  'p': self.pt.get_current_tracked_params()}})
        previous_content[self.trace.group_name]['ops_parameters'] = params_dict
        with open(self.trace.json_trace_path, 'w') as jfile:

            json.dump(previous_content, jfile, indent=4)

class ArchetypeIdentifier:
    """
    Operation used to generate a subset of patch belonging to a cluster useful for the study of the cluster properties.
    """

    __name__ = 'ArchetypeIdentifier'
    def __init__(self,trace):
        """
        Initialize the operation. The ArchetypeIdentifier operation is used to select a subset of patches from the
        input data, sampled from the most internal part of a cluster, which should in principle contain the most
        relevant features of a given cluster. The patches selected in this way are called in bmmltools Archetype
        of the cluster.

        :param trace: (bmmltools.core.tracer.Trace object) trace on which the operation act.
        """
        self.trace = trace
        self.output_dataset = ['post_'+get_capitals(ArchetypeIdentifier.__name__)+'_dataset']
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

    def apply(self,patch_shape,archetype_threshold=0.5,N_archetype = 50,extrapoints_per_dimension=(3,3,3),
              filter_by_column=None,save_archetype_mask=False):
        """
        Apply the operation to the trace. This operation select a set of patches in the most inner part of the cluster
        (inner in the sense of the patch space). The input data for this operation is a dataframe containing the
        description of the clusters in patch space (i.e. the 3 spatial coordinates in patch space, the label, other
        information about the nature of the point, if available).

        :param patch_space: (tuple[int]) shape of the patch used in this operation.
        :param archetype_threshold: (float between 0 and 1) optional, threshold on the normalized distance transform
                                    used to define the sampling region for the archetypes of a given cluster.
        :param N_archetype: (int) optional, number of archetypes sampled for each cluster.
        :param extrapoints_per_dimension: (tuple[int]) optional, dilation factor for each dimension of the
                                          patch space. This is used in order to have more resolution in the estimation
                                          of the normalized distance transform which determine the sampling region for
                                          the archetypes of each cluster.
        :param filter_by_column: (str or None) optional, when it is not None, it is the name of the column used to
                                 filter the points in the patch space. The point selected in this filtering operation
                                 are the one having 1 in the column specified here. If None is given, this step is
                                 ignored.
        :param save_archetype_mask: (bool) optional, if True the mask showing the region from which the archetype has
                                    been sampled is saved in the trace file folder.
        """
        if hasattr(self.trace,'seed'):

            np.random.seed(self.trace.seed)

        with self.pt:

            self.extrapoints_per_dimension = extrapoints_per_dimension
            self.patch_shape = patch_shape
            self.N_archetype = N_archetype
            self.archetype_threshold = archetype_threshold
            self.filter_by_column = filter_by_column
            self.save_archetype_mask = save_archetype_mask

        label_dataset_name,archetype_source_name = self.input_dataset

        print('>----------------<')
        print('Archetype identification')
        print('Preparing expanded patch space...',end='\r')
        with h5py.File(self.trace.hdf5_trace_path, 'r') as file:

            volume_shape = file[self.trace._hdf5_inner_path + archetype_source_name].shape

        labels_df = self.trace.__getattribute__(label_dataset_name)
        if self.filter_by_column is not None:

            try:

                labels_df = labels_df[labels_df[filter_by_column]==1]

            except:

                print('Warning: column \'{}\' not found in the input dataset to filter the dataset\'s rows. '
                      'All rows used instead'.format(label_dataset_name))
                print('Preparing expanded patch space...', end='\r')

        Zmax = max(labels_df['Z'])
        Zmin = min(labels_df['Z'])
        Ymax = max(labels_df['Y'])
        Ymin = min(labels_df['Y'])
        Xmax = max(labels_df['X'])
        Xmin = min(labels_df['X'])
        expanded_patch_space_shape = (extrapoints_per_dimension[0] * (Zmax - Zmin + 1),
                                      extrapoints_per_dimension[1] * (Ymax - Ymin + 1),
                                      extrapoints_per_dimension[2] * (Xmax - Xmin + 1))
        expanded_patch_space = -np.ones(expanded_patch_space_shape,dtype=np.int8)
        for k in range(expanded_patch_space_shape[0]):

            for j in range(expanded_patch_space_shape[1]):

                for i in range(expanded_patch_space_shape[2]):

                    z = k // extrapoints_per_dimension[0]
                    y = j // extrapoints_per_dimension[1]
                    x = i // extrapoints_per_dimension[2]
                    label_df = labels_df[(labels_df['Z'] == z) &
                                         (labels_df['Y'] == y) &
                                         (labels_df['X'] == x)]['label']
                    if len(label_df)>0:

                        label = int(label_df)
                        expanded_patch_space[k, j, i] = label

        print('Preparing expanded patch space...[OK]')

        Zs = []
        Ys = []
        Xs = []
        labels = []
        archetypes = []
        with h5py.File(self.trace.hdf5_trace_path, 'r') as file:

            all_labels = list(set(labels_df['label'].to_list()))
            for n,L in enumerate(all_labels):

                print('Generating archetypes...[labels {}/{}]'.format(n+1,len(all_labels)),end='\r')
                dt_expanded_patch_space = distance_transform_edt(expanded_patch_space == L)
                archetype_zone = (
                            dt_expanded_patch_space > np.max(dt_expanded_patch_space) * archetype_threshold).astype(int)
                p = archetype_zone / np.sum(archetype_zone)
                p = p.flatten()
                labels += N_archetype * [L, ]
                Zs_loc = []
                while True:

                    idx = np.random.choice(np.arange(0, p.shape[0]), p=p)
                    idx = np.unravel_index(idx, shape=expanded_patch_space_shape)
                    idx = tuple(
                        np.ceil(np.array(idx) / np.array(expanded_patch_space_shape) * np.array(volume_shape)).astype(
                            int))
                    zmin = idx[0] - patch_shape[0] // 2
                    zmax = zmin + patch_shape[0]
                    ymin = idx[1] - patch_shape[1] // 2
                    ymax = ymin + patch_shape[1]
                    xmin = idx[2] - patch_shape[2] // 2
                    xmax = xmin + patch_shape[2]
                    if zmax <= volume_shape[0] and zmin >= 0 and ymax <= volume_shape[1] and ymin >= 0 and \
                            xmax <= volume_shape[2] and xmin >= 0:

                        Zs_loc.append(zmin)
                        Ys.append(ymin)
                        Xs.append(xmin)
                        archetypes.append(
                            file[self.trace._hdf5_inner_path + archetype_source_name][zmin:zmax, ymin:ymax, xmin:xmax])

                    if len(Zs_loc) == N_archetype:

                        break

                Zs += Zs_loc
        print('Generating archetypes...[OK]')

        print('Writing result on trace...',end='')
        results = {'archetype_patch_coordinates': pd.DataFrame({'Z': Zs, 'Y': Ys, 'X': Xs, 'label': labels}),
                   'archetypes': np.array(archetypes)}
        self.trace.__setattr__(self.output_dataset[0],results)
        print('[OK]')
        if self.save_archetype_mask:

            print('Saving archetypes masks...',end='')
            self._save_mask(results,patch_shape,volume_shape)
            print('[OK]')

        print('>----------------<',end='\r')

        self._update_graph_dict()
        self._update_params_dict()
        if self.trace.enable_operations_tracking:

            self.trace.add_operation(ArchetypeIdentifier.__name__,self,self.pt.get_current_tracked_params())

        return self.output_dataset

    def _save_mask(self,results,patch_shape,volume_shape):
        """
        Save the mask indicating where the archetypes has been sampled.

        :param results: (pandas.DataFrame) dataframe containing the archetype coordinates for each cluster.
        :param patch_shape: (tuple[int]) shape of the patch used to produce the archetypes.
        :param volume_shape: (tuple[int]) shape of the input data from which the archetypes has been sampled.
        """
        trace_pos = self.trace.get_trace_pos()
        saving_folder = manage_path(self.trace.trace_file_path() + os.sep +
                                    ArchetypeIdentifier.__name__ + '_{}'.format(trace_pos))
        for L in list(set(results['archetype_patch_coordinates']['label'].to_list())):

            df = results['archetype_patch_coordinates'][results['archetype_patch_coordinates']['label'] == L]
            Zs_L = df.loc[:, 'Z'].to_list()
            Ys_L = df.loc[:, 'Y'].to_list()
            Xs_L = df.loc[:, 'X'].to_list()
            mask = np.zeros(volume_shape, dtype=bool)
            for k, j, i in zip(Zs_L, Ys_L, Xs_L):

                mask[k:k + patch_shape[0], j:j + patch_shape[1], i:i + patch_shape[2]] |= True

            np.save(saving_folder + os.sep + 'label_{}_archetype_mask.npy'.format(L),mask)

    def read(self):
        """
        Read operation results in the hdf5 file. For this plugin, the results are organized in the form of a pandas
        dataframe (coordinates in patch space of the archetype) and numpy array (the archetype). The first is saved as
        csv file, while the second as npy file.
        """
        trace_pos = self.trace.get_trace_pos()
        path_to_readings_folder = manage_path(self.trace.trace_readings_path() + os.sep +
                                              ArchetypeIdentifier.__name__ + '_{}'.format(trace_pos))
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
                                  'edge': ArchetypeIdentifier.__name__}})
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

        params_dict.update({key: {'op': ArchetypeIdentifier.__name__,
                                  'p': self.pt.get_current_tracked_params()}})
        previous_content[self.trace.group_name]['ops_parameters'] = params_dict
        with open(self.trace.json_trace_path, 'w') as jfile:

            json.dump(previous_content, jfile, indent=4)

class RotationalSimilarityIdentifier:
    """
    Operation used to suggest possible identification among cluster based on their similarity under rotation.
    """

    __name__ = 'RotationalSimilarityIdentifier'
    def __init__(self, trace):
        """
        Initialize the operation. The RotationalSimilarityIdentifier is the operation used to suggest a possible
        identification of two cluster based on their similarity under rotation.

        :param trace: (bmmltools.core.tracer.Trace object) trace on which the operation act.
        """
        self.trace = trace
        self.output_dataset = ['post_' + get_capitals(RotationalSimilarityIdentifier.__name__) + '_dataset']
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

    def apply(self ,p_threshold = 0.6 ,smooth = False ,sigma = 0.5 ,spherical_coordinates_shape = (25 ,18 ,36),
              bin_used_in_radial_dist = (1,)):
        """
        Apply the operation to the trace. This operation determine based on a set of archetype for each cluster, if
        two or more of them can be identified based on similarities among their radial distributions. The identification
        is done by performing the one sample Hotelling t2 test in simmetrized manner, and with a check in transifivity
        when more than two clusters is involved.

        :param p_threshold: (float between 0 and 1) optional, threshold on the identification probability above which
                            two clusters may be identified.
        :param smooth: (bool) optional, if True the modulus of the pDFTs are smoothed with a gaussian before the
                       computation of the radial distribution.
        :param sigma: (float) optional, when 'smooth' is True, this is the standard deviation of the gaussian used
                      for the smoothing.
        :param spherical_coordinates_shape: (tuple[int]) optional, shape of the archetype in when represented in
                                            spherical coordinates. The shape entries are the following:

                                                        (N radii, N theta angles, N phi anges)

        :param bin_used_in_radial_dist: (tuple[int]) optional, bins of the radial distribution used for the analysis.
                                        It has to be specified as follow:

                                                       (position of the lower bin, position of the upper bin).

                                        The bin considered are the one between the lower and the upp. If nothing is
                                        specified in the first or second entry, the smaller of larger bin possible is
                                        assumed.
        """
        with self.pt:

            self.p_threshold = p_threshold
            self.smooth = smooth
            self.sigma = sigma
            self.spherical_coordinates_shape = spherical_coordinates_shape
            self.bin_used_in_radial_dist = bin_used_in_radial_dist

        print('>----------------<')
        print('Labels identification for rotational similarity')

        # get inputs dataset
        self.archetype_dict = self.trace.__getattribute__(self.input_dataset[0])
        self.label_df = self.trace.__getattribute__(self.input_dataset[1])

        # setup
        self.archetypes_df = self.archetype_dict['archetype_patch_coordinates']
        self.archetypes_patches = self.archetype_dict['archetypes']
        self.patch_shape = self.archetypes_patches.shape[1:]
        self.label_list = list(set(self.archetypes_df.loc[:,'label'].to_list()))

        # initialize change of coordinate tools
        self.CtoS = ChangeCoordinateSystem(**{'reference_frame_origin': np.array(self.patch_shape) // 2,
                                         'xyz_to_XYZ_inv_map': spherical_to_cartesian,
                                         'xyz_to_XYZ_specs': {'new_shape': self.spherical_coordinates_shape,
                                                              'X_bounds': [0, self.patch_shape[0] // 2],
                                                              'Y_bounds': [0, 180],
                                                              'Z_bounds': [0, 360],
                                                              'XYZ_ordering': [0, 1, 2]}})

        # compute equivalence relation from rotational similarity of the archetypes
        results_dict, intermediate_results = self._compute_archetype_and_stats()
        results_dict,equality_prob = self._compute_identification_probabilities(intermediate_results,results_dict)
        results_dict = self._compute_ll_max_correlation_angles(results_dict)
        results_dict, identification_dict = self._get_equivalence_relation(equality_prob,results_dict)

        # save results: 1 save results_dict
        self.trace.__setattr__(self.output_dataset[0],results_dict)

        # save result: 2 update label_df
        self._update_label_dataframe(identification_dict)
        self.trace.__setattr__(self.input_dataset[1],self.label_df)

        print('>----------------<', end='\r')

        self._update_graph_dict()
        self._update_params_dict()
        if self.trace.enable_operations_tracking:

            self.trace.add_operation(RotationalSimilarityIdentifier.__name__, self, self.pt.get_current_tracked_params())

        return self.output_dataset

    def _compute_archetype_and_stats(self):
        """
        Compute the radial distributions, their mean values and covariance matrix for all the archetypes.

        :return: (dict,dict) dictionaries containing the mean archetype, mean radial distributions, and some
                 intermediate result.
        """
        n_radii ,n_thetas ,n_phis = self.spherical_coordinates_shape
        _, Thetas, _ = np.mgrid[:n_radii, :n_thetas, :n_phis]

        results_dict = {}
        intermediate_results = {}
        for L in self.label_list:

            L_index_list = self.archetypes_df[self.archetypes_df['label'] == L].index.to_list()
            mdfts = []
            radial_distributions = []
            for idx in L_index_list:

                patch = self.archetypes_patches[idx, ...]
                mdft_patch = np.abs(1 / (np.sqrt(np.prod(patch.shape))) *
                                    np.fft.fftshift(periodic_smooth_decomposition_dft3(patch)[0]))
                if self.smooth:

                    mdft_patch = gaussian_filter(mdft_patch,self.sigma)

                mdfts.append(mdft_patch)
                mdft_patch_spherical = self.CtoS.transform(mdft_patch)
                rd_mdft_patch = np.sum(mdft_patch_spherical * np.sin(grad_to_rad(Thetas)),
                                       axis=(1, 2))  # radial integration
                radial_distributions.append(rd_mdft_patch)

            mean_mdft = np.mean(np.array(mdfts), axis=0)
            results_dict.update({'archetype_label_{}_mod_dft3d'.format(L): mean_mdft})

            mean_radial_distribution = np.mean(np.array(radial_distributions), axis=0)
            cov_radial_distribution = np.cov(np.array(radial_distributions).T)
            intermediate_results.update({'Label_{}'.format(L): {'mean_rd': mean_radial_distribution,
                                                                'cov_rd': cov_radial_distribution}})
            results_dict.update({'archetype_label_{}_mean_radial_distribution'.format(L): mean_radial_distribution})

        self.N_archetypes = len(L_index_list)
        return results_dict,intermediate_results

    @staticmethod
    def _hotellig_t2_test(X_bar, covX, Y_bar, N_x, x_TH):
        """
        One-way Hotelling t2-test.

        :param X_bar: (ndarray) mean value of the sample under analysis.
        :param covX: (naddary) covariance matrix of the samples under analysis
        :param Y_bar: (ndarray) reference mean value used in the test.
        :parma N_x: (int) number of sample used for the analysis.
        :param x_TH: (float) acceptance threshold for the test (used to compute the statistical power).
        :return: (float,float) identification probability and statistical power of the test.
        """
        p = len(X_bar)
        covX_inv = np.linalg.inv(covX)
        t_square = np.dot(X_bar - Y_bar, np.dot(covX_inv, X_bar - Y_bar))
        x = (N_x - p) / (p * (N_x - 1)) * t_square

        p_id = f_distribution.sf(x, p, N_x - p)
        power = non_centred_f_distribution.sf(x_TH, p, N_x - p, t_square)
        return p_id, power

    @staticmethod
    def _simmetrize_min_values(x):
        """
        Utility function. Used to compute the minimum entries among the upper and lower diagonal part of a matrix.

        :param x: (ndarray) matrix.
        """
        ut_x = np.triu(x)
        lt_x = np.tril(x)
        x = np.minimum(ut_x, lt_x.T)
        return x + np.triu(x, k=1).T

    def _compute_identification_probabilities(self,intermediate_results,results_dict):
        """
        Compute the identification probabilities among clusters.

        :param intermediate_results:
        :param results_dict:
        """
        n = self.bin_used_in_radial_dist[0]
        m = self.spherical_coordinates_shape[0]
        if len(self.bin_used_in_radial_dist) == 2:

            m = np.minimum(m,self.bin_used_in_radial_dist[1])

        p = m - n
        x_TH = f_distribution.isf(self.p_threshold, p, self.N_archetypes - p)
        equality_prob = np.zeros(2 * (len(self.label_list),))
        statistical_power = np.zeros(2 * (len(self.label_list),))
        for i, L0 in enumerate(self.label_list):

            Pr = []
            Po = []
            for j, L1 in enumerate(self.label_list):

                if L0 != L1:

                    X_bar = intermediate_results['Label_{}'.format(L0)]['mean_rd'][n:m]
                    covX = intermediate_results['Label_{}'.format(L0)]['cov_rd'][n:m, n:m]
                    Y_bar = intermediate_results['Label_{}'.format(L1)]['mean_rd'][n:m]

                    p_id, power = self._hotellig_t2_test(X_bar, covX, Y_bar, self.N_archetypes,x_TH)

                    Pr.append(p_id)
                    Po.append(power)

                else:

                    Pr.append(1)
                    Po.append(1)

            equality_prob[i, :] = np.array(Pr)
            statistical_power[i, :] = np.array(Po)

        equality_prob = self._simmetrize_min_values(equality_prob)
        equality_prob_df = self._label_square_array_to_df(equality_prob,
                                                          ['Label_{}'.format(L) for L in self.label_list])
        statistical_power = self._simmetrize_min_values(statistical_power)
        statistical_power_df = self._label_square_array_to_df(statistical_power,
                                                              ['Label_{}'.format(L) for L in self.label_list])
        results_dict.update({'identification_probability': equality_prob_df,
                             'test_statistical_power': statistical_power_df})
        return results_dict,equality_prob

    def _compute_ll_max_correlation_angles(self,results_dict):
        """
        Compute the Label-Label angles by looking at the maximum of the (weighted) correlation in spherical coordinates.

        :param results_dict: (dict) dictionary containing the modulus of the mean pDFT of the  archetypes.
        :return: (dict) updated result dictionary with the angles
        """
        n_radii, n_thetas, n_phis = self.spherical_coordinates_shape
        angles_tensor = np.zeros(2*(len(self.label_list),) + (2,))
        for i in range(len(self.label_list)):

            for j in range(i + 1, len(self.label_list)):

                L0 = self.label_list[i]
                x = self.CtoS.transform(results_dict['archetype_label_{}_mod_dft3d'.format(L0)])
                L1 = self.label_list[j]
                y = self.CtoS.transform(results_dict['archetype_label_{}_mod_dft3d'.format(L1)])

                # best matching angle
                correlations = []
                weights = []
                for i_r in range(1, x.shape[0]):

                    x_padded = np.hstack([x[i_r], x[i_r], x[i_r]])
                    x_padded = np.hstack(
                        [x_padded.T, x_padded.T, x_padded.T]).T  # pad x to simulate periodicity in the center
                    tmp = match_template(x_padded, y[i_r])
                    correlations.append(tmp)

                    gx, gy = np.gradient(x[i_r])
                    weights.append(np.sqrt(np.mean(gx ** 2 + gy ** 2)))

                weights = np.array(weights) / sum(weights)
                correlations = np.array(correlations)[:, n_thetas // 2:n_thetas // 2 + n_thetas,
                               n_phis // 2:n_phis // 2 + n_phis]  # consider only the center (i.e. eliminate padding 'artefacts')
                total_correlation = (np.expand_dims(weights, axis=(1, 2)) * correlations).sum(axis=0)
                theta, phi = np.squeeze(np.argwhere(total_correlation == np.max(total_correlation)))
                theta_grad = (180 // n_thetas) * theta - 90
                phi_grad = (360 // n_phis) * phi - 180
                angles_tensor[i, j, :] = angles_tensor[j, i, :] = np.array([theta_grad, phi_grad])

        ll_theta_angles_df = self._label_square_array_to_df(angles_tensor[:, :, 0],
                                                            ['Label_{}'.format(L) for L in self.label_list])
        ll_phi_angles_df = self._label_square_array_to_df(angles_tensor[:, :, 1],
                                                          ['Label_{}'.format(L) for L in self.label_list])
        results_dict.update({'ll_theta_angles_df': ll_theta_angles_df,
                             'll_phi_angles_df': ll_phi_angles_df})
        return results_dict

    @staticmethod
    def _search_couple(items_list, item):
        """
        Utility function.

        :param items_list:
        :param item:
        :return:
        """
        i = 0
        for el in items_list:

            if el == item:

                return i

            i = i + 1

        return -1

    @staticmethod
    def _is_transitive(relation):
        """
        Check if a symmetric binary relation is transitive on a set. The relation is expressed as a list of tuples,
        indicating the relation between two elements of a set.

        Example:    - relation = [(1,2),(3,4),(1,0)], which express a binary relation on the set {0,1,2,3,4}.

                    - the couple (1,2) means that the element 1 is in relation with the element 2.

        The check is done by assuming that the relation span a CONNECTED GRAPH: transitivity holds if the graph expressed
        by the relation is TRIVIAL (all the vertex are connected with each other). In practice, the adjacenty matrix of the
        graph expressed by the relation is checked.

        :param relation: (list[tuples[int]]) the symmetric binaray relation to check.
        :return: (boolean) True if the relation is transitive.
        """
        vertex_list = list(set(np.array(relation).flatten()))
        N_vertex = len(vertex_list)

        translator = {i: n for n, i in enumerate(vertex_list)}
        adjacency_matrix = np.zeros((N_vertex, N_vertex))
        for couple in relation:

            adjacency_matrix[translator[couple[0]], translator[couple[1]]] = 1

        trivial_graph_adjacency_matrix = np.triu(np.ones((N_vertex, N_vertex)), 1)
        return np.prod(adjacency_matrix == trivial_graph_adjacency_matrix) == 1

    @staticmethod
    def _get_binary_relation_from_mask(mask):
        """
        Derive a binary relation given a mask.

        :param mask: (ndarray) identification matrix.
        :return: (list[tuple[int]]) binary relation (see '_is_transitive' for more information).
        """
        mask = np.triu(mask, k=1).astype(bool)
        return [tuple(pair) for pair in np.array(np.where(mask == True)).T]

    @staticmethod
    def _split_in_subgroups(relation):
        """
        Identify the connected components of a binary relation.

        :param relation: (list[tuples[int]]) the symmetric binary relation  (see '_is_transitive' for more information).
        :return: the connected component of the binary relation analysed.
        """
        to_analyse = copy.copy(relation)
        to_analyse = np.array(to_analyse)
        all_involved_vertex = set(to_analyse.flatten())
        subgroups = []
        while True:

            result = [list(all_involved_vertex)[0]]
            Lres = len(result)
            Lres_set = len(result)
            delta_Lres = 1
            while True:

                Lres_set0 = Lres_set
                Lres0 = Lres
                for i in range(Lres0 - delta_Lres, Lres0):
                    r = result[i]
                    tmp = list(to_analyse[np.sum(to_analyse == r, axis=1).astype(bool)].flatten())
                    result = result + tmp

                Lres = len(result)
                Lres_set = len(list(set(result)))
                delta_Lres = Lres - Lres0
                delta_Lres_set = Lres_set - Lres_set0
                if not delta_Lres_set > 0:
                    break

            subgroups.append(list(set(result)))
            all_involved_vertex = all_involved_vertex.difference(set(result))
            if len(all_involved_vertex) == 0:
                break

        return subgroups

    def _produce_identification_dictionary(self,relation, subgroups, score_list):
        """
        Check transitivity of the candidate binary relation, by checking transitivity.

        :param relation: (list[tuple[int]])
        :param subgroups: ()
        :param score_list: (list[float]) score to check if transitivity is lacking (indentification probability in
                           this case)
        :return: (dict) identification dictionary.
        """
        super_groups = []
        super_groups_score = []
        for grouping in subgroups:

            super_group = []
            super_group_score = []
            for couple in relation:

                if couple[0] in grouping and couple[1] in grouping:

                    super_group.append(couple)
                    super_group_score.append(score_list[self._search_couple(relation, couple)])

            super_groups.append(super_group)
            super_groups_score.append(super_group_score)

        # check transitivity of each 'connected identification' and act consequentially
        identification_dict = {}
        for sg, sg_score in zip(super_groups, super_groups_score):

            if self._is_transitive(sg):

                tmp_id_list = list(set(np.array(sg).flatten()))
                sorted(tmp_id_list)
                identification_dict.update({tmp_id_list[0]: tmp_id_list[1:]})

            else:

                best = sg[np.argmin(sg_score)]
                key = min(best)
                val = [max(best)]
                identification_dict.update({key: val})  # here criteria to chose the best identification

        return identification_dict

    def _get_equivalence_relation(self,equality_prob,results_dict):
        """
        Organize the identification results.

        :param equality_prob: (ndarray) identification probabilities.
        :param results_dict: (dict) dictionary containing the identification results.
        :return: (dict,dict)
        """
        rotational_similarity_mask = equality_prob > self.p_threshold
        candidate_bin_relations = self._get_binary_relation_from_mask(rotational_similarity_mask)
        if len(candidate_bin_relations) > 0:

            bin_relations_subgroups = self._split_in_subgroups(candidate_bin_relations)
            bin_relation_scores = [equality_prob[couple[0], couple[1]] for couple in candidate_bin_relations]
            identification_dict = self._produce_identification_dictionary(candidate_bin_relations,
                                                                         bin_relations_subgroups,
                                                                         bin_relation_scores)
        else:

            identification_dict = {}

        equivalence_relation = np.eye(equality_prob.shape[0])
        for k in identification_dict:

            for j in identification_dict[k]:

                line = copy.copy(identification_dict[k])
                equivalence_relation[k, j] = equivalence_relation[j, k] = 1
                if len(line) > 1:

                    if j in line:

                        del line[line.index(j)]

                    for i in line:

                        equivalence_relation[i, j] = equivalence_relation[j, i] = 1

        identification_df = self._label_square_array_to_df(equivalence_relation,
                                                           ['Label_{}'.format(L) for L in self.label_list])
        results_dict.update({'identification_df': identification_df})
        return results_dict, identification_dict

    def _update_label_dataframe(self,identification_dict):
        """
        Update the dataframe produced by the ClusterValidator with the suggested identification (the column 'RS_label'
        is added)

        :param identification_dict: (dict) dictionary containing the suggested identifications.
        """
        new_column = []
        for i in range(len(self.label_df)):

            l = self.label_df.loc[i, 'label']
            flag = False
            for key in identification_dict:

                identification_map = identification_dict[key]
                if l in identification_map:

                    new_column.append(key)
                    flag = True

            if not flag:

                new_column.append(l)

        self.label_df['RS_label'] = new_column

    @staticmethod
    def _label_square_array_to_df(arr,names_list):
        """
        Organized the result in order to be saved on the trace.

        :param arr: (ndarray) array to save.
        :param names_list: (list[str]) name of the column entries of the array to save.
        :return: (pandas.DataFrame) dataframe used to save the result.
        """
        df = pd.DataFrame(arr)
        df.columns = names_list
        df['index'] = names_list
        return df.set_index('index')

    def read(self):
        """
        Read operation results in the hdf5 file. For this plugin, the results are organized in the form of a pandas
        dataframe and numpy array. The first is saved as csv file, while the second as npy file.
        """
        trace_pos = self.trace.get_trace_pos()
        path_to_readings_folder = manage_path(self.trace.trace_readings_path() + os.sep +
                                              RotationalSimilarityIdentifier.__name__ + '_{}'.format(trace_pos))
        res = self.trace.__getattribute__(self.output_dataset[0])
        for key in res.keys():

            if type(res[key]) == pd.DataFrame:

                res[key].to_csv(path_to_readings_folder + os.sep + key + '.csv')

            else:

                np.save(path_to_readings_folder + os.sep + key + '.npy', res[key])

        dataset_name = None
        if hasattr(self,'input_dataset'):

            dataset_name = self.input_dataset[1]

        elif len(self.output_dataset) > 1:

            dataset_name = self.output_dataset[1]

        if dataset_name is not None:

            res2 = self.trace.__getattribute__(dataset_name)
            res2.to_csv(path_to_readings_folder+os.sep+dataset_name+'.csv')

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
                                   'edge': RotationalSimilarityIdentifier.__name__}})
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
                previous_content[self.trace.group_name]['trace_graph'][str(key - 1)]['node'] == self.output_dataset:
            key = key - 1

            del params_dict[str(key)]

        params_dict.update({key: {'op': RotationalSimilarityIdentifier.__name__,
                                  'p': self.pt.get_current_tracked_params()}})
        previous_content[self.trace.group_name]['ops_parameters'] = params_dict
        with open(self.trace.json_trace_path, 'w') as jfile:

            json.dump(previous_content, jfile, indent=4)


########


class ClusterValidator2:
    """
    Operation used to validate from a geometric point of view the cluster found by a clustering algorithm.
    """

    __name__ = 'ClusterValidator'
    def __init__(self, trace):
        """
        Initialize the operation. The ClustererValidator operation is used to evaluate if the clusters found are
        reasonable or not using purely geometrical considerations. What is checked is:

        * sufficient degree of continuity in patch space;

        * sufficient volume in patch space.

        :param trace: (bmmltools.core.tracer.Trace object) trace on which the operation act.
        """
        self.trace = trace
        self.output_dataset = ['post_' + get_capitals(ClusterValidator.__name__) + '_dataset']
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

    def apply(self, patch_space_volume_th=1, patch_space_core_volume_th=1, criteria='3d'):
        """
        Apply the operation to the trace. Apply the validation procedure to a set of clusters.

        :param patch_space_volume_th: (int) optional, threshold on the volume in patch space below witch a cluster is
                                      eliminated.
        :param patch_space_core_volume_th: (int) optional, threshold on the volume of the core part of a cluster in
                                           patch space below witch a cluster is eliminated.
        :param criteria: (str) optional, it can be '3d' or '2d'. Here a specified the kind of validation procedure
                         to use (for dataset that are "2d-like" use '2d').
        """
        with self.pt:

            self.patch_space_volume_th = patch_space_volume_th
            self.patch_space_core_volume_th = patch_space_core_volume_th
            self.criteria = criteria

        print('>----------------<')
        print('Valid cluster identification')

        #
        result_df = self.trace.__getattribute__(self.input_dataset[0])

        #
        labels = list(set(result_df.loc[:, 'label'].to_list()))
        Zmax = max(result_df.loc[:, 'Z'])
        Ymax = max(result_df.loc[:, 'Y'])
        Xmax = max(result_df.loc[:, 'X'])
        labels_in_patch_space = -2 * np.ones((Zmax + 1, Ymax + 1, Xmax + 1))
        for i in range(len(result_df)):

            z, y, x, l = result_df.iloc[i].to_list()
            labels_in_patch_space[z, y, x] = l

        valid_labels_df = pd.DataFrame()
        for l in labels:

            l_in_patch_space = (labels_in_patch_space == l)

            # check label spatial continuity and define core part
            l_in_patch_space = binary_erosion(l_in_patch_space, border_value=True)
            l_in_patch_space = binary_dilation(l_in_patch_space, border_value=False)
            l_in_patch_space = binary_fill_holes(l_in_patch_space).astype(np.uint8)
            if self.criteria == '2d':

                l_core_in_patch_space = np.zeros(labels_in_patch_space.shape, dtype=np.uint8)
                for k in range(labels_in_patch_space.shape[0]):

                    l_core_in_patch_space[k, ...] = binary_erosion(l_in_patch_space[k, ...], border_value=True)

            else:


                l_core_in_patch_space = binary_erosion(l_in_patch_space, border_value=True).astype(np.uint8)

            l_boundary_point_in_patch_space = l_in_patch_space - l_core_in_patch_space

            if self.criteria == '2d':

                l_z_bilayer_in_patch_space = l_y_bilayer_in_patch_space = l_x_bilayer_in_patch_space = []
                # l_z_bilayer_in_patch_space = []
                # l_y_bilayer_in_patch_space = self._find_bilayer_points(l_boundary_point_in_patch_space, 1)
                # l_x_bilayer_in_patch_space = self._find_bilayer_points(l_boundary_point_in_patch_space, 2)

            else:

                # define bilayer part
                l_z_bilayer_in_patch_space = self._find_bilayer_points(l_boundary_point_in_patch_space, 0)
                l_y_bilayer_in_patch_space = self._find_bilayer_points(l_boundary_point_in_patch_space, 1)
                l_x_bilayer_in_patch_space = self._find_bilayer_points(l_boundary_point_in_patch_space, 2)

            # check validity
            is_valid = True
            if patch_space_volume_th > 0:
                is_valid &= (np.sum(l_core_in_patch_space) + len(l_z_bilayer_in_patch_space) +
                             len(l_y_bilayer_in_patch_space) + len(l_x_bilayer_in_patch_space)) >= patch_space_volume_th

            if patch_space_core_volume_th > 0:
                is_valid &= np.sum(l_core_in_patch_space) >= patch_space_core_volume_th

            # get and classify valid label coordinates
            l_coords_in_patch_space = np.vstack(np.where(l_in_patch_space == 1)).astype(np.uint8).T.tolist()
            l_core_coords_in_patch_space = np.vstack(np.where(l_core_in_patch_space == 1)).astype(np.uint8).T.tolist()

            #
            for i in range(len(l_coords_in_patch_space)):
                coord = l_coords_in_patch_space[i]
                is_core = (coord in l_core_coords_in_patch_space)
                is_bilayer = (coord in l_z_bilayer_in_patch_space +
                              l_y_bilayer_in_patch_space +
                              l_x_bilayer_in_patch_space)
                row = {'Z': coord[0],
                       'Y': coord[1],
                       'X': coord[2],
                       'label': int(l),
                       'core_point': is_core,
                       'bilayer_point': is_bilayer,
                       'z_bilayer': (coord in l_z_bilayer_in_patch_space),
                       'y_bilayer': (coord in l_y_bilayer_in_patch_space),
                       'x_bilayer': (coord in l_x_bilayer_in_patch_space),
                       'boundary_point': not (is_core or is_bilayer)}
                valid_labels_df = valid_labels_df.append(row, ignore_index=True)

            valid_labels_df = valid_labels_df.astype(int)

        # remove uncertain labeling of points in patch space
        for z in range(Zmax + 1):

            for y in range(Ymax + 1):

                for x in range(Xmax + 1):

                    tmp = valid_labels_df[
                        (valid_labels_df['Z'] == z) & (valid_labels_df['Y'] == y) & (valid_labels_df['X'] == x)]
                    if len(tmp) > 1:
                        valid_labels_df.drop(valid_labels_df[(valid_labels_df['Z'] == z) &
                                                             (valid_labels_df['Y'] == y) &
                                                             (valid_labels_df['X'] == x)].index, inplace=True)

        valid_labels_df.reset_index(inplace=True, drop=True)
        self.trace.__setattr__(self.output_dataset[0], valid_labels_df)
        print('>----------------<', end='\r')

        self._update_graph_dict()
        self._update_params_dict()
        if self.trace.enable_operations_tracking:
            self.trace.add_operation(ClusterValidator.__name__, self, self.pt.get_current_tracked_params())

        return self.output_dataset

    @staticmethod
    def _npidx(idx, axis, pos=None, n_dims=3):
        """
        Utility method. Produce the correct list of slice method in order to take the desired portion of an array
        by specifying axis, extrema and kind of interval required.

        :param idx: (list[int]) list defining the interval.
        :param axis: (int) axis along which the interval are defined.
        :param pos: (str) how the interval is defined of interval. It can be 'start','stop','step', 'start_stop',
                    'start_step','stop_step' depending on how the
        :param n_dims: (int) optional, number of dimension of the array to slice
        """
        index = n_dims * [slice(None, None, None)]
        if type(idx) is int:

            if pos is None:
                index[axis] = idx

            if pos == 'start':

                index[axis] = slice(idx, None, None)

            elif pos == 'stop':

                index[axis] = slice(None, idx, None)

            elif pos == 'step':

                index[axis] = slice(None, None, idx)

        elif type(idx) in [tuple, list]:

            if len(idx) == 2:

                if pos == 'start_stop' or pos is None:

                    index[axis] = slice(idx[0], idx[1], None)

                elif pos == 'start_step':

                    index[axis] = slice(idx[0], None, idx[1])

                elif pos == 'stop_step':

                    index[axis] = slice(None, idx[0], idx[1])

            elif len(idx) == 3:

                index[axis] = slice(idx[0], idx[1], idx[2])

        return tuple(index)

    def _find_bilayer_points(self, object_in_path_space, ax=0):
        """
        Find bilayer points.

        :param object_in_path_space: (ndarray) 3d array containing the cluster in patch space
        :param ax: (int) axis with respect to which the bilayer points are computed.
        """
        W1 = np.zeros(object_in_path_space.shape)
        W1[self._npidx(-1, ax, pos='stop')] = \
            (np.diff(object_in_path_space, axis=ax) == 0) * object_in_path_space[self._npidx(-1, ax, pos='stop')]
        W2 = np.zeros(object_in_path_space.shape)
        W2[self._npidx(1, ax, pos='start')] = \
            (np.diff(object_in_path_space[self._npidx(-1, ax, pos='step')], axis=ax)[
                 self._npidx(-1, ax, pos='step')] == 0) * \
            object_in_path_space[self._npidx(1, ax, pos='start')]
        W = (W1 + W2 > 0).astype(np.uint8)
        bilayer_points_ax = []
        for i in range(object_in_path_space.shape[ax]):
            bilayer_points_ax.append(binary_erosion(W[self._npidx(i, ax)], border_value=1))
            # bilayer_points_ax.append(binary_erosion(object_in_path_space[self._npidx(i, ax)], border_value=1))

        bilayer_points_ax = np.array(bilayer_points_ax)
        return np.vstack(np.where(bilayer_points_ax == True)).T.tolist()

    def read(self, name=None, save_as='csv'):
        """
        Read operation results in the hdf5 file. For this plugin, the results are organized in the form of a pandas
        dataframe.

        :param save_as: (str) optional, format used to save the result. It can be 'csv' of 'json'.
        :param name: (str) optional, name of the result file.
        """
        tmp_df = self.trace.__getattribute__(self.output_dataset[0])
        trace_pos = self.trace.get_trace_pos()
        path_to_readings_folder = manage_path(self.trace.trace_readings_path() + os.sep +
                                              ClusterValidator.__name__ + '_{}'.format(trace_pos))
        if name is None:
            name = self.output_dataset[0]

        if save_as == 'csv':

            tmp_df.to_csv(path_to_readings_folder + os.sep + name + '.csv')

        elif save_as == 'json':

            tmp_df.to_json(path_to_readings_folder + os.sep + name + '.json')

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
                                   'edge': ClusterValidator.__name__}})
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
                previous_content[self.trace.group_name]['trace_graph'][str(key - 1)]['node'] == self.output_dataset:
            key = key - 1
            del params_dict[str(key)]

        params_dict.update({key: {'op': ClusterValidator.__name__,
                                  'p': self.pt.get_current_tracked_params()}})
        previous_content[self.trace.group_name]['ops_parameters'] = params_dict
        with open(self.trace.json_trace_path, 'w') as jfile:
            json.dump(previous_content, jfile, indent=4)
