# Title: 'segmentation.py'
# Author: Curcuraci L.
# Date: 18/10/2020
#
# Scope: supervised segmentation of 3d binary images.

"""
Operations used to perform segmentation at voxel level using supervised methods.
"""

#################
#####   LIBRARIES
#################


import numpy as np
import h5py
import tempfile
import os
import json
import joblib

from skimage import feature, future
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from skimage import filters
from itertools import combinations_with_replacement

from bmmltools.utils.basic import manage_path, ParametersTracker, get_capitals
from bmmltools.utils.io_utils import write_rrn


###############
#####   CLASSES
###############


class RandomForestSegmenter:
    """
    Segmentation based on supervised training of a Random Forest classifiers.
    """

    __name__ = 'RandomForestSegmenter'
    def __init__(self, trace):
        """
        Initialize the operation. The RandomForestSegmenter operation train from a standard set of features extracted
        from an input binary image a random forest classifier to recognize a certain label, defined from some
        clustering. The classifier is applied then to each voxel of the input image.

        :param trace: (bmmltools.core.tracer.Trace object) trace on which the operation act.
        """
        self.trace = trace
        self.output_dataset = ['post_' + get_capitals(RandomForestSegmenter.__name__) + '_dataset']
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

    def apply(self, patch_shape, label_type='label', reference_points='core_point', sigma_min=1, sigma_max=10,
              n_sigma=3, intensity=True, edge=True, texture=True, direction=True, N_training_samples_per_label=500,
              inference_split_shape=(3, 3, 3), n_estimators=50, save_trained_random_forest=False):
        """
        Apply the operation to the trace. The operation train a random forest estimator by estimator (to reduce the
        RAM usage) from a set of features and using as target the labels produced by the Clusterer or ClusterValidator
        operations.

        :param patch_shape: (tuple[int]) shape of the patch used to produce the clustering.
        :param label_type: (str) optional, it can be 'label' or 'RS_label' if present. Kind of label used for the
                           definition of the classifier target.
        :param reference_points: (str) optional, it can be 'core_point', 'bilayer_point', 'boundary_point' or None. Are
                                 the type of points used to define the labeled region of the image, which define the
                                 target of the classifier. If None is given, all the points are used. Core, bilayer
                                 or boundary points are defined according to the ClusterValidator operation.
        :param sigma_min: (float) optional, minimal value of the sigma used to define the scale at which features are
                          computed.
        :param sigma_max: (float) optional, maximal value of the sigma used to define the scale at which features are
                          computed.
        :param n_sigma: (int) optional, number of sigma used to define the different scales at which the features are
                        computed.
        :param intensity: (bool) optional, if True intensity features at various scales are used.
        :param edge: (bool) optional, if True edge features at various scales are used.
        :param direction: (bool) optional, if True direction features at various scales are used
        :param N_training_samples_per_label: (bool) optional, number of data points used to train an estimator of the
                                             random forest for each label.
        :param inference_split_shape: (tuple[int]) optional, number of split per dimension used during the inference
                                      step. Each entries of the tuple represent the number of splits done for each axis
                                      for the definition of the chuck of data loaded in RAM during the inference step.
        :param n_estimators: (int) optional, number of estimators used by the random forest.
        :param save_trained_random_forest: (bool) optional, if True the trained random forest is saved using joblib.
        """
        if hasattr(self.trace,'seed'):

            np.random.seed(self.trace.seed)

        with self.pt:

            self.patch_shape = patch_shape
            self.label_type = label_type
            self.reference_points = reference_points
            self.sigma_min = sigma_min
            self.sigma_max = sigma_max
            self.n_sigma = n_sigma
            self.intensity = intensity
            self.edge = edge
            self.texture = texture
            self.direction = direction
            self.N_training_samples_per_label = N_training_samples_per_label
            self.inference_split_shape = inference_split_shape
            self.n_estimators = n_estimators
            self.save_trained_random_forest = save_trained_random_forest

        print('>----------------<')
        print('Segmentation with Random Forest')

        # load inputs
        data = self.trace.__getattribute__(self.input_dataset[0])
        labels_df = self.trace.__getattribute__(self.input_dataset[1])

        # preliminary operations
        data_shape = data.shape
        sel_labels_df = labels_df
        if self.reference_points is not None:

            sel_labels_df = labels_df[labels_df[self.reference_points] == 1]

        Zmin = min(sel_labels_df['Z'])
        Zmax = max(sel_labels_df['Z'])
        Ymin = min(sel_labels_df['Y'])
        Ymax = max(sel_labels_df['Y'])
        Xmin = min(sel_labels_df['X'])
        Xmax = max(sel_labels_df['X'])
        labels = list(set(sel_labels_df[self.label_type].tolist()))
        if len(labels) == 0:

            raise ValueError('No labels available with these input arguments.')

        # segmentation with random forest classifier
        tempdir = os.getcwd() + os.sep + 'bmmltools_temporary_files'
        tempfile.tempdir = manage_path(tempdir)
        with tempfile.TemporaryFile() as temp:

            # open a temporary hdf5 file to store computed feature
            temph5 = h5py.File(temp, 'a')

            # compute features
            sigmas = np.logspace(np.log2(self.sigma_min), np.log2(self.sigma_max), num=self.n_sigma, base=2,
                                 endpoint=True, )
            for s in sigmas:

                print('Computing features at scale {}...'.format(s), end='\r')
                gaussian_smoothed = filters.gaussian(data, s, preserve_range=False)
                if self.intensity:

                    temph5 = self._append_to_dataset(temph5, gaussian_smoothed)

                if self.edge:

                    edge_s = filters.sobel(gaussian_smoothed)
                    temph5 = self._append_to_dataset(temph5, edge_s)
                    del edge_s

                if self.texture:

                    text_s = self._texture_filter(gaussian_smoothed)
                    for i in range(len(text_s)):

                        temph5 = self._append_to_dataset(temph5, text_s[i])

                    del text_s

                del gaussian_smoothed
                if self.direction:

                    dir_s = feature.structure_tensor(data, mode='reflect', sigma=s)
                    for x in dir_s:

                        temph5 = self._append_to_dataset(temph5, x)

                    del dir_s

            print('Features computed!')

            # train random forest
            batch_params = (Zmin, Zmax, Ymin, Ymax, Xmin, Xmax)
            clf = RandomForestClassifier(n_estimators=1, n_jobs=4, max_depth=10, max_samples=0.05,
                                         warm_start=True, random_state=self.trace.seed)
            for i in range(self.n_estimators):

                print('Fitting random forest estimator {} over {}...'.format(i + 1, self.n_estimators), end='\r')
                XX, yy = self._get_batch(temph5, sel_labels_df, labels, batch_params)
                XX, yy = shuffle(XX, yy,random_state=self.trace.seed)
                clf.fit(XX, yy)
                clf.n_estimators += 1

            print('Random forest classifier trained!')

            Nz, Ny, Nx = self.inference_split_shape
            with h5py.File(self.trace.hdf5_trace_path, 'a') as tracefile:

                try:

                    del tracefile[self.trace._hdf5_inner_path+self.output_dataset[0]]

                except:

                    pass

                result = tracefile[self.trace._hdf5_inner_path].create_dataset(self.output_dataset[0],
                                                                               shape=data_shape,
                                                                               dtype=np.uint8)
                for n, zz in enumerate(self._inference_bounds(data_shape, 0, Nz)):

                    for m, yy in enumerate(self._inference_bounds(data_shape, 1, Ny)):

                        for p, xx in enumerate(self._inference_bounds(data_shape, 2, Nx)):

                            print(
                                'Segmentation with random forest of part {} over {}...'.format(n + m + p, Nz * Ny * Nx))
                            inf_patch_features = temph5['x'][zz[0]:zz[1], yy[0]:yy[1], xx[0]:xx[1]]
                            inf_patch_result = future.predict_segmenter(inf_patch_features, clf)
                            result[zz[0]:zz[1], yy[0]:yy[1], xx[0]:xx[1]] = inf_patch_result.astype(np.uint8)

                fullkey = '{}.{}'.format(self.trace.group_name, self.output_dataset[0])
                if not (fullkey in self.trace.numpy_variables_traced_on_hd):

                    self.trace.numpy_variables_traced_on_hd.append(fullkey)
                    tracefile[self.trace._hdf5_inner_path].attrs['numpy_variables_traced_on_hd'] = \
                        self.trace._filter_traced_list(self.trace.numpy_variables_traced_on_hd)

            temph5.close()

        if self.save_trained_random_forest:

            trace_pos = self.trace.get_trace_pos()
            saving_path = manage_path(self.trace.trace_file_path() + os.sep +
                                      RandomForestSegmenter.__name__+'_{}'.format(trace_pos))
            joblib.dump(clf, saving_path + os.sep + 'random_forest.joblib')

        # remove temporary folder
        os.rmdir(tempdir)
        print('>----------------<', end='\r')

        self._update_graph_dict()
        self._update_params_dict()
        if self.trace.enable_operations_tracking:

            self.trace.add_operation(RandomForestSegmenter.__name__, self, self.pt.get_current_tracked_params())

        return self.output_dataset

    @staticmethod
    def _texture_filter(gaussian_filtered):
        """
        from skimage
        """
        H_elems = [np.gradient(np.gradient(gaussian_filtered)[ax0], axis=ax1)
                   for ax0, ax1 in combinations_with_replacement(range(gaussian_filtered.ndim), 2)]
        eigvals = feature.hessian_matrix_eigvals(H_elems)
        return eigvals

    @staticmethod
    def _append_to_dataset(h5_fileobj, data, dataset_name='x'):
        """
        Append an array (of the correct shape) to an already existing dataset on an hdf5 file.

        :param h5_fileobj: (File object) file object of an hdf5 file (already open).
        :param: (ndarray) data to append.
        :param: (str) name of the dataset where the array is appended.
        """
        data_shape = data.shape
        if dataset_name in h5_fileobj.keys():

            dataset_shape = h5_fileobj[dataset_name].shape
            new_ax_len = h5_fileobj[dataset_name].shape[-1] + 1
            h5_fileobj[dataset_name].resize(new_ax_len, axis=len(dataset_shape) - 1)
            h5_fileobj[dataset_name][:, :, :, new_ax_len - 1] = data

        else:

            h5_fileobj.create_dataset(dataset_name, data=np.expand_dims(data, axis=-1), maxshape=data_shape + (None,))

        return h5_fileobj

    @staticmethod
    def _inference_bounds(data_shape, ax, Nax):
        """
        Define the bounds along a given axis of the chunck used during the inference

        :param data_shape: (tuple[int]) shape of the data to chunck.
        :param ax: (int) axis considered.
        :param Nax: (int) number of axis.
        """
        axbounds = np.ceil(np.linspace(0, data_shape[ax], Nax + 2))
        return [(int(axbounds[i]), int(axbounds[i + 1])) for i in range(Nax + 1)]

    def _get_batch(self, h5fileobj, df, labels, batch_params):
        """
        Get a batch of data for the training of a single estimator of the random forest.

        :param h5fileobj: (File object) file object of an hdf5 file (already open) containing the inputs of the
                          classifiers.
        :param df: (pandas.DataFrame) dataframe containing the information about the labeling of the patches (e.g.
                   output of the Clusterer or ClusterValidator)
        :param labels: (list[int]) list of the labels used to train the classifier.
        :param batch_params: (list) batch parameters (region bounds from which the input data are sampled).
        """
        Zmin, Zmax, Ymin, Ymax, Xmin, Xmax = batch_params
        X = []
        Y = []
        for L in labels:

            N = 0
            while True:

                z = np.random.randint(Zmin * self.patch_shape[0], (Zmax + 1) * self.patch_shape[0])
                y = np.random.randint(Ymin * self.patch_shape[1], (Ymax + 1) * self.patch_shape[1])
                x = np.random.randint(Xmin * self.patch_shape[2], (Xmax + 1) * self.patch_shape[2])
                try:

                    l = df[(df['Z'] == z // self.patch_shape[0]) & (df['Y'] == y // self.patch_shape[1]) &
                           (df['X'] == x // self.patch_shape[2])][self.label_type].tolist()[0]
                    if L == l:

                        X.append(h5fileobj['x'][z, y, x, :])
                        Y.append(l + 1)
                        N += 1

                    if N == self.N_training_samples_per_label:

                        break

                except:

                    continue

        return X, Y

    def read(self,name=None,save_as='npy'):
        """
        Read operation results in the hdf5 file. For this plugin, the results are organized in the form of a pandas
        dataframe.

        :param save_as: (str) optional, format used to save the result. It can be 'npy' of 'rrn'.
        :param name: (str) optional, name of the result file.
        """
        tmp_array = self.trace.__getattribute__(self.output_dataset[0])
        trace_pos = self.trace.get_trace_pos()
        path_to_readings_folder = manage_path(self.trace.trace_readings_path() + os.sep +
                                              RandomForestSegmenter.__name__+'_{}'.format(trace_pos))
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

                    link_list.append((i_name, o_name))

            key = len(tmp_dict)
            if tmp_dict[key - 1]['node'] == self.output_dataset:
                key = key - 1

            tmp_dict.update({key: {'node': self.output_dataset,
                                   'link': list(set(link_list)),
                                   'edge': RandomForestSegmenter.__name__}})
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

        params_dict.update({key: {'op': RandomForestSegmenter.__name__,
                                  'p': self.pt.get_current_tracked_params()}})
        previous_content[self.trace.group_name]['ops_parameters'] = params_dict
        with open(self.trace.json_trace_path, 'w') as jfile:
            json.dump(previous_content, jfile, indent=4)

