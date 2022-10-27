# title: 'explanation.py'
# author: Curcuraci L.
# date: 04/08/2022
#
# scope: Collect ML explanation methods or related

"""
Operation useful to get an explanation of the clustering obtained starting from a set of understandable features.
"""


#################
#####   LIBRARIES
#################


import numpy as np
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import dill

from joblib import delayed,Parallel
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance,partial_dependence
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from skimage.filters import threshold_multiotsu
from sklearn.metrics import balanced_accuracy_score
from GPyOpt.methods import BayesianOptimization

from bmmltools.utils.basic import manage_path,generate_parameter_space,get_capitals,ParametersTracker


################
#####    CLASSES
################


class MultiCollinearityReducer:
    """
    Operation used to reduce the multicollinerity in some input dataset
    """

    __name__ = 'MultiCollinearityReducer'
    def __init__(self,trace):
        """
        Initialize the operation. The MultiCollinearityReducer reduce the level of linear correlation in the dataset
        used to get an explanation of some target variable.

        :param trace: (bmmltools.core.tracer.Trace object) trace on which the operation act.
        """
        self.trace = trace
        self.output_dataset = ['post_'+get_capitals(MultiCollinearityReducer.__name__)+'_dataset']
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

    def apply(self,data_columns,target_columns,VIF_th,return_linear_association=None):
        """
        Apply the operation to the trace. This operation performs the reduction of multicollinerity on the dataset
        specified by the user (in the input of this operation and defined according to the setting of this method).

        :param data_columns: (list[str]) name od the columns used a s input feature for the explanation.
        :param target_columns: (list[str]) name for the features (typicaly 1) used as variable(s) to explain.
        :param VIF_th: (float) threshold on the variance inflation factor.
        :param return_linear_association: (None or str) optional, it can be None, 'pairwise', or 'full'.
        """
        # parameter initialization
        with self.pt:

            self.data_columns = list(data_columns)
            self.target_columns = list(target_columns)
            self.VIF_th = VIF_th
            self.return_linear_association = return_linear_association

        # core operations
        data_df = self._split_data_from_rest()
        selected_features_name = self._find_approximate_linear_independent_features(data_df)
        self._produce_output(selected_features_name)
        if return_linear_association == 'pairwise':

            self._pairwise_linear_association(data_df,selected_features_name)

        if return_linear_association == 'full':

            self._full_linear_association(data_df,selected_features_name)

        self._update_graph_dict()
        self._update_params_dict()
        if self.trace.enable_operations_tracking:

            self.trace.add_operation(MultiCollinearityReducer.__name__,self,self.pt.get_current_tracked_params())

        return self.output_dataset

    def _split_data_from_rest(self):
        """
        Split the data columns from all the dataset in the input dataframe.
        """
        df = self.trace.__getattribute__(self.input_dataset[0])
        data_df = df.loc[:,self.data_columns]
        return data_df

    def _find_approximate_linear_independent_features(self,data_df):
        """
        VIF reduction step.

        :param data_df: (pandas.DataFrame)
        """
        fname = list(data_df.columns)
        while True:

            X = data_df[fname].to_numpy()
            VIFs = []
            for i in range(len(fname)):

                res = self._compute_VIF(X,i)
                VIFs.append(res['VIF'])

            VIFs = np.array(VIFs)
            if np.any(VIFs >= self.VIF_th):

                idx_to_delete = np.argmax(VIFs)
                fname.pop(idx_to_delete)

            else:

                break

        return fname

    @staticmethod
    def _compute_VIF(data_matrix,column_index):
        """
        Compute the variance inflation factor (VIF) for one feature.

        :param data_matrix: (numpy.array) numpy array containing the data used;
        :param column_index: (int) index of the column associated to the feature selected
        :return: dictionary containing the column index, the coefficients of the linear model for the prediction of the
                 selected feature, and the corresponding VIF.
        """
        # get data
        y = data_matrix[:, column_index:column_index + 1]
        xpre = data_matrix[:, 0:column_index]
        xpost = data_matrix[:, column_index + 1:]
        x = np.concatenate([np.ones(y.shape), -xpre, -xpost], axis=1)

        # lstq fit coefficients
        lsq_result = np.linalg.lstsq(x, y, rcond=None)
        coeff = lsq_result[0]

        # compute R2
        yhat = np.dot(x, coeff)
        ymean = y.mean(0, keepdims=True)
        ssreg = np.sum(np.square(y - yhat))
        sstot = np.sum(np.square(y - ymean))
        R2 = 1 - ssreg / sstot

        # compute VIF
        VIF = 1 / (1 - R2)
        return {'column_index': column_index, 'lstq_coeff': coeff, 'R2': R2, 'VIF': VIF}

    def _produce_output(self,selected_feature_names):
        """
        Save result on the trace.

        :param selected_feature_names: (list[str]) list of names of the features surviving the multicollinearity
                                       screening.
        """
        df = self.trace.__getattribute__(self.input_dataset[0])
        selected_df = df.loc[:,selected_feature_names+self.target_columns]
        selected_df.columns = selected_feature_names+['target_output']
        self.trace.__setattr__(self.output_dataset[0],selected_df)

    def _pairwise_linear_association(self,data_df,selected_features_name):
        """
        Produce the pairwise linear association dictionary. This dictionary for each selected feature Y, contains
        dictionaries (one for all the other remaining feature Y in the dataset) where the following information are listed:

        * name of the feature X;

        * VIF between X and Y;

        * the coefficients :math:`m` and :math:`q` of the relation :math:`Y = mX+q`.

        :param data_df: (pandas.dataframe) dataframe containing the data used for this analysis.
        :param selected_feature_name: (list[str]) list of the name of the columns of the data_df which are used to produce
                                      the association dictionary (i.e. the selected features Y).
        """
        all_features_name = list(data_df.columns)
        eliminated_features_name = list(set(all_features_name) - set(selected_features_name))
        association_dict = {}
        for f1 in eliminated_features_name:

            associated = []
            for f2 in selected_features_name:

                Xd = data_df[[f2, f1]].to_numpy()
                res = self._compute_VIF(Xd, 1)
                associated.append({'independent_variable_name': f2,
                                   'VIF': res['VIF'],
                                   'lstq_coeff': res['lstq_coeff'].tolist()})

            association_dict.update({f1: associated})

        # save result
        trace_pos = self.trace.get_trace_pos()
        path_to_trace_file_folder = manage_path(self.trace.trace_file_path() + os.sep +
                                                MultiCollinearityReducer.__name__ + '_{}'.format(trace_pos))
        with open(path_to_trace_file_folder+os.sep+'pairwise_association.json','w') as jfile:

            json.dump(association_dict,jfile,indent=4)

    def _full_linear_association(self,data_df,selected_features_name):
        """
        Produce the full linear association dictionary. This dictionary for each selected feature Y, the following
        information listed below:

        * name of the features X_1,...X_n-1;

        * VIF

        * the coefficients :math:`m`s and :math:`q` of the relation :math:`Y = m_1X_1+...+m_n-1 X_n-1+q`.

        :param data_df: (pandas.dataframe) dataframe containing the data used for this analysis.
        :param selected_feature_name: (list[str]) list of the name of the columns of the data_df which are used to produce
                                      the association dictionary (i.e. the selected features Y).
        """
        all_features_name = list(data_df.columns)
        eliminated_features_name = list(set(all_features_name) - set(selected_features_name))
        association_dict = {}
        for f1 in eliminated_features_name:

            cols = selected_features_name + [f1]
            Xd = data_df[cols].to_numpy()
            res = self._compute_VIF(Xd, cols.index(f1))
            association_dict.update({f1: {'independent_variables_name': selected_features_name,
                                          'VIF': res['VIF'],
                                          'lstq_coeff': res['lstq_coeff'].tolist()}})

        # save result
        trace_pos = self.trace.get_trace_pos()
        path_to_trace_file_folder = manage_path(self.trace.trace_file_path()+os.sep+
                                                MultiCollinearityReducer.__name__+ '_{}'.format(trace_pos))
        with open(path_to_trace_file_folder+os.sep+'full_association.json','w') as jfile:

            json.dump(association_dict,jfile,indent=4)

    def read(self,name=None,save_as = 'csv'):
        """
        Read operation results in the hdf5 file. For this plugin, the results are organized in the form of a pandas
        dataframe.

        :param save_as: (str) optional, format used to save the result. It can be 'csv' of 'json'.
        :param name: (str) optional, name of the result file.
        """
        tmp_df = self.trace.__getattribute__(self.output_dataset[0])
        trace_pos = self.trace.get_trace_pos()
        path_to_readings_folder = manage_path(self.trace.trace_readings_path() + os.sep +
                                              MultiCollinearityReducer.__name__ + '_{}'.format(trace_pos))
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

                    link_list.append((i_name,o_name))

            key = len(tmp_dict)
            if tmp_dict[key-1]['node'] == self.output_dataset:

                key = key-1

            tmp_dict.update({key: {'node': self.output_dataset,
                                   'link': list(set(link_list)),
                                   'edge': MultiCollinearityReducer.__name__}})
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

            key = key - 1

        params_dict.update({str(key): {'op': MultiCollinearityReducer.__name__,
                                       'p': self.pt.get_current_tracked_params()}})
        previous_content[self.trace.group_name]['ops_parameters'] = params_dict
        with open(self.trace.json_trace_path, 'w') as jfile:

            json.dump(previous_content, jfile, indent=4)

class ExplainWithClassifier:
    """
    Operation used to get an explanation of some target variable in term of mean permutation importance and
    partial dependency of an ensemble of classifiers.
    """

    __name__ = 'ExplainWithClassifier'
    def __init__(self,trace):
        """
        Initialize the operation. The ExplainWithClassifier operation compute the feature permutation importance
        and the model partial dependency from an ensemble of binary classifier for each label. The decision of
        the classifiers for the recognition of a given target label can be explained using these two quantities,
        from which an explanation of the label in term of the input feature can be deduced.

        :param trace: (bmmltools.core.tracer.Trace object) trace on which the operation act.
        """
        self.trace = trace
        self.output_dataset = ['post_' + get_capitals(ExplainWithClassifier.__name__) + '_dataset']
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

    def apply(self,test_to_train_ratio = 0.1,model_type = 'RandomForest',n_kfold_splits = 5,n_grid_points=100,save_graphs=False):
        """
        Apply the operation to the trace. This operation  trains an ensemble of binary classifiers and use them to
        compute the feature permutation importance (PI) and the partial dependency (PDs) for each feature. This is done
        for each target label.

        :param test_to_train_ratio: (float between 0 and 1) fraction of input dataset used as test set.
        :param model_type: DO NOT CHANGE
        :param n_kfold_splits: (int) optional, number of k-fold splits.
        :param n_grid_points: (int) optional, number of points used to evaluate the partial dependency.
        :param save_graphs: (bool) optional, if True the graph of the PI and PDs are saved for all labels.
        """
        #
        with self.pt:

            self.n_kfold_splits = n_kfold_splits
            self.model_type = model_type
            self.test_to_train_ratio = test_to_train_ratio
            self.save_graph = save_graphs
            self.n_grid_points = n_grid_points

        #
        (X_norm_train,y_train),(X_norm_test,y_test) = self._prepare_dataset()
        results = self._compute_PIs_and_PDs(X_norm_train,y_train,X_norm_test,y_test)
        self.trace.__setattr__(self.output_dataset[0], results)
        if self.save_graph:

            self._produce_graphs(results)

        self._update_graph_dict()
        self._update_params_dict()
        if self.trace.enable_operations_tracking:

            self.trace.add_operation(ExplainWithClassifier.__name__,self,self.pt.get_current_tracked_params())

        return self.output_dataset

    def _prepare_dataset(self):
        """
        Prepare dataset for the computation of training of the ensemble of classifiers.

        :return: (tuple[ndarray],tuple[ndarray]) inputs/target test/train splits.
        """
        df = self.trace.__getattribute__(self.input_dataset[0])
        all_columns = list(df.columns)
        X = df[all_columns[:-1]].to_numpy()
        y = df[all_columns[-1]].to_numpy()
        self.feature_names = all_columns[:-1]

        # Normalize data
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=self.test_to_train_ratio,shuffle=True,stratify=y)
        self._m = X_train.min(0, keepdims=True)
        self._M = X_train.max(0, keepdims=True)
        X_norm_train = (X_train-self._m)/(self._M-self._m)
        X_norm_test = (X_test-self._m)/(self._M-self._m)
        return (X_norm_train,y_train),(X_norm_test,y_test)

    def _get_parameter_space(self):
        """
        Parameter space used to generate the model ensamble...[EXPAND HERE IF MORE MODEL NEED TO BE CONSIDERED...]
        """
        if self.model_type == 'RandomForest':

            p_space = {'n_estimators': [10, 50, 100],
                       'criterion': ['gini', 'entropy'],
                       'max_depth': [None, 5, 10],
                       'min_samples_split': [2, 5, 10],
                       'max_features': [None, 'sqrt', 'log2'],
                       'n_jobs': [-2]}

            pcombs, pnames = generate_parameter_space(p_space)
            return [{pnames[n]: comb[n] for n in range(len(comb))} for comb in pcombs]

        else:

            pass

    def _compute_PIs_and_PDs(self,X_norm_train,y_train,X_norm_test,y_test):
        """
        Computation of the mean PI and PDs for each label from the ensemble of classifiers

        :param X_norm_train: (ndarray) train input dataset.
        :param y_train: (ndarray) train target.
        :param X_norm_test: (ndarray) test input dataset.
        :param y_test: (ndarray) test target.
        :return: (dict) dictionary with all the PI and PDs for each labels plus the average F1 score of the classifiers
                 ensemble.
        """
        self._label_list = list(set(y_train))
        if self.save_graph:

            self._scores = []

        results = {}
        for L in self._label_list:

            #
            single_label_y_train = (y_train == L).astype(int)
            single_label_y_test = (y_test == L).astype(int)

            #
            params_to_test = self._get_parameter_space()

            #
            pis = []
            scores = []
            par_dips = []
            for n, params in enumerate(params_to_test):

                print('Label {} - {} - {}/{}'.format(L,self.model_type,n,len(params_to_test)), end='\r')
                skf = StratifiedKFold(n_splits=self.n_kfold_splits,shuffle=True)
                models = []
                tmpscores = []
                for tr_idx, val_idx in skf.split(X_norm_train, single_label_y_train):

                    # train-validation split
                    Xtr, ytr = X_norm_train[tr_idx], single_label_y_train[tr_idx]
                    Xval, yval = X_norm_train[val_idx], single_label_y_train[val_idx]

                    # balance the dataset
                    total = len(ytr)
                    pos = np.sum(ytr)
                    neg = total - pos
                    weight_for_0 = (1 / neg) * (total / 2.0)
                    weight_for_1 = (1 / pos) * (total / 2.0)
                    class_weight = {0: weight_for_0, 1: weight_for_1}

                    # train model
                    rfc = RandomForestClassifier(**params, class_weight=class_weight)
                    rfc.fit(Xtr, ytr)
                    models.append(rfc)
                    tmpscores.append(f1_score(yval, rfc.predict(Xval)))

                rfcbest = models[np.argmin(tmpscores)]
                ypred = rfcbest.predict(X_norm_test)
                scores.append(f1_score(single_label_y_test, ypred, average='weighted'))
                pis.append(permutation_importance(rfcbest, X_norm_test, single_label_y_test, n_jobs=-1))

                def func_to_par(i):

                    # par_dip_y, par_dip_x = partial_dependence(rfcbest, X_norm_test, features=[i], percentiles=[0,1])
                    par_dip_bunch = partial_dependence(rfcbest, X_norm_test, features=[i], percentiles=[0, 1])
                    par_dip_y = par_dip_bunch['average'][0,:]
                    par_dip_x = par_dip_bunch['values'][0]
                    return [self.feature_names[i], [par_dip_x, par_dip_y]]

                tmp_par_list = Parallel(n_jobs=-2)(delayed(func_to_par)(i) for i in range(len(self.feature_names)))
                par_dips.append({elem[0]: elem[1] for elem in tmp_par_list})

            #
            scores = np.array(scores)
            self._scores.append(np.max(scores))

            #
            pi_means = np.array([pi['importances_mean'] for pi in pis])
            est_pi = np.sum(np.expand_dims(scores,axis=-1)*pi_means,axis=0)/np.sum(scores)
            pi_stds = np.array([pi['importances_std'] for pi in pis])
            est_pi_std = np.sqrt(np.sum(np.expand_dims((scores/np.sum(scores))**2,axis=-1)*pi_stds**2,axis=0))
            tmp_df = pd.DataFrame({'estimated_permutation_importance_mean': list(est_pi),
                                   'estimated_permutation_importance_std': list(est_pi_std),
                                   'features': self.feature_names})
            tmp_df.set_index('features',inplace=True)
            results.update({'PI_Label_{}'.format(L): tmp_df})

            #
            for idx,name in enumerate(self.feature_names):

                M = self._M[0,idx]
                m = self._m[0,idx]

                # pds_axis = np.array(par_dips[0][name][0])*(M-m)+m
                pds_axis = par_dips[0][name][0] * (M - m) + m

                pds = np.array([elem[name][1] for elem in par_dips])
                est_pd = np.sum(np.expand_dims(scores,axis=-1)*pds,axis=0)/np.sum(scores)
                est_pd_std = np.sqrt(np.sum(
                    np.expand_dims(scores/np.sum(scores),axis=1)*(pds-np.expand_dims(est_pd,axis=0))**2,axis=0))
                if len(pds_axis) != self.n_grid_points:

                    new_pds_axis = np.linspace(pds_axis.min(), pds_axis.max(),self.n_grid_points)
                    func1 = interp1d(pds_axis, est_pd, kind='linear')
                    func2 = interp1d(pds_axis, est_pd_std, kind='linear')

                    est_pd = func1(new_pds_axis)
                    est_pd_std = func2(new_pds_axis)
                    pds_axis = new_pds_axis

                # pd_name_final = np.vstack([est_pd, est_pd_std, pds_axis])
                tmp_df = pd.DataFrame({'estimated_partial_dependency_mean': list(est_pd),
                                       'estimated_partial_dependency_std': list(est_pd_std),
                                       'feature_values': list(pds_axis)})

                results.update({'PD_Label_{}_feature_{}'.format(L,name): tmp_df})

        f1_score_df = pd.DataFrame({'label': ['Label_{}'.format(L) for L in self._label_list],
                                    'f1_score': self._scores})
        f1_score_df.set_index('label',inplace=True)
        results.update({'classifier_f1_scores': f1_score_df})
        return results

    def _produce_graphs(self,x):
        """
        Produce the graphs for the PI and PDs and save them in a folder called 'Label [LABEL NUMBER]' in the trace
        file folder. This is done for each Label.
        """
        PIs = {'PI_Label_{}'.format(L): x['PI_Label_{}'.format(L)] for L in self._label_list}
        PDs = {'PD_Label_{}_feature_{}'.format(L,f): x['PD_Label_{}_feature_{}'.format(L,f)]
               for L in self._label_list for f in self.feature_names}
        trace_pos = self.trace.get_trace_pos()
        saving_folder = manage_path(self.trace.trace_file_path() + os.sep +
                                    ExplainWithClassifier.__name__ + '_{}'.format(trace_pos))
        for n,L in enumerate(self._label_list):

            #
            est_pi = PIs['PI_Label_{}'.format(L)].loc[:,'estimated_permutation_importance_mean']
            est_pi_std = PIs['PI_Label_{}'.format(L)].loc[:,'estimated_permutation_importance_std']
            saving_folder_label = manage_path(saving_folder+os.sep+'Label_{}'.format(L))

            #
            fig, ax = plt.subplots(figsize=(20, 20))
            ax.set_title('     Permutation importance {}  '
                         '\nLabel: {} | mean f1 score: {}'.format(self.model_type,L,self._scores[n]))
            ax.bar(list(range(len(self.feature_names))),est_pi,yerr=est_pi_std,
                   align='center',alpha=0.5,ecolor='black',capsize=10)
            ax.set_ylabel('relevance')
            ax.set_xlabel('features')
            plt.xticks(list(range(len(self.feature_names))), labels=self.feature_names, rotation=45)
            ax.yaxis.grid(True)
            plt.savefig(saving_folder_label+os.sep+'Label_{}_-_{}_permutation_importance.pdf'.format(L,self.model_type))
            plt.close(fig)

            #
            for n,name in enumerate(self.feature_names):

                #
                x_axis = PDs['PD_Label_{}_feature_{}'.format(L,name)].loc[:,'feature_values']
                est_pd = PDs['PD_Label_{}_feature_{}'.format(L,name)].loc[:,'estimated_partial_dependency_mean']
                est_pd_std = PDs['PD_Label_{}_feature_{}'.format(L,name)].loc[:,'estimated_partial_dependency_std']

                #
                fig, ax = plt.subplots(figsize=(20, 20))
                ax.set_title('Label {} - Feature {} partial dependence'.format(L,name))
                ax.plot(x_axis,est_pd)
                ax.fill_between(x_axis,est_pd-est_pd_std,est_pd+est_pd_std,color='b',alpha=0.1)
                ax.set_xlabel(name)
                for label in ax.get_xticklabels():

                    label.set_ha("right")
                    label.set_rotation(25)
                    label.set_rotation_mode('anchor')

                #
                plt.savefig(saving_folder_label+os.sep+
                            'Label_{}_feature_{}_-_{}_partial_dependence.pdf'.format(L,name,self.model_type))
                plt.close(fig)

    def read(self,save_as = 'csv'):
        """
        Read operation results in the hdf5 file. For this plugin, the results are organized in the form of a pandas
        dataframe.

        :param save_as: (str) optional, format used to save the result. It can be 'csv' or 'json'.
        """
        res = self.trace.__getattribute__(self.output_dataset[0])
        labels = [elem.split('_')[-1] for elem in res.keys() if 'PI' in elem]
        features = [elem.split('_')[-1] for elem in res.keys() if 'PD_Label_{}'.format(labels[0]) in elem]
        trace_pos = self.trace.get_trace_pos()
        saving_folder = manage_path(self.trace.trace_readings_path() + os.sep +
                                    ExplainWithClassifier.__name__ + '_{}'.format(trace_pos))
        for L in labels:

            saving_folder_by_label = manage_path(saving_folder+os.sep+'Label_{}'.format(L))
            self._save_df(res['PI_Label_{}'.format(L)],saving_folder_by_label,
                          'Label_{}_permutation_importance'.format(L),save_as)
            for f in features:

                self._save_df(res['PD_Label_{}_feature_{}'.format(L,f)],saving_folder_by_label,
                              'Label_{}_feature_{}_partial_dependency'.format(L,f),save_as)

    def _save_df(self,df,path,name,format):
        """
        Organize the results in a specific format and save it as dataframe

        :param df: (pandas.DataFrame) dataframe to save.
        :param path: (str) path where the dataframe is saved.
        :param name: (str) saved file name.
        :param format: (str) file format. It can be 'csv' or 'json'
        """
        saving_path = path+os.sep+name+'.'+format
        if format == 'csv':

            df.to_csv(saving_path)

        elif format == 'json':

            df.to_json(saving_path)

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

            tmp_dict.update({key: {'node': self.output_dataset,
                                   'link': list(set(link_list)),
                                   'edge': ExplainWithClassifier.__name__}})
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

            key = key - 1

        params_dict.update({str(key): {'op': ExplainWithClassifier.__name__,
                                       'p': self.pt.get_current_tracked_params()}})
        previous_content[self.trace.group_name]['ops_parameters'] = params_dict
        with open(self.trace.json_trace_path, 'w') as jfile:

            json.dump(previous_content, jfile, indent=4)

class InterpretPIandPD:
    """
    Operation used to deduce a cluster definition for the PI and PDs in simple situations.
    """

    __name__ = 'InterpretPIandPD'
    def __init__(self, trace):
        """
        Initialize the operation. The InterpretPIandPD try to estimate automatically the intervals defining a given
        target label when the PI and PDs are not too complex (i.e. when at most two thresholds in the PDs are needed
        in order to define an interval for the cluster definition).

        :param trace: (bmmltools.core.tracer.Trace object) trace on which the operation act.
        """
        self.trace = trace
        self.output_dataset = ['post_' + get_capitals(InterpretPIandPD.__name__) + '_dataset']
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

    def apply(self, positive_PI_th=0.8, n_bins_pd_mean=40, prominence=None, adjust_accuracy=True,
              bayes_optimize_interpretable_model=False, bo_max_iter=100, save_interpretable_model=False):
        """
        Apply the operation to the trace. This operation try to derive the intervals defining the clusters
        in simple cases.

        :param positive_PI_th: (float between 0 and 1) optional, fraction of positive PI used to get the explanation.
        :param n_bins_pd_mean: (int) optional, number of points used to evaluate the partial dependencies.
        :param prominence: (None or float) optional, prominence parameter of the peak finder algorithm used.
        :param adjust_accuracy: (bool) optional, if True the adjusted accuracy is used to evaluate the interpretable
                                model.
        :param bayes_optimize_interpretable_model: (bool) optional, if True the intervals bounds are optimized in order
                                                   maximize the accuracy.
        :param bo_max_iter: (int) optional, maximal number of iteration of the bayesian optimization algorithm.
        :param save_interpretable_model: (bool) optional, if True the interpretable model (and its parameters) are
                                         saved.
        """
        with self.pt:

            self.positive_PI_th = positive_PI_th
            self.n_bins_pd_mean = n_bins_pd_mean
            self.prominence = prominence
            self.adjust_accuracy = adjust_accuracy
            self.bayes_optimize_interpretable_model = bayes_optimize_interpretable_model
            self.save_interpretable_model = save_interpretable_model
            self.bo_max_iter = bo_max_iter

        # prepare inputs
        res = self.trace.__getattribute__(self.input_dataset[0])
        labels = [elem.split('_')[-1] for elem in res.keys() if 'PI' in elem]
        features = [elem.split('_')[-1] for elem in res.keys() if 'PD_Label_{}'.format(labels[0]) in elem]
        ref = self.trace.__getattribute__(self.input_dataset[1])

        # get interpretation
        interpretation_results = {}
        interpretation_results = self._analyze_PIs(res,features,labels,interpretation_results)
        interpretation_results = self._analyze_PDs(res,features,labels,interpretation_results)
        if self.bayes_optimize_interpretable_model:

            interpretation_results = self._bayes_optimize_interpretable_model(ref,features,labels,interpretation_results)

        interpretation_results = self._compute_interpretation_accuracy(ref,features,labels,interpretation_results)
        if self.save_interpretable_model:

            self._save_model()
            self._save_params(labels,features,interpretation_results)

        # save results
        self.trace.__setattr__(self.output_dataset[0], interpretation_results)

        self._update_graph_dict()
        self._update_params_dict()
        if self.trace.enable_operations_tracking:

            self.trace.add_operation(InterpretPIandPD.__name__,self,self.pt.get_current_tracked_params())

        return self.output_dataset

    def _analyze_PIs(self, res, features, labels, interpretation_results):
        """
        Analyze the permutation importance for each cluster to select the most relevant feature for the description of
        a given cluster.

        :param res: (pandas.DataFrame) result of the ExplainWithClassifier operation.
        :parma features: (list[str]) list of features names.
        :param labels: (list[str]) list of labels.
        :param interpretation_results: (dict) dictionary where results are saved and information from previous step is
                                       stored.
        :return: updated interpretation results.
        """
        # screen permutation importance
        PI_result = {'features': features}
        for L in labels:

            perm_imp_L = res['PI_Label_{}'.format(L)].loc[:, 'estimated_permutation_importance_mean'].tolist()
            normalized_positive_perm_imp_L = np.maximum(perm_imp_L, 0) / np.sum(np.maximum(perm_imp_L, 0))
            np.sort(normalized_positive_perm_imp_L)
            sorted_normalized_positive_perm_imp_L, sorted_features = \
                zip(*sorted(zip(normalized_positive_perm_imp_L, features), reverse=True))
            total_pi_means = 0
            selected_features = []
            for i in range(len(sorted_normalized_positive_perm_imp_L)):

                total_pi_means += sorted_normalized_positive_perm_imp_L[i]
                selected_features.append(sorted_features[i])
                if total_pi_means > self.positive_PI_th:
                    break

            col_L = []
            for f in features:

                col_L.append(f in selected_features)

            PI_result.update({'Label_{}'.format(L): col_L})

        # organize result as pandas dataframe
        PI_result_df = pd.DataFrame(PI_result)
        PI_result_df.set_index('features', inplace=True)
        PI_result_df = PI_result_df.transpose()

        # save result
        interpretation_results.update({'Feature_relevance': PI_result_df})
        return interpretation_results

    def _analyze_PDs(self, res, features, labels, interpretation_results):
        """
        Analyze the partial dependencies to estimate the values defining the a given cluster.

        :param res: (pandas.DataFrame) result of the ExplainWithClassifier operation.
        :parma features: (list[str]) list of features names.
        :param labels: (list[str]) list of labels.
        :param interpretation_results: (dict) dictionary where results are saved and information from previous step is
                                       stored.
        :return: updated interpretation results.
        """
        PI_result_df = interpretation_results['Feature_relevance']
        for L in labels:

            for f in features:

                if PI_result_df.loc['Label_{}'.format(L),f]:

                    pd_mean = res['PD_Label_{}_feature_{}'.format(L, f)].loc[:,
                              'estimated_partial_dependency_mean'].to_numpy()
                    f_axis = res['PD_Label_{}_feature_{}'.format(L, f)].loc[:, 'feature_values'].to_numpy()
                    pd_hist, pd_bins = np.histogram(pd_mean, bins=self.n_bins_pd_mean, density=True)
                    if self.prominence == None:

                        self.prominence = 0.8 * np.max(pd_hist)  # 32*np.max(pd_hist)/40

                    peak_pos = find_peaks(pd_hist, prominence=self.prominence)[0]
                    n_classes = len(peak_pos) + 2
                    img = pd_mean[None, :]
                    TTHH = np.max(threshold_multiotsu(img, classes=n_classes, nbins=self.n_bins_pd_mean))
                    intervals = []
                    f_axis_threshold_values = []
                    for i in range(len(pd_mean)):

                        if pd_mean[i] < TTHH:

                            intervals.append(0)

                        else:

                            intervals.append(1)

                        if i > 0 and intervals[-2] == intervals[-1]:

                            del intervals[-1]

                        elif i > 0:

                            f_axis_threshold_values.append(f_axis[i])

                    f_axis_threshold_values.append(f_axis.max())
                    intervals = intervals + [0]
                    f_axis_threshold_values = [f_axis.min()] + f_axis_threshold_values
                    PD_result_L_df = pd.DataFrame({'thresholds': f_axis_threshold_values,
                                                   'post_thresholds_value': intervals})
                    interpretation_results.update({'Label_{}_feature_{}_interval'.format(L, f): PD_result_L_df})

        return interpretation_results

    def _compute_interpretation_accuracy(self, ref, features,labels, interpretation_results):
        """
        Compute accuracy of the interpretable classifier.

        :param res: (pandas.DataFrame) result of the ExplainWithClassifier operation.
        :parma features: (list[str]) list of features names.
        :param labels: (list[str]) list of labels.
        :param interpretation_results: (dict) dictionary where results are saved and information from previous step is
                                       stored.
        :return: updated interpretation results.
        """
        interpretation_accuracy_score = {}
        for L in labels:

            y = ref.loc[:, ref.columns.tolist()[-1]].to_numpy()
            y_L = np.array(y == int(L), dtype=int)
            X = ref.loc[:, ref.columns.tolist()[:-1]].to_numpy()

            ir_keys = interpretation_results.keys()
            sel_features = [elem.split('_')[-2] for elem in ir_keys if 'Label_{}'.format(L) in elem]
            idx_features = [features.index(f) for f in sel_features]
            THss = [interpretation_results['Label_{}_feature_{}_interval'.format(L, f)]['thresholds'].tolist()[1:-1] for
                    f in sel_features]
            post_TH_vals = [
                interpretation_results['Label_{}_feature_{}_interval'.format(L, f)]['post_thresholds_value'].tolist()
                for f in sel_features]
            params = (idx_features, THss, post_TH_vals)
            y_pred = self.interpretable_model(X, params)

            interpretation_accuracy_score.update(
                {'Label_{}'.format(L): [balanced_accuracy_score(y_L, y_pred, adjusted=self.adjust_accuracy)]})

        ias_df = pd.DataFrame(interpretation_accuracy_score)
        ias_df = ias_df.transpose()
        ias_df.columns = ['balanced_{}accuracy'.format('adjusted_' if self.adjust_accuracy else '')]
        interpretation_results.update({'interpretation_accuracy': ias_df})
        return interpretation_results

    def _save_model(self):
        """
        Save the interpretable model using dill.
        """
        trace_pos = self.trace.get_trace_pos()
        with open(manage_path(self.trace.trace_file_path() + os.sep + InterpretPIandPD.__name__+'_{}'.format(trace_pos))
                  + os.sep + 'trace_{}_interpretable_classifier.dill'.format(self.trace.trace_code), 'wb') as dillfile:

            dill.dump(self.interpretable_model, dillfile)

    def _save_params(self, labels, features, interpretation_results):
        """
        Save the parameters of the interpretable model in a json file

        :param labels: (list[int]) list of labels.
        :param features: (list[str]) list of the features used.
        :param interpretation_results: (dict) dictionary containing the interpretation results
        """
        trace_pos = self.trace.get_trace_pos()
        params_json = {}
        for L in labels:

            ir_keys = interpretation_results.keys()
            sel_features = [elem.split('_')[-2] for elem in ir_keys if 'Label_{}'.format(L) in elem]
            idx_features = [features.index(f) for f in sel_features]
            THss = [interpretation_results['Label_{}_feature_{}_interval'.format(L, f)][
                        'thresholds'].tolist()[1:-1] for f in sel_features]
            post_TH_vals = [
                interpretation_results['Label_{}_feature_{}_interval'.format(L, f)][
                    'post_thresholds_value'].tolist() for f in sel_features]
            params = (idx_features, THss, post_TH_vals)
            params_json.update({'Label_{}'.format(L): params})

        with open(manage_path(self.trace.trace_file_path() + os.sep + InterpretPIandPD.__name__+'_{}'.format(trace_pos))
                  + os.sep + 'trace_{}_classifier_params.json'.format(self.trace.trace_code), 'w') as jsonfile:

            # content = json.dumps(params_json)
            json.dump(params_json, jsonfile,indent=4)

    def _bayes_optimize_interpretable_model(self, ref, features,labels, interpretation_results):
        """
        Bayesian optimization routine for the maximization of the accuracy.

        :param res: (pandas.DataFrame) result of the ExplainWithClassifier operation.
        :parma features: (list[str]) list of features names.
        :param labels: (list[str]) list of labels.
        :param interpretation_results: (dict) dictionary where results are saved and information from previous step is
                                       stored.
        :return: updated interpretation results.
        """
        print('Bayesian optimization of the interpretable classifier')
        X = ref.loc[:, ref.columns.tolist()[:-1]].to_numpy()
        y = ref.loc[:, ref.columns.tolist()[-1]].to_numpy()
        for L in labels:

            print('Optimizing label {} definition...'.format(L),end='')

            # prepare all the necessary quantities
            y_L = np.array(y == int(L), dtype=int)
            ir_keys = interpretation_results.keys()
            sel_features = [elem.split('_')[-2] for elem in ir_keys if 'Label_{}'.format(L) in elem]
            idx_features = [features.index(f) for f in sel_features]

            # prepare the initial threshold values for the bayesian optimization algorithm for a given label
            THss0 = [interpretation_results['Label_{}_feature_{}_interval'.format(L, f)]['thresholds'].tolist()[1:-1]
                     for f in sel_features]
            flatten_THss0 = []
            for elem in THss0:

                for el in elem:
                    flatten_THss0.append(el)

            # define the domain for the bayesian optimization
            post_TH_vals = []
            domain = []
            for n, f in enumerate(sel_features):

                post_TH_vals.append(
                    interpretation_results['Label_{}_feature_{}_interval'.format(L, f)][
                        'post_thresholds_value'].tolist())

                if post_TH_vals[-1] == [1, 0, 0] or post_TH_vals[-1] == [0, 1, 0]:

                    xmin = interpretation_results['Label_{}_feature_{}_interval'.format(L, f)][
                        'thresholds'].tolist()[0]
                    xmax = interpretation_results['Label_{}_feature_{}_interval'.format(L, f)][
                        'thresholds'].tolist()[-1]
                    domain.append({'name': 'var_{}'.format(f), 'type': 'continuous', 'domain': (xmin, xmax)})

                elif post_TH_vals[-1] == [0, 1, 0, 0] or post_TH_vals[-1] == [1, 0, 1, 0]:

                    xmin = interpretation_results['Label_{}_feature_{}_interval'.format(L, f)][
                        'thresholds'].tolist()[0]
                    xmax = interpretation_results['Label_{}_feature_{}_interval'.format(L, f)][
                        'thresholds'].tolist()[-1]
                    domain.append({'name': 'var_{}1'.format(f), 'type': 'continuous', 'domain': (xmin, xmax)})
                    domain.append({'name': 'var_{}2'.format(f), 'type': 'continuous', 'domain': (xmin, xmax)})

            # define loss function to optimize
            def loss(x):

                THss = []
                i = 0
                for i_vals in post_TH_vals:

                    if i_vals == [1, 0, 0] or i_vals == [0, 1, 0]:

                        THss.append([x[0][i]])
                        i += 1

                    if i_vals == [0, 1, 0, 0] or i_vals == [1, 0, 1, 0]:

                        THss.append([x[0][i], x[0][i + 1]])
                        i += 2

                params = (idx_features, THss, post_TH_vals)
                y_pred = self.interpretable_model(X, params)
                return 1 - balanced_accuracy_score(y_L, y_pred, adjusted=self.adjust_accuracy)

            # bayesian optimization routine
            myBopt = BayesianOptimization(f=loss, domain=domain, X=np.array([flatten_THss0]))
            myBopt.run_optimization(max_iter=self.bo_max_iter)
            best_thresholds = myBopt.x_opt

            # update hte interpretation_results dictionary with the new thresholds
            i = 0
            for f in sel_features:

                tmp_df = interpretation_results['Label_{}_feature_{}_interval'.format(L, f)]
                if len(tmp_df) == 3:
                    tmp_df.loc[1, 'thresholds'] = best_thresholds[i]
                    i += 1

                if len(tmp_df) == 4:
                    tmp_df.loc[1, 'thresholds'] = best_thresholds[i]
                    tmp_df.loc[2, 'thresholds'] = best_thresholds[i + 1]
                    i += 2

                interpretation_results['Label_{}_feature_{}_interval'.format(L, f)] = tmp_df

            print('[Ok]')

        return interpretation_results

    @staticmethod
    def interpretable_model(X, params):
        """
        Interpretable model, i.e. the "if-else classifier" one can construct from the interval found.

        :param X: (ndarray) input features.
        :param params: (tuple) classifier parameter (relevent feature index, thresholds, kind of inequality to use)
        :return: (ndarray) prediction.
        """
        idx_features, THs, post_TH_vals = params
        pred_y = np.array(len(X) * [True])
        for n, idx in enumerate(idx_features):

            tmp_sel_x = X[:, idx]
            if post_TH_vals[n] == [1, 0, 0]:

                th_f = THs[n][0]
                tmp_pred_y = (tmp_sel_x <= th_f)

            elif post_TH_vals[n] == [0, 1, 0]:

                th_f = THs[n][0]
                tmp_pred_y = (tmp_sel_x >= th_f)

            elif post_TH_vals[n] == [0, 1, 0, 0]:

                th_f1 = THs[n][0]
                th_f2 = THs[n][1]
                tmp_pred_y = np.logical_and(tmp_sel_x >= th_f1, tmp_sel_x <= th_f2)

            elif post_TH_vals[n] == [1, 0, 1, 0]:

                th_f1 = THs[n][0]
                th_f2 = THs[n][1]
                tmp_pred_y = np.logical_and(tmp_sel_x <= th_f1, tmp_sel_x <= th_f2)

            else:

                tmp_pred_y = pred_y

            pred_y &= tmp_pred_y

        return pred_y

    def read(self, save_as = 'csv'):
        """
        Read operation results in the hdf5 file. For this plugin, the results are organized in the form of a pandas
        dataframe.

        :param save_as: (str) optional, format used to save the result. It can be 'csv' or 'json'.
        """
        res = self.trace.__getattribute__(self.output_dataset[0])
        trace_pos = self.trace.get_trace_pos()
        saving_folder = manage_path(self.trace.trace_readings_path() + os.sep +
                                    InterpretPIandPD.__name__ + '_{}'.format(trace_pos))
        for key in res.keys():

            if 'Label_' in key:

                L = key.split('_')[1]
                saving_folder_by_label = manage_path(saving_folder + os.sep + 'Label_{}'.format(L))
                self._save_df(res[key], saving_folder_by_label, key, save_as)

            else:

                self._save_df(res[key], saving_folder, key, save_as)

    def _save_df(self, df, path, name, format):
        """
        Organize the results in a specific format and save it as dataframe

        :param df: (pandas.DataFrame) dataframe to save.
        :param path: (str) path where the dataframe is saved.
        :param name: (str) saved file name.
        :param format: (str) file format. It can be 'csv' or 'json'
        """
        saving_path = path + os.sep + name + '.' + format
        if format == 'csv':

            df.to_csv(saving_path)

        elif format == 'json':

            df.to_json(saving_path)

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

            tmp_dict.update({key: {'node': self.output_dataset,
                                   'link': list(set(link_list)),
                                   'edge': InterpretPIandPD.__name__}})
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

            key = key - 1

        params_dict.update({str(key): {'op': InterpretPIandPD.__name__,
                                       'p': self.pt.get_current_tracked_params()}})
        previous_content[self.trace.group_name]['ops_parameters'] = params_dict
        with open(self.trace.json_trace_path, 'w') as jfile:

            json.dump(previous_content, jfile, indent=4)