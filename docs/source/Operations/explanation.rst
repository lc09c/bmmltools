===========
Explanation
===========




In :py:mod:`bmmltools.operations.explanation` are collected all the methods which can be used to get an explanation of
some clusters in terms of a series of human-understandable input features.

.. _multicollinearity_reducer_section:

MultiCollinearityReducer
========================


:py:mod:`MultiCollinearityReducer <bmmltools.operations.explanation.MultiCollinearityReducer>` is used to detect and
reduce the multicollinearity in the dataset used for the explanation in order to get a more stable and reliable
explanation in terms of the features selected by the user.


Transfer function
-----------------


Given an input dataset organized in tabular form, where each columns correspond to a feature and each row correspond to
a different data point, the multicollinearity reduction consist in selecting a subset of feature which are not to much
linearly related among each other. This happens by computing for each feature :math:`i` the Variance Inflation Factor
(VIF) defined as

.. math::

    VIF_i = \frac{1}{1-R_i^2}

where :math:`R_i^2` is the usual :math:`R^2` coefficient used to evaluate the goodness of fit of a linear regression
model trained to predict the :math:`i`-th feature in terms of all the other features in the dataset. Once the VIF is
computed for all the features, if some of them are above a certain threshold value, the feature with the highest VIF
is removed from the dataset. This procedure is repeated till the VIFs of all the remaining variables are below the
chosen threshold. The surviving features are approximately linear independent among each other.


Initialization and parameters
-----------------------------


In the layer initialization one have to specify:

* the trace on which the this operation act.

The layer parameters of the :py:func:`apply()` method are:

* ``data_columns``: (list[str]) list containing the name of the columns of the dataframe specified in the input 0 of
  this operations.

* ``target_columns``: (str or list[str]) name(s) of the column(s) containing the target variable(s) to explain.

* ``VIF_th``: (float) threshold one the Variance Inflation Factor.

* ``return_linear_association``: (str) optional, it can None or 'pairwise' or 'full'. When is not None, the linear
  association dictionary is saved as json file in the trace file folder. Two possible linear association are available:

  * *pairwise*, where the coefficients of a linear regression model to predict the value of each eliminated feature in
    term of **one of the feature surviving the multicollinarity screening** are saved.

  * *full*, where the coefficients of a linear regression model to predict the value of each eliminated feature in
    term of **all of the features surviving the multicollinarity screening** are saved.


Inputs and outputs
------------------


The operation has the following inputs:

.. collapse:: <b>input 0</b>

   .. epigraph::
      |
      | *description*: Input dataframe where as set of input features and target variable(s) are stored
      | *data type*: pandas dataframe.
      | *dataframe shape*: :math:`N \times x`, where :math:`N` is the number of different data point from which the
        features are computed, and x is the number of columns for the features plus the number of columns for the
        targets plus possibly other columns.
      | *columns names*: selected by the user, but some of them will be the features and some of them the target(s).
|
The operation has the following outputs:

.. collapse:: <b>output 0</b>

   .. epigraph::
      |
      | *description*: Output dataframe where the surviving features are approximately linear independent.
      | *data type*: pandas dataframe.
      | *data shape*: :math:`N \times M`, where :math:`N` is the number of different data point from which the
        features are computed, and :math:`M` is the number of columns for the features plus the number of columns for
        the targets.
      | *columns names*: names of the selected features and name(s) fo the target(s).
|
.. _explain_with_classifier_section:

ExplainWithClassifier
=====================


:py:mod:`ExplainWithClassifier <bmmltools.operations.explanation.ExplainWithClassifier>` is used to get an explanation
of a target variable using a certain set of features in set theoretic terms (if possible).


Transfer function
-----------------


This operation computes the permutation importance (PI) and the partial dependency (PD) of each feature from a of
classifier trained to recognize a given target label using a set of input feature. To reduce the model dependency of
these quantities are computed for an ensemble of different classifier (trained with different hyperparameters) is used
and mean and standard deviation of each PI and PD are returned (weighted using the model F1 score to recognize the
target variable). By looking to these two quantities one can deduce an explanation for a given cluster variable as
follow:

- From the PI one can deduce the features which are most important for the recognition of a given label as the one
  having the highest positive values.

- From the PD of the most important features identified with the PI, and knowing that the classifiers output 1 when
  when the target variable is recognized, the range of value of a given feature where the PD is high, is likely to be
  the region of values which that feature should have in order to be recognized as belonging to the given label.

In this way one can deduce for each label a series of intervals of a small subset of label which can be used to define
the label in term of human understandable features.


Initialization and parameters
-----------------------------


In the layer initialization one have to specify:

* the trace on which the this operation act.

The layer parameters of the :py:func:`apply()` method are:

* ``test_to_train_ratio``: (float between 0 and 1) optional, fraction of dataset used for the performance evaluation
  of the random forest trained.

* ``model_type``: (str) optional, use just 'RandomForest' for the moment (which is also the default value). In principle
  this should be the model type used to get the explanation, but currently only random forest is available.

* ``n_kfold_splits``: (int) optional, number of train-validation split used to select the best random forest for each
  parameter combination.

* ``n_grid_points``: (int) optional, number of different values of a feature on which the partial dependence is
  computed.

* ``save_graphs``: (bool) optional, if True the graph of the feature permutation importance and the partial dependency
  for each label are saved in the trace file folder.


Inputs and outputs
------------------


The operation has the following inputs:

.. collapse:: <b>input 0</b>

   .. epigraph::
      |
      | *description*: Input dataframe where as set of input features and a target variable is stored. The target
        variable is **always** assumed to be the last column of the dataframe.
      | *data type*: pandas dataframe.
      | *dataframe shape*: :math:`N \times x`, where :math:`N` is the number of different data point from which the
        features are computed, and x is the number of columns for the features plus one.
      | *columns names*: names of the features and name of the target.
|
The operation has the following outputs:

.. collapse:: <b>output 0</b>

   .. epigraph::
      |
      | Dictionary. The key below is present.

      .. collapse:: <i>classifier_f1_scores</i>

         .. epigraph::
               |
               | *description*: Dataframe containing the F1 score of the classifier trained for the various target.
               | *data type*: pandas dataframe.
               | *dataframe shape*: :math:`N \times 1`, where :math:`N` is the number of labels.
               | *columns names*: for each label Label_x, where x is label.
               | *columns description: average F1 score for the trained classifiers, which can be used to deduce if the
                 input features contains enough information to learn to recognize the various labels.
      |
      For each label the key below are present:

      .. collapse:: <i>PI_Label_x</i> where x is the label

         .. epigraph::
               |
               | *description*: Dataframe containing the permutation importance for the considered label.
               | *data type*: pandas dataframe.
               | *dataframe shape*: :math:`N \times 2`, where :math:`N` is the number of features used.
               | *columns names*: estimated_permutation_importance_mean, estimated_permutation_importance_std.
               | *columns description: mean and standard deviation of the permutation importance.
      |
      For each label and feature the key below is present:

      .. collapse:: <i>PD_Label_x_feature_f</i> where x is the label and f the feature

         .. epigraph::
               |
               | *description*: Dataframe containing the partial dependency of a given feature for the considered label.
               | *data type*: pandas dataframe.
               | *dataframe shape*: :math:`N \times 2`, where :math:`N` is the number of features used.
               | *columns names*: estimated_partial_dependency_mean, estimated_partial_dependency_std.
               | *columns description: mean and standard deviation of the partial dependency of a given feature.
|
.. _interpret_pi_and_pd_section:

InterpretPIandPD
================


:py:mod:`InterpretPIandPD <bmmltools.operations.explanation.InterpretPIandPD>` is used to get automatically an
interpretation of the PI and PD (computed with the operation :ref:`ExplainWithClassifier
<explain_with_classifier_section>`) in terms of simple intervals of the of features.


Transfer function
-----------------


This operation tries to perform the following two operations:

1. *Estimate the most relevant features for the definition of a label*. This is done by sorting the features in
   decreasing order with respect the permutation importance of the label. The most important features are selected by
   taking the sorted features till the total positive permutation importance reach a certain threshold set by the user
   is reached.

2. *Estimate the intervals of the most relevant features which can be used to define the label*. This is done using an
   histogram based estimation techniques, since no a priori functional behavior can be assumed for the partial
   dependency. This estimation techniques work as follow: an histogram is contracted from the values of the partial
   dependency and then the Otsu threshold for this histogram is computed. The normal Otsu threshold is used if the
   histogram has just two peaks, when more than two peaks are detected an Otsu multithreshold technique is used and the
   highest threshold is considered. The intervals one is looking for are the one where the partial dependency is above
   the found threshold.

The intervals found in the point 2, can be used to define an interpretable model, i.e. a model where the classification
can be understood in terms of if-else condition on the input features (which are assumed human understandable). However,
the result the interval defined in the point 2 it is likely to be suboptimal, i.e. the accuracy of the interpretable
model is not maximal. For a limited number of cases (i.e. for intervals which can be defined with one or two threshold)
a bayesian optimization procedure which find the best intervals, in the sense of maximize the (balanced) accuracy of the
interpretable model.


Initialization and parameters
-----------------------------


In the layer initialization one have to specify:

* the trace on which the this operation act.

The layer parameters of the :py:func:`apply()` method are:

* ``positive_PI_th``: (float between 0 an 1) optional, fraction of positive permutation importance which the most
  relevant features need to explain in order to be selected for the explanation of a given label.

* ``n_bins_pd_mean``: (int) optional, number of bins of the partial dependency used in the histogram based detection
  method for the interval definition.

* ``prominence``: (None or float) optional, prominence parameter of the peak finder algorithm used in the histogram
  based detection method for the interval definition.

* ``adjust_accuracy``: (bool) optional, if True the adjusted balanced accuracy is used to compute the interpretable
  model performance

* ``bayes_optimize_interpretable_model``: (bool) if True the interval deduced using the histogram based detection method
  are optimized using bayesian optimization in order to maximize the accuracy.

* ``bo_max_iter``: (int) optional, maximum number of iterations of the bayesian optimization.

* ``save_interpretable_model``: (bool) optional, if True the interpretable model is saved.



Inputs and outputs
------------------


The operation has the following inputs:

.. collapse:: <b>input 0</b>

   .. epigraph::
      |
      | *description*: Output of the :ref:`ExplainWithClassifier <explain_with_classifier_section>` operation.
      | *data type*: dictionary.

.. collapse:: <b>input 1</b>

   .. epigraph::
      |
      | *description*: Output of the :ref:`MultiCollinearityReducer <multicollinearity_reducer_section>` operation.
      | *data type*: pandas dataframe.
|
The operation has the following outputs:

.. collapse:: <b>output 0</b>

   .. epigraph::
      |
      | Dictionary. The key below is present.

      .. collapse:: <i>Feature_relevance</i>

         .. epigraph::
               |
               | *description*: Dataframe containing information about the feature relevance for all the labels.
               | *data type*: pandas dataframe.
               | *dataframe shape*: :math:`N \times M`, where :math:`N` is the number of labels and M the number of
                 features.
               | *columns names*: features name.
               | *columns description: for each label (row of the dataframe) as 1 is present if the feature is relevant
                 for the label description and 0 otherwise.

      .. collapse:: <i>interpretation_accuracy</i>

         .. epigraph::
               |
               | *description*: Dataframe containing the accuracy of the interpretable model.
               | *data type*: pandas dataframe.
               | *dataframe shape*: :math:`N \times 1`, where :math:`N` is the number of labels.
               | *columns names*: balanced_accuracy (or balanced_adjusted_accuracy, depending on the user input).
               | *columns description: for each label (row of the dataframe) the balanced (adjusted) accuracy is stored.
      |
      For each label the key below are present:

      .. collapse:: <i>Label_x_feature_f_interval</i> where x is the label and f is the name of a feature relevant for
                    the label x.

         .. epigraph::
               |
               | *description*: Interval of the fearure f for the definition of the label l .
               | *data type*: pandas dataframe.
               | *dataframe shape*: :math:`2 \times x`, where :math:`x` is the number of threshold used to define the
                 interval.
               | *columns names*: thresholds, post_thresholds_value.
               | *columns description*: in the 'thresholds' column the value where the thresholds defining the interval,
                 while in the 'post_thresholds_value' the value assumed by an indicator function *after* the threshold.
      |