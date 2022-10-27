============
Segmentation
============


In :py:mod:`bmmltools.operations.segmentation` all the feature segmentation methods are collected. Segmentation methods
in this model are assumed to work at voxel level.

.. _random_forest_segmentation_section:

RandomForestSegmenter
=====================


:py:mod:`RandomForestSegmenter <bmmltools.operations.segmentation.RandomForestSegmenter>` is used to produce a
segmentation at voxel level by training a random forest in a supervised manner to assign to each voxel a given
label.


Transfer function
-----------------


The segmentation is done by computing a set of feature (see below) from the data to label and train a random forest
in order to classify different voxels in different labels, using the valid labels identified by the `ClusterValidator
<cluster_validator_section>`_ after clustering. The feature used are the quantities below:

* intensity feature, i.e. the input data convolved with a gaussian kernel (1 feature).

* edge features, i.e. the sobolev filter applied to the input data convolved with a gaussian kernel (1 feature).

* texture features, i.e. the eigenvalues of the Hessian matrix computed at each voxels computed
  using the input data convolved with a gaussian kernel (3 features).

* directional features, i.e. the elements of the structure matrix computed at each voxel computed using the
  input data convolved with a gaussian kernel (6 features, see `here <https://scikit-image.org/docs/stable/api/skimage.
  feature.html#skimage.feature.structure_tensor>`_ for more details).

The user can select which feature use or not. All these features are computed at different scales, i.e. by convolving
the input data with a gaussian kernel having different :math:`\sigma` (scale parameter).


Initialization and parameters
-----------------------------


In the layer initialization one have to specify:

* the trace on which the this operation act.

The layer parameters of the :py:func:`apply()` method are:

* ``patch_shape``: (tuple[int]) shape of the patch used to define the valid clusters given in the input 0.

* ``label_type``: (str) optional, it can be ``'label'`` or ``'RS_label'`` (for rotationally similar labelling if
  available).

* ``reference_points``: (str) optional, it can be ``None``, ``'core_point'``, ``'bilayer_point'`` or
  ``'boundary_point'``. It is kind of point in the patch space used to define the labels on which the random forest
  classifier is trained. The default value is ``'core_point'``.

* ``sigma_min``: (float) optional, minimum scale at which the features are computes for the training of the random
  forest classifier.

* ``sigma_max``: (float) optional, maximum scale at which the features are computes for the training of the random
  forest classifier.


* ``n_sigma``: (int) optional, number of different scales at which the features for the training of the random forest
  classifier are computed.

* ``intensity``: (bool) optional, if True the "intensity features" are computed and used for the training of the random
  forest. By "intensity features" the input data convolved with a gaussian kernel having sigma equal to the scale
  parameters specified by the user is understood.

* ``edge``: (bool) optional, if True the "edge features" are computed and used for the training of the random
  forest. By "edge features" the the Sobolev filter applied to the input data convolved with a gaussian kernel having
  sigma equal to the scale parameters specified by the user is understood.

* ``texture``: (bool) optional, if True the "texture features" are computed and used for the training of the random
  forest. See `here <https://scikit-image.org/docs/stable/api/skimage.feature.html#
  skimage.feature.multiscale_basic_features>`_ for more about this kind of features.

* ``direction``: (bool) optional, if True the "direction features" are computed and used for the training of the random
  forest. See `here <https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.structure_tensor>`_
  for more about this kind of features.

* ``N_training_samples_per_label``: (int) optional, number of voxels with all the computed features (i.e. number of
  samples) used to train *each* estimator (decision tree) of the random forest for each label.

* ``inference_split_shape``: (tuple[int]) number of split of the input dataset per dimension used to do the inference.
  This is needed especially when the input data *plus* all the features computed cannot fit in RAM.

* ``n_estimators``: (int) optional, number of estimators (decision tree) used in the random forest classifier.

* ``save_trained_random_forest``: (bool) if True the trained random forest is saved in the trace file folder.


Inputs and outputs
------------------


The operation has the following inputs:

.. collapse:: <b>input 0</b>

   .. epigraph::
      |
      | *description*: Input 3d dataset to segment.
      | *data type*: 3d numpy array.
      | *data shape*: :math:`(N_z,N_y,N_x)`, where :math:`N_i` is the number of voxels along the i-th dimension.

.. collapse:: <b>input 1</b>

   .. epigraph::
      |
      | *description*: Table containing only the valid clusters and information about the kind of point in the
        patch space (i.e. the typical output of :ref:`ClusterValidator <cluster_validator_section>`).
      | *data type*: pandas dataframe.
      | *dataframe shape*: :math:`N \times 10`, where :math:`N` is the number of different row  contained in the
        dataframe specified in the input 0.
      | *columns names*: Z, Y, X, label, core_point, bilayer_point, z_bilayer, y_bilayer, x_bilayer, boundary_point.
      | *columns description*: the first three columns are the z/y/x coordinate in patch space of the feature used to
        produce the clustering, while the cluster label is saved in the label columns. After that, information about
        the nature of the the point in the patch space, is indicated with a 1 in the corresponding column (and 0
        otherwise).
|
The operation has the following outputs:

.. collapse:: <b>output 0</b>

   .. epigraph::
      |
      | *description*: Labelled 3d dataset. **Note that 1 is added to all the labels, since 0 is also typically the
        convention about empty space in the data to segment**.
      | *data type*: 3d numpy array.
      | *data shape*: :math:`(N_z,N_y,N_x)`, where :math:`N_i` is the number of voxels along the i-th dimension for the
                      operation input.
|