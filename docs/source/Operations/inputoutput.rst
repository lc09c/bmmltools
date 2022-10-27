==
Io
==


In :py:mod:`bmmltools.operations.io` all the input-output methods are collected.

.. _input_section:

Input
=====


:py:mod:`Input <bmmltools.operations.io.Input>` is used to declare the input data on which the trace have to work
*taken from a data object*.


Transfer function
-----------------


None


Initialization and parameters
-----------------------------


In the layer initialization one have to specify:

* the trace on which the this operation act.

The layer parameters of the :py:func:`apply()` method are:

* ``data``: (:py:mod:`bmmltools.operations.core.Data>` object) Data object where the input dataset is stored. The
  dataset name have to be specified in the operation input.


Inputs and outputs
------------------


The operation has the following inputs:

.. collapse:: <b>input 0</b>

   .. epigraph::
      |
      | *description*: Input dataset.
      | *data type*: numpy array or dataframe.
|
The operation has the following outputs:

.. collapse:: <b>output 0</b>

   .. epigraph::
      |
      | *description*: Dataset used as input.
      | *data type*: numpy array or dataframe.
|

InputFromTrace
==============


:py:mod:`InputFromTrace <bmmltools.operations.io.InputFromTrace>` is used to declare the input data on which the trace
have to work *taken from the same trace on which this operation act*.


Transfer function
-----------------


None


Initialization and parameters
-----------------------------


In the layer initialization one have to specify:

* the trace on which the this operation act.

The layer parameters of the :py:func:`apply()` method are:

* ``dataset_name``: (str) Name of the dataset used as input.

* ``dataset_group``: (str) Name of hdf5 group in which the input dataset is located.

Inputs and outputs
------------------


The operation has the following outputs:

.. collapse:: <b>output 0</b>

   .. epigraph::
      |
      | *description*: Dataset used as input.
      | *data type*: numpy array or dataframe.
|

.. _output_raw_labels_section:

OutputRawLabels
===============


:py:mod:`OutputRawLabels <bmmltools.operations.io.OutputRawLabels>` is used to produce the output labelling obtained
from the :ref:`Clustering <cluster_section>` or :ref:`Clustering_HDBSCAN <cluster_HDBSCAN_section>` operations, applied
on the inout dataset.


Transfer function
-----------------


None


Initialization and parameters
-----------------------------


In the layer initialization one have to specify:

* the trace on which the this operation act.

The layer parameters of the :py:func:`apply()` method are:

* ``patch_shape``: (tuple[int]) shape of the patch used to perform the clustering, i.e. the patch used to create the
  patch space.

* ``save_separate_masks``: (bool) optional, if True the mask for each label is saved separately, otherwise a colored
  mask is produced where all the labels are present.


Inputs and outputs
------------------


The operation has the following inputs:

.. collapse:: <b>input 0</b>

   .. epigraph::
      |
      | *description*: Input dataset on which the labelling have to be applied.
      | *data type*: 3d numpy array.
      | *data shape*: :math:`(N_z,N_y,N_x)`, where :math:`N_i` is the number of voxels along the i-th dimension for the
                      operation input.
|

.. collapse:: <b>input 1</b>

   .. epigraph::
      |
      | *description*: Dataframe containing the labelling in the patch space, see output of the
        :ref:`Clustering <cluster_section>` or :ref:`Clustering_HDBSCAN <cluster_HDBSCAN_section>` operation.
      | *data type*: pandas dataframe.
|
The outputs of this operations are saved in the output folders of the trace.


OutputValidLabels
=================


:py:mod:`OutputValidLabels <bmmltools.operations.io.OutputValidLabels>` is used to produce the output labelling obtained
from the :ref:`ClusterValidator <cluster_validator_section>` operation, applied on the inout dataset.


Transfer function
-----------------


None


Initialization and parameters
-----------------------------


In the layer initialization one have to specify:

* the trace on which the this operation act.

The layer parameters of the :py:func:`apply()` method are:

* ``patch_shape``: (tuple[int]) shape of the patch used to perform the clustering, i.e. the patch used to create the
  patch space.

* `` label_kind``: (str) optional, it can be ``'label'`` or ``'RS_label'`` . Kind of label to plot (i.e. the usual one
  or the one identified via rotational similarity)

* ``point_kind``: (str) optional, it can be ``'all'``, ``'core'``, ``'bilayer'`` or ``'boundary'``. Kind of point
  in the patch space used to produce the labelling, according to the classification done by the
  :ref:`ClusterValidator <cluster_validator_section>` operation.

* ``save_separate_masks``: (bool) optional, if True the mask for each label is saved separately, otherwise a colored
  mask is produced where all the labels are present.


Inputs and outputs
------------------


The operation has the following inputs:

.. collapse:: <b>input 0</b>

   .. epigraph::
      |
      | *description*: Input dataset on which the labelling have to be applied.
      | *data type*: 3d numpy array.
      | *data shape*: :math:`(N_z,N_y,N_x)`, where :math:`N_i` is the number of voxels along the i-th dimension for the
                      operation input.
|

.. collapse:: <b>input 1</b>

   .. epigraph::
      |
      | *description*: Dataframe containing the labelling in the patch space, see output of the
        :ref:`ClusterValidator <cluster_validator_section>` operation.
      | *data type*: pandas dataframe.
|
The outputs of this operations are saved in the output folders of the trace.


OutputSegmentation
==================


:py:mod:`OutputSegmentation <bmmltools.operations.io.OutputSegmentation>` is used to produce the output labelling
obtained from the :ref:`RandomForestSegmenter <random_forest_segmentation_section>` operation, applied
on the inout dataset.


Transfer function
-----------------


None


Initialization and parameters
-----------------------------


In the layer initialization one have to specify:

* the trace on which the this operation act.

The layer parameters of the :py:func:`apply()` method are:

* ``use_RS_labels``: (bool) optional, if True the rotationally similar labels are assumed in the rendering. **Note that
  if this is True, the operation assumes that also** :ref:`RandomForestSegmenter <random_forest_segmentation_section>`
  **was trained with these labels** (i.e. setting ``label = 'RS_label'`` in this operation).

* ``save_separate_masks``: (bool) optional, if True the mask for each label is saved separately, otherwise a colored
  mask is produced where all the labels are present.


Inputs and outputs
------------------


The operation has the following inputs:

.. collapse:: <b>input 0</b>

   .. epigraph::
      |
      | *description*: Input dataset on which the labelling have to be applied.
      | *data type*: 3d numpy array.
      | *data shape*: :math:`(N_z,N_y,N_x)`, where :math:`N_i` is the number of voxels along the i-th dimension for the
                      operation input.
|

.. collapse:: <b>input 1</b>

   .. epigraph::
      |
      | *description*: Labelled 3d dataset., see output of the
        :ref:`RandomForestSegmenter <random_forest_segmentation_section>` operation.
      | *data type*: 3d numpy array.
      | *data shape*: :math:`(N_z,N_y,N_x)`, where :math:`N_i` is the number of voxels along the i-th dimension for the
                      operation input.
|

.. collapse:: <b>input 2</b>

   .. epigraph::
      |
      | *description*: Dataframe with the valid clusters, see output of the
        :ref:`ClusterValidator <cluster_validator_section>` operation.
      | *data type*: pandas dataframe.

|
The outputs of this operations are saved in the output folders of the trace.