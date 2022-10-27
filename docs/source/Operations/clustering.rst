==========
Clustering
==========


In :py:mod:`bmmltools.operations.clustering` all the clustering related methods are collected.

.. _cluster_section:

Clusterer
=========


:py:mod:`Clusterer <bmmltools.operations.clustering.Clusterer>` is the generic methods which can be used to apply some
clustering algorithm available in scikit-learn (see `here <https://scikit-learn.org/stable/modules/classes.html
#module-sklearn.cluster>`_) to some dataset present on a trace. Two pre-initialized clusterer are available:

* :py:mod:`Clusterer_KMean <bmmltools.operations.clustering.Clusterer_KMean>` where the K-Mean clustering is used;

* :py:mod:`Clusterer_DBSCAN <bmmltools.operations.clustering.Clusterer_DBSCAN>` where the DBSCAN clustering is used.


Transfer function
-----------------


Given an input array :math:`I = \lbrace i_l \rbrace_{l \in [0,1,\cdots,N-1]}` with shape :math:`(N,x)`, where :math:`N`
is the number of different data point in the input dataset while :math:`x` is the number of features on which the
clustering algorithm is applied, the layer outputs the sequence of labels
:math:`\lbrace o[l] \rbrace_{l \in [0,1,\cdots,N]}` where

.. math::

   o[l] = \mbox{CLU}(i_l)

where :math:`\mbox{CLU}` is the chosen clustering algorithm.


Initialization and parameters
-----------------------------


In the layer initialization one have to specify:

* the trace on which the this operation act;

* a clustering class among the scikit-learn clustering methods, or a scikit-learn compatible clustering class. This
  initialization parameter do not need to be specified when one uses :py:mod:`Clusterer_KMean <bmmltools.operations.
  clustering.Clusterer_KMean>` or :py:mod:`Clusterer_DBSCAN <bmmltools.operations.clustering.Clusterer_DBSCAN>`.

The layer parameters of the :py:func:`apply()` method are:

* ``p``: (dict) dictionary containing the initialization parameters of the clustering algorithm.


Inputs and outputs
------------------


The operation has the following inputs:

.. collapse:: <b>input 0</b>

   .. epigraph::
      |
      | *description*: Input dataset on which the clustering is performed.
      | *data type*: numpy array.
      | *data shape*: :math:`(N,x)`, where :math:`N` is the number of different data point in the input dataset
        while :math:`x` is the number of features on which the clustering algorithm is applied.

.. collapse:: <b>input 1</b>

   .. epigraph::
      |
      | *description*: Input dataframe where the 3d coordinate in the patch space of the features in the input 0
        are stored.
      | *data type*: pandas dataframe.
      | *dataframe shape*: :math:`N \times 3`, where :math:`N` is the number of different data point contained in the
        array specified in the input 0.
      | *columns names*: Z, Y, X.
      | *columns description*: z/y/x coordinate in patch space of the features contained in the input 0. The
        correspondence between the three coordinates and the features has to be understood row-by-row, i.e. the
        i-th index of the dataframe row correspond to the i-th element along the 0 axis of the array specified in
        the input 0.
|
The operation has the following outputs:

.. collapse:: <b>output 0</b>

   .. epigraph::
      |
      | *description*: dataset with the clustering result.
      | *data type*: pandas dataframe.
      | *dataframe shape*: :math:`N \times 4`, where :math:`N` is the number of different data point contained in the
        array specified in the input 0.
      | *column names*: Z, Y, X, label.
      | *column description*: the first 3 columns are the z/y/x coordinate in patch space of the input feature, while
        in the label column the result of the clustering algorithm (i.e. the label associated to the clusters found) is
        stored.
|
.. _cluster_HDBSCAN_section:

Clusterer HDBSCAN
=================


:py:mod:`Clusterer_HDBSCAN <bmmltools.operations.clustering.Clusterer_HDBSCAN>` is used to apply the HDBSCAN
clustering algorithm (see `here <https://hdbscan.readthedocs.io/en/latest/index.html>`_) to some dataset present on a
trace.


Transfer function
-----------------


Given an input array :math:`I = \lbrace i_l \rbrace_{l \in [0,1,\cdots,N-1]}` with shape :math:`(N,x)`, where :math:`N`
is the number of different data point in the input dataset while :math:`x` is the number of features on which the
clustering algorithm is applied, the layer outputs the sequence of labels
:math:`\lbrace o[l] \rbrace_{l \in [0,1,\cdots,N]}` where

.. math::

   o[l] = \mbox{argmax}(\mbox{softHDBSCAN}(i_l))

where :math:`\mbox{softHDBSCAN}` is the HDBSCAN clustering algorithm used in soft clustering mode (i.e. assigning the
each data point the probability to belong to *each* cluster rather than the cluster label itself). The cluster assigned
to a given input :math:`i_l` is the one with the highest probability.


Initialization and parameters
-----------------------------


In the layer initialization one have to specify:

* the trace on which the this operation act.

The layer parameters of the :py:func:`apply()` method are:

* ``p``: (dict) dictionary containing the initialization parameters of the HDBSCAN clustering algorithm.

* ``save_model``: (bool) optional, if True the trained HDBSCAN clustering algorithm is saved using joblib.

* ``trained_model_path``: (str) optional, if not None the HDBSCAN algorithm is loaded from a joblib file. Therefore the
  algorithm is not trained but the loaded model is used instead (which is assumed already trained). Note that when this
  option is used it is not ensured that the clustering algorithm will work for any combination of initialization
  parameters used during the training.


Inputs and outputs
------------------


The operation has the following inputs:

.. collapse:: <b>input 0</b>

   .. epigraph::
      |
      | *description*: Input dataset on which the clustering is performed.
      | *data type*: numpy array.
      | *data shape*: :math:`(N,x)`, where :math:`N` is the number of different data point in the input dataset
        while :math:`x` is the number of features on which the clustering algorithm is applied.

.. collapse:: <b>input 1</b>

   .. epigraph::
      |
      | *description*: Input dataframe where the 3d coordinate in the patch space of the features in the input 0
        are stored.
      | *data type*: pandas dataframe.
      | *dataframe shape*: :math:`N \times 3`, where :math:`N` is the number of different data point contained in the
        array specified in the input 0.
      | *columns names*: Z, Y, X.
      | *columns description*: z/y/x coordinate in patch space of the features contained in the input 0. The
        correspondence between the three coordinates and the features has to be understood row-by-row, i.e. the
        i-th index of the dataframe row correspond to the i-th element along the 0 axis of the array specified in
        the input 0.
|
The operation has the following outputs:

.. collapse:: <b>output 0</b>

   .. epigraph::
      |
      | *description*: dataset with the clustering result.
      | *data type*: pandas dataframe.
      | *dataframe shape*: :math:`N \times 4`, where :math:`N` is the number of different data point contained in the
        array specified in the input 0.
      | *column names*: Z, Y, X, label.
      | *column description*: the first 3 columns are the z/y/x coordinate in patch space of the input feature, while
        in the label column the result of the clustering algorithm (i.e. the label associated to the clusters with the
        highest probability) is stored.
|
.. _cluster_validator_section:
ClusterValidator
================


:py:mod:`ClusterValidator <bmmltools.operations.clustering.ClusterValidator>` is used validate the clustering obtained
from the clustering algorithm available in bmmltools. The clustering algorithm does not take explicitly into account
the spatial requirements which a true cluster should have, like the spatial continuity and a sufficiently big volume.
This operation check that.


Transfer function
-----------------


Given an input table where the 3d space coordinates in the patch space of a given label are listed, the validity of the
label assigned to a given point in the patch space is checked with the 3 following criteria:

* labels are valid if they are sufficiently continuous in patch space, i.e. they survive to a binary erosion followed by
  a binary dilation in patch space (eventually filling the holes remaining inside the labels);

* after the erosion-dilation process a cluster is considered valid if it has a volume bigger than a given threshold (and
  similarly is checked checked of the volume of the core part of the cluster is above a certain threshold);

* after the erosion-dilation process a point in the patch space is valid if it is assigned to just one label.

The points are also classified in 3 categories:

* *core point* of a cluster, i.e. the points in the patch space which are not at the boundary with another label. These
  point are **assumed** to contain a good representation of the component defining the cluster. Core points are defined
  by eroding 1 time the valid point of the cluster.

* *bilayer points* of a cluster, i.e. the points in the patch space having spatial continuity in at least two of the
  three dimensions (these points are not considered if the validation is done in 2D mode).

* *boundary points* of a cluster, i.e. the points which valid but are not cor or bilayer points. These points are
  assumed to be unreliable to study a cluster, since they should contain mixture of different components (being at the
  boundary of the cluster).


Initialization and parameters
-----------------------------


In the layer initialization one have to specify:

* the trace on which the this operation act.

The layer parameters of the :py:func:`apply()` method are:

* ``patch_space_volume_th``: (int) optional, minimum volume a cluster should have in the patch space to considered as
  a valid cluster (default value is 1);

* ``patch_space_core_volume_th``: (int) optional, minimum volume the core part of a cluster should have in the patch
  space to considered as a valid cluster (default value is 1).


Inputs and outputs
------------------


The operation has the following inputs:

.. collapse:: <b>input 0</b>

   .. epigraph::
      |
      | *description*: Table where coordinates in patch space and corresponding labels are listed.
      | *data type*: pandas dataframe.
      | *data shape*: :math:`N \times 4`, where :math:`N` is the number of data points.
      | *columns names*: Z, Y, X, label.
      | *columns description*: z/y/x coordinate in patch space of the transformed patches contained in the
        *transformed_patch* array. The correspondence between the three coordinates and the transformed patch
        has to be understood row-by-row, i.e. the i-th index of the dataframe row correspond to the i-th element
        along the 0 axis of the *transformed patch* array.
|
The operation has the following outputs:

.. collapse:: <b>output 0</b>

   .. epigraph::
      |
      | *description*: Table containing *only the valid clusters* and information about the kind of point in the
        patch space.
      | *data type*: pandas dataframe.
      | *dataframe shape*: :math:`N \times 10`, where :math:`N` is the number of different row  contained in the
        dataframe specified in the input 0.
      | *columns names*: Z, Y, X, label, core_point, bilayer_point, z_bilayer, y_bilayer, x_bilayer, boundary_point.
      | *columns description*: the first three columns are the z/y/x coordinate in patch space of the feature used to
        produce the clustering, while the cluster label is saved in the label columns. After that, information about
        the nature of the the point in the patch space, is indicated with a 1 in the corresponding column (and 0
        otherwise).

|
.. _archetype_identifier_section:

ArchetypeIdentifier
===================


:py:mod:`ArchetypeIdentifier <bmmltools.operations.clustering.ArchetypeIdentifier>` is used to define a set of
representative for each cluster found. The elements belonging to this set of representative are called archetypes
of the clusters. They can be used to study the properties of a given cluster using less computational resources.


Transfer function
-----------------


The archetype are defined by sampling a region of the patch space (suitably expanded if needed) according to the
probability distribution constructed by normalizing the distance transform of the region in the (expanded) patch space
corresponding to a given cluster. The sampling region is the one having probability above a given threshold selected
by the user.


Initialization and parameters
-----------------------------


In the layer initialization one have to specify:

* the trace on which the this operation act.

The layer parameters of the :py:func:`apply()` method are:

* ``patch_shape``: (tuple[int]) shape of the patch used to define the features used for the clustering.

* ``archetype_threshold``: (float between 0 and 1) optional, threshold used on the normalized distance transform of each
  cluster to define the sampling region for the archetypes. The sampling region for a given cluster

* ``N_archetype``: (int) optional, number of archetype per cluster.

* ``extrapoints_per_dimension``: (tuple[int]) optional, dilation factor for each dimension of the patch space used to
  have a more realistic spatial region for the sampling of the archetypes.

* ``filter_by_column``: (str) optional, when this field is not None, this is name of the column to filter dataframe
  specified in the input dataset. Only the points having 1 in this column are used to define the archetypes of each
  cluster. When this field is None, all dataframe given in the input is used.

* ``save_archetype_mask``: (bool) optional, if True a mask showing for each cluster the actual region sampled for the
  archetype is saved in the trace file folder.


Inputs and outputs
------------------


The operation has the following inputs:


.. collapse:: <b>input 0</b>

   .. epigraph::
      |
      | *description*: Table containing *only the valid clusters* and information about the kind of point in the
        patch space.
      | *data type*: pandas dataframe.
      | *dataframe shape*: :math:`N \times 10`, where :math:`N` is the number of different row  contained in the
        dataframe specified in the input 0.
      | *columns names*: Z, Y, X, label, core_point, bilayer_point, z_bilayer, y_bilayer, x_bilayer, boundary_point.
      | *columns description*: the first three columns are the z/y/x coordinate in patch space of the feature used to
        produce the clustering, while the cluster label is saved in the label columns. After that, information about
        the nature of the the point in the patch space, is indicated with a 1 in the corresponding column (and 0
        otherwise).

.. collapse:: <b>input 1</b>

   .. epigraph::
      |
      | *description*: Dataset from which the archetype are sampled.
      | *data type*: 3d numpy array.
      | *data shape*: :math:`(N_z,N_y,N_x)`, where :math:`N_i` is the number of voxels along the i-th dimension.
|
The operation has the following outputs:

.. collapse:: <b>output 0</b>

   .. epigraph::
      |
      | Dictionary with keys:

      .. collapse:: <i>archetype</i>

         .. epigraph::
               |
               | *description*: Dataset containing the archetypes of all the labels.
               | *data type*: numpy array.
               | *data shape*: :math:`(N,P_z,P_y,Px)`, where :math:`(P_z,P_y,Px)` is the patch shape and :math:`N` is
                 total number of archetype is equal to the number of archetype (parameter selected by the user)
                 multiplied by the number of valid clusters.

      .. collapse:: <i>archetype_patch_coordinates</i>

         .. epigraph::
            |
            | *description*: Table where coordinates in patch space and corresponding labels of the sampled archetype
              are listed.
            | *data type*: pandas dataframe.
            | *data shape*: :math:`N \times 4`, where :math:`N` is the number of data points.
            | *columns names*: Z, Y, X, label.
            | *columns description*: The first three columns are z/y/x coordinate in patch space of sampled archetype,
              while in the label column the corresponding label is stored. The correspondence between the three
             coordinates and le label and the archetypes stored in the *archetype* key of the output dictionary
              has to be understood row-by-row, i.e. the i-th index of the dataframe row correspond to the i-th element
              along the 0 axis of the archetype array.

|
.. _rotational_similarity_identifier_section:

RotationalSimilarityIdentifier
==============================


:py:mod:`RotationalSimilarityIdentifier <bmmltools.operations.clustering.RotationalSimilarityIdentifier>` is used to
suggest possible identification of the clusters based on similarity under rotation of two labels.


Transfer function
-----------------


Two labels are considered similar under rotation when the procedure describe below give a positive result.

1. All the archetypes of two different clusters, say A and B, are taken and evaluated in the spherical coordinates.

2. The radial distribution of each archetype is computed by integrating over the angles for both the clusters.

3. The mean value and the covariance matrix of the radial distribution is computed for both clusters.

4. The one-value Hotelling T-test is run to see if the archetype of the cluster A can be sampled from a probability
   distribution with mean value B

5. The one-value Hotelling T-test is run to see if the archetype of the cluster B can be sampled from a probability
   distribution with mean value A

6. When both test performed at point 4 and 5 give positive answer, the identification of the two labels is suggested.

After that the identification test is executed the angles among two clusters are computed by looking at the mean
modulus of the 3d DFT of archetypes considered and performing a weighted correlation among the angular parts of the
archetype at various radii. More about the rotational identification procedure and the angles among different archetypes
can be found in [LINK TO PAGE IN THE MISCELLANEOUS SECTION OR WHATEVER] section.


Initialization and parameters
-----------------------------


In the layer initialization one have to specify:

* the trace on which the this operation act.

The layer parameters of the :py:func:`apply()` method are:

* ``p_threshold``: (float between 0 and 1) optional, threshold below which two cluster are considered rotationally
  similar.

* ``smooth``: (bool) optional, if True the modulus of the 3d DFT is smoothed using a gaussian filter.

* ``sigma``: (float) optional, standard deviation of the gaussian filter used to smooth the modulus of the 3ed DFT.

* ``spherical_coordinates_shape``: (tuple[int]) optional, shape of the modulus of the 3d DFT once evaluated in spherical
  coordinates. The :math:`\rho\theta\phi`-ordering is assumed.

* ``bin_used_in_radial_dist``: (tuple[int]) optional, tuple indicating the start and stop bin of the radial distribution
  considered for the statistical test.


Inputs and outputs
------------------


The operation has the following inputs:

.. collapse:: <b>input 0</b>

   .. epigraph::
      |
      | *description*: Dictionary containing the output of the :ref:`ArchetypeIdentifier <archetype_identifier_section>`.
        Refer to that for more detail.

.. collapse:: <b>input 1</b>

   .. epigraph::
      |
      | *description*: Table containing the valid clusters and information about the kind of point in the
        patch space.
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
      | Dictionary with keys:

      .. collapse:: <i>identification_probability</i>

         .. epigraph::
               |
               | *description*: Table where the identification probability of each pair of labels is written.
               | *data type*: pandas dataframe.
               | *data shape*: :math:`N_l \times N_l`, where :math:`N_l` is the number of valid labels.
               | *columns names*: for each cluster with label :math:`x` a column called ``Label_:math:`x``` is present.

      .. collapse:: <i>test_statistical_power</i>

         .. epigraph::
               |
               | *description*: Table where the power of the statistical test performed among the two labels is written.
               | *data type*: pandas dataframe.
               | *data shape*: :math:`N_l \times N_l`, where :math:`N_l` is the number of valid labels.
               | *columns names*: for each cluster with label :math:`x` a column called ``Label_:math:`x``` is present.

      .. collapse:: <i>ll_theta_angles_df</i>

         .. epigraph::
               |
               | *description*: Table where theta angle among each pair of labels is written. Clearly this number make
                 sense only for the pairs that can be identified.
               | *data type*: pandas dataframe.
               | *data shape*: :math:`N_l \times N_l`, where :math:`N_l` is the number of valid labels.
               | *columns names*: for each cluster with label :math:`x` a column called ``Label_:math:`x``` is present.

      .. collapse:: <i>ll_phi_angles_df</i>

         .. epigraph::
               |
               | *description*: Table where phi angle among each pair of labels is written. Clearly this number make
                 sense only for the pairs that can be identified.
               | *data type*: pandas dataframe.
               | *data shape*: :math:`N_l \times N_l`, where :math:`N_l` is the number of valid labels.
               | *columns names*: for each cluster with label :math:`x` a column called ``Label_:math:`x``` is present.

      .. collapse:: <i>identification_df</i>

         .. epigraph::
               |
               | *description*: Table where if two labels can be identified or not according to the user setting is
                 written.
               | *data type*: pandas dataframe.
               | *data shape*: :math:`N_l \times N_l`, where :math:`N_l` is the number of valid labels.
               | *columns names*: for each cluster with label :math:`x` a column called ``Label_:math:`x``` is present.

|
In addition the dataframe specified by the input 1 a column called ``RS_label`` is added where the labels considered
similar under rotation by the algorithm are identified. The other columns of this dataframe are left unchanged.
