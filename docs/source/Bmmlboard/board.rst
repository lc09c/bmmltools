=========
bmmlboard
=========

bmmlboard can be used to visualize intermediate results stored on trace. Not all the intermediate result can be
visualized via bmmlboard, since it is not always possible to find an useful representation of the result without
knowing the specific scope of the visualization. In this case the user need to access to these data directly from the
trace if a visualization is needed.

To run bmmlboard, simpy run the Ananconda propt and in the same python environment where bmmltools is installed
simply run

.. code::

    > python -m bmmltools.run_bmmlboard

It is a streamlit based app (therefore it need a web browser to work) and organized in 3 different module. Once
bmmltools open, specify in the main page the **absolute** path of the trace to inspect. On the menu' on the left the
available modules are listed. By clicking on one of them the different visualization tools can be used. For all the
modules one need to specify the group name used in the trace to store the intermediate result. Some module
require to select the dataset used as input data.


Label Visualizer
================


It is the module used to visualize the intermediate result produced by the operations listed below:

* :ref:`PatchDiscreteFourierTransform3d <patch_dft_3d_section>`;

* :ref:`Clusterer <cluster_section>`;

* :ref:`Clusterer_HDBSCAN <cluster_HDBSCAN_section>`;

* :ref:`ClusterValidator <cluster_validator_section>`;

* :ref:`ArchetypeIdentifier <archetype_identifier_section>`;

* :ref:`RotationalSimilarityIdentifier <rotational_similarity_identifier_section>`.


Segmentation Visualizer
=======================


It is the module used to visualize the intermediate result produced by the operations listed below:

* :ref:`RandomForestSegmenter <random_forest_segmentation_section>`.


Explainer Visualizer
====================

It is the module used to visualize the intermediate result produced by the operations listed below:

* :ref:`MultiCollinearityReducer <multicollinearity_reducer_section>`;

* :ref:`ExplainWithClassifier <explain_with_classifier_section>`;

* :ref:`InterpretPIandPD <interpret_pi_and_pd_section>`.