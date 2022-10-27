# Convolutional autoencoder as feature extraction step

The scripts contained here cna be used to apply bmmlboard using a convolutional autoencoder in the feature extraction
step (instead of the discrete fourier transform 3D). The scripts have been written as a proof of concept, and a direct 
integration in bmmltools is not available for all the steps at the moment. 

More precisely, the autoencoder training has to be done using a separate script, in the current version of bmmltools.
By the way, once the autoencoder has been trained, the extraction of the features from some binary dataset present on
some trace can be done regularly using the ``PatchTransform3d``, where custom feature extraction functions can be used.
To keep the situation simple, the autoencoder used in the proof of concept has been done at 2d level, rather than 3d.
However, in the final application, the trained 2d autoencoder has been used to label a dataset in 3d. The strategy 
adopted is the following:

* extract features for each 2d (YX-)slice of a 3d patch using the 2d autoencoder;
* average the results with respect to the Z axis. 
  
In this way, one can extract meaningful information from a 3d patch, assuming that the sample does not vary too rapidly 
in the Z-direction. The scripts are organized as follows:

1. ``generate_dataset.py``: script containing the instruction for the random sampling of a dataset of 2d images
   used to train the 2d convolutional autoencoder.

2. ``train.py``: script containing the code for the training od the 2d convolutional autoencoder (a 3d implementation
   of the code is also available there).

3. ``identification.py``: script used to perform manually the raw clustering of a (collection of) slice(s). It is 
   the usual series of operations explained in the tutorials available in the bmmltools documentation, except that
   the feature extraction step is now done using the trained autoencoder (i.e. feature extraction -> dimensional 
   reduction with PCA -> standardization of the dataset -> clustering with HDBSCAN).

4. ``autoencoder_pipeline_example.py``: script containing an example of how the same thing that can be done with the
   ``identification.py`` script can be done using the bmmltools framework, once the autoencoder has been trained. 
