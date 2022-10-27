# Title: 'io_utils'
# Author: Curcuraci L.
# Date: 13/10/2022
#
# Scope: io utils used in bmmltools.

"""
Basic I/O utils used in bmmltools.
"""


#################
#####   LIBRARIES
#################


import numpy as np
import os
import glob
import imageio
import pandas as pd


#################
#####   FUNCTIONS
#################


def write_rrn(array, filepath, filename, description='', additional_field=None):
    """
    Save an array in an rrn file.

    :param array: (ndarray) numpy array to save;
    :param filepath: (str) path where the file is saved (without the file name);
    :param filename: (str) name of the file to save;
    :param description: (str) [optional] description of the array content;
    :param additional_field: (str) [optional] additional comment field. To read this field please use the convention

                                                    <COMMENT NAME> = <COMMENT CONTENT> ,

                             i.e. separate name of the additional comment field and the actual content of the comment
                             with '='. To write more than one additional comment separate two comment lines with a '\n',
                             e.g.

                                <COMMENT NAME 1> = <COMMENT CONTENT 1> \n <COMMENT NAME 2> = <COMMENT CONTENT 2> .
    """
    header = 'This is a readable raw numeric file (.rrn)' \
             '\n' \
             '\nname = {}' \
             '\ntype = {}' \
             '\nshape = {}' \
             '\ndescription = {} '.format(filename, array.dtype, array.shape, description)
    if additional_field is not None:

        header = header + '\n' + additional_field

    header = header + '\ndata_reconstruction_info= reshape (using the C-like index order) the 1-d array ' \
                      'obtained by reading this file as a simple txt (e.g. in python use numpy.reshape method).'
    if filename.find('.rrn') < 0:

        filename = filename + '.rrn'

    np.savetxt(filepath+os.sep+filename, array.reshape(-1), header=header, delimiter=',', comments='#')

def read_rrn(filepath, comments='#', return_comments=False):
    """
    Read an rrn file.

    :param filepath: (str) path to the file (including the filename);
    :param comments: (str) character used to signal the comments in the file.
    :param return_comments: (boolean) if True a dictionary with the rrn comments is returned;
    :return: file content (and comment dictionary eventually).
    """
    comment_lines = {}
    with open(filepath, 'r') as file:

        for currentline in file:

            if currentline[0] == comments:

                if currentline.find('=') > -1:

                    key, value = currentline[1:].split('=')
                    comment_lines.update({key.lstrip().rstrip(): value.lstrip().rstrip()})

            else:

                break

    if 'shape' in comment_lines.keys():

        comment_lines['shape'] = eval(comment_lines['shape'])

    loaded_data = np.loadtxt(filepath, comments=comments, dtype=np.dtype(comment_lines['type']))
    loaded_data = loaded_data.reshape(comment_lines['shape'])
    if not return_comments:

        return loaded_data

    return loaded_data, comment_lines


def create_skan_df(path_to_skan_features,label_df,key,saving_path,saving_name='skan_features_df.json',
                   filter_by=None,feature_file_extension='tif'):
    """
    Create the skan feature dataframe from the feature files produced by the skan software.

    :param path_to_skan_features: (raw str) path to the root folder containing the skan features.
    :param label_df: (pandas.Dataframe) dataframe containing the coordinates in patch space and the corresponding label
                     (typically the output of the Clusterer or ClusterValidator on the trace).
    :param key: (str) name of the key used as label.
    :param saving_path: (str) path to the folder where the file skan feature dataframe is saved.
    :param saving_name: (str) name of the skan feature dataframe.
    :param filter_by: (str) name of the column used to filter the label_df: the filter keep the lines having 1 in this
                      column.
    :param feature_file_extension: (str) file extension of the skan features.
    """
    if filter_by is None:

        df_to_use = label_df.loc[:,['Z','Y','X',key]]

    else:

        df_to_use = label_df[label_df[filter_by]==1]
        df_to_use = df_to_use.loc[:,['Z','Y','X',key]]

    features_name = [name for name in os.listdir(path_to_skan_features) if os.path.isdir(path_to_skan_features+os.sep+name)]
    path_to_subfolders = [path_to_skan_features+os.sep+name for name in features_name]
    all_features = []
    true_features_name = []
    for n,path in enumerate(path_to_subfolders):

        try:

            path_to_feature_tif_file = glob.glob(path+os.sep+'**.{}'.format(feature_file_extension))[0]
            feature = np.array(imageio.volread(path_to_feature_tif_file))
            all_features.append(feature)
            true_features_name.append(features_name[n])

        except:

            continue

    data = []
    for el in df_to_use.to_numpy():

        data_line = []
        Z,Y,X,L = el
        data_line += [Z,Y,X]
        for feature in all_features:

            data_line.append(feature[Z,Y,X])

        data_line += [L,]
        data.append(data_line)

    feature_df = pd.DataFrame(data,columns=['Z','Y','X']+true_features_name+['label',])
    feature_df.dropna(axis=0,inplace=True)
    feature_df.to_json(saving_path+os.sep+saving_name,indent=4)
    print('skan features dataframe saved in {}!'.format(saving_path+os.sep+saving_name))
