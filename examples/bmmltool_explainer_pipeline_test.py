#
#
#
#
#

"""

"""

#### LIBRARIES

import os
import numpy as np
import h5py
import pandas as pd

from bmmltools.core.data import Data
from bmmltools.core.tracer import Trace
from bmmltools.operations.io import Input
from bmmltools.operations.explanation import MultiCollinearityReducer,ExplainWithClassifier,InterpretPIandPD

#### MAIN


## Load data
data = Data(working_folder=r'ml_explain3/test/data2')
data.load_pandas_df_from_json(r'ml_explain3/skan_features.json','skan_features',drop_columns=['cube_space_coord'])

## create a trace
trace = Trace()
trace.create(working_folder=r'ml_explain3/test/op',group_name='explainer')

## machine learning model
x = Input(trace).i('skan_features').apply(data)
x = MultiCollinearityReducer(trace).io(x,'post_mcr_dataset').apply(data_columns = ['bd','belsd','blsd','bmel','bml','bmt',
                                                                          'epd','mjt','mjtsd','nj','phim','phisd',
                                                                          'sa','sad','ssa','thetam','thetasd','tmt',
                                                                          'tmtsd','tth','tthsd','vf'],
                                                                    target_columns = ['RI_label'],
                                                                    VIF_th= 5,
                                                                    return_linear_association='full')
x_ref = x
x = ExplainWithClassifier(trace).io(x,'post_ewc_dataset').apply(save_graphs=True)
x = InterpretPIandPD(trace).io(x+x_ref,'label_interpretation').apply(bayes_optimize_interpretable_model=True,
                                                                     save_interpretable_model=True)
## intermediate result readings
MultiCollinearityReducer(trace).o('post_mcr_dataset').read()
ExplainWithClassifier(trace).o('post_ewc_dataset').read()
InterpretPIandPD(trace).o('label_interpretation').read()