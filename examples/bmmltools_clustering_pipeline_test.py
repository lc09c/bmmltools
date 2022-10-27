#
#
#
#
#

"""

"""

#################
#####   LIBRARIES
#################


from bmmltools.core.tracer import Trace
from bmmltools.core.data import Data
from bmmltools.operations.io import Input,OutputRawLabels,OutputValidLabels
from bmmltools.operations.feature import Binarizer,PatchDiscreteFourierTransform3D,\
    DimensionalReduction_PCA,DataStandardization
from bmmltools.operations.clustering import Clusterer_HDBSCAN,ClusterValidator,ArchetypeIdentifier,\
    RotationalSimilarityIdentifier


############
#####   MAIN
############


## Load data (uncomment if not data to link is avaialbe)
# sea_urchin_path = r'\\HOME\udata\curcuraci\My Documents\Documents\Science\Data\21_09_21__SeaUrchin\sample1\bin.tif'
# data = Data(working_folder=r'ml_explain3/test/data_num',chunks=(50,50,50))
# data.load_stack(path = sea_urchin_path,dataset_name='sea_urchin')

## Link data (uncomment if a new data to load is avaialbe)
data = Data()
data.link(r'ml_explain3/test/data_num','7759')

## create a trace
trace = Trace(enable_operations_tracking=True)
trace.create(working_folder=r'ml_explain3/test/op_num',group_name='segmenter')
# trace.link(trace_folder=r'ml_explain3/test/op_num/trace_4836',group_name='segmenter')

## machine learning model
x = Input(trace).i('reg_sea_urchin').apply(data)
x = Binarizer(trace).i(x).apply()
x1 = PatchDiscreteFourierTransform3D(trace)\
        .io(x,'post_pdft3d_inference_dataset')\
        .apply(patch_shape=(50,50,50))
x2 = PatchDiscreteFourierTransform3D(trace)\
        .io(x,'post_pdft3d_training_dataset')\
        .apply(patch_shape=(50,50,50),random_patches=True,n_random_patches=200)
x = DimensionalReduction_PCA(trace)\
        .io(x1+x2,['post_dm_pca_inference_dataset'])\
        .apply(inference_key='module',training_key='module',save_model=True)
x = DataStandardization(trace).io(x[0],'post_ds_inference_dataset').apply(axis=(1,0))
x = Clusterer_HDBSCAN(trace)\
        .io([x[0],'post_pdft3d_inference_dataset'],'raw_labels_dataset')\
        .apply(p=dict(min_cluster_size=15,prediction_data=True,metric='euclidean',min_samples=2),save_model=True)
x3 = ClusterValidator(trace).io(x,'valid_labels_dataset').apply()
x = ArchetypeIdentifier(trace)\
        .io(x3+['post_b_dataset',],'archetype_dataset')\
        .apply(patch_shape=(50,50,50),extrapoints_per_dimension=(2,2,2),save_archetype_mask=False)
x = RotationalSimilarityIdentifier(trace).io(x+x3,'post_rsi_dataset').apply()

## intermediate result readings
ClusterValidator(trace).o(x3).read()
RotationalSimilarityIdentifier(trace).o(x+x3).read()

## Outputs
OutputRawLabels(trace).i(['post_b_dataset','raw_labels_dataset']).apply(patch_shape=(50,50,50))
OutputValidLabels(trace).i(['post_b_dataset','valid_labels_dataset']).apply(patch_shape=(50,50,50))

## Save operations
trace.save_operations_dict()