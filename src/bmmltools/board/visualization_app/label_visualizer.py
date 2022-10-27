# Title: 'label_visualizer.py'
# Author: Curcuraci L.
# Date: 19/10/2022
#
# Scope: visualization app for the identification step (clustering).

"""
Visualization app for PatchDiscreteFourierTransform3D,  Clusterer, ClusterValidator, RotationalSimilarityIdentifier, and
ArchetypeIdentifier operations.
"""


def run():


    #################
    #####   LIBRARIES
    #################


    import os
    import streamlit as st
    import h5py
    import stvis
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import ndimage
    import seaborn as sns
    import plotly.graph_objs as go
    import copy
    import json

    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    from bmmltools.core.tracer import Trace
    from bmmltools.board.backend.tracegraph import build_trace_graph, inspect_graph
    from bmmltools.operations.feature import PatchDiscreteFourierTransform3D
    from bmmltools.operations.clustering import Clusterer, ClusterValidator, RotationalSimilarityIdentifier, \
        ArchetypeIdentifier
    from bmmltools.board.backend.visualization import plotly_color_palette


    #################
    #####   FUNCTIONS
    #################


    @st.cache
    def cached_load(trace, attribute):

        return trace.__getattribute__(attribute)

    @st.cache
    def cached_load_key_from_dict(trace,name,key):

        return trace.read_dictionary_key(name,key)

    @st.cache(allow_output_mutation=True,ttl=100)
    def plot_volume(vol,plot_kind = 'volume',ISmin = 0.7,ISmax = 1.0,use_log_scale = True,smooth = True,sigma = 0.5,
                    return_figure=False):

        X,Y,Z= np.mgrid[:vol.shape[2],:vol.shape[1],:vol.shape[0]]
        if use_log_scale:

            vol = np.log(vol)

        if smooth:

            vol = ndimage.gaussian_filter(vol,sigma)

        vol /= vol.max()
        if plot_kind == 'volume':

            to_plot = go.Volume(x=X.flatten(), y=Y.flatten(), z=Z.flatten(),value=vol.flatten(),
                                isomin=ISmin,isomax=ISmax,opacity=0.1,surface_count=25)

        else:

            to_plot = go.Isosurface(x=X.flatten(),y=Y.flatten(),z=Z.flatten(),value=vol.flatten(),
                                    isomin=ISmin,isomax=ISmax,caps=dict(x_show=False, y_show=False))

        fig = go.Figure(data=to_plot)
        fig.update_layout(scene_xaxis_showticklabels=False,
                          scene_yaxis_showticklabels=False,
                          scene_zaxis_showticklabels=False)
        if return_figure:

            return fig

        fig.show()

    @st.cache(allow_output_mutation=True,ttl=100)
    def plot_points3d(df,labels,selected_labels,max_n_ticks = 10,return_figure=False):

        color_palette = sns.color_palette("hls", np.max(labels)+1)
        color_palette = plotly_color_palette(color_palette)
        n_xticks = np.minimum(df.loc[:, 'X'].max() - df.loc[:, 'X'].min(),max_n_ticks)
        n_yticks = np.minimum(df.loc[:, 'Y'].max() - df.loc[:, 'Y'].min(),max_n_ticks)
        n_zticks = np.minimum(df.loc[:, 'Z'].max() - df.loc[:, 'Z'].min(),max_n_ticks)
        layout = go.Layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
                           scene=dict(xaxis=dict(nticks=int(n_xticks),range=[df.loc[:, 'X'].min(), df.loc[:, 'X'].max()]),
                                      yaxis=dict(nticks=int(n_yticks),range=[df.loc[:, 'Y'].min(), df.loc[:, 'Y'].max()]),
                                      zaxis=dict(nticks=int(n_zticks),range=[df.loc[:, 'Z'].min(), df.loc[:, 'Z'].max()])),
                           scene_aspectmode='cube',
                           legend=dict(yanchor="bottom",y=0.01,xanchor="left",x=0.01) )
        plot_figure = go.Figure(layout=layout)
        for sel_label in selected_labels:

            sel_df = df[df['label'] == sel_label]
            X = sel_df.loc[:, 'X'].to_list()
            Y = sel_df.loc[:, 'Y'].to_list()
            Z = sel_df.loc[:, 'Z'].to_list()
            t = go.Scatter3d(x=X, y=Y, z=Z,
                             mode='markers',
                             marker={'size': 7,
                                     'opacity': 1,
                                     'color': 'rgb{}'.format(color_palette[sel_label])},
                             name='Label {}'.format(sel_label))
            plot_figure.add_trace(t)

        if return_figure:

            return plot_figure

        plot_figure.show()


    ############
    #####   MAIN
    ############


    # st.set_page_config(page_title='Labels visualizer', page_icon='🔍')

    st.sidebar.header('Labels visualizer')
    if 'trace_folder_path' in list(st.session_state.keys()):

        # extract trace information
        trace_folder_path = st.session_state['trace_folder_path']
        trace_code = st.session_state['trace_code']
        hdf5_trace_path = trace_folder_path+os.sep+'trace_{}.hdf5'.format(trace_code)
        json_trace_path = trace_folder_path+os.sep+'trace_{}.json'.format(trace_code)

        #
        st.sidebar.write('Current trace code: {}'.format(trace_code))
        with h5py.File(hdf5_trace_path,'r') as h5file:

            available_groups = list(h5file.keys())

        group_name = st.sidebar.selectbox('Select the group containing the result the labelling tool: ',
                                  options=['',]+available_groups)
        with h5py.File(hdf5_trace_path,'r') as h5file:

            _inner_path = group_name
            if _inner_path == '':

                _inner_path = '/'

            available_datasets = list(h5file[_inner_path].keys())
            if 'input_dataset' in available_datasets:

                del available_datasets[available_datasets.index('input_dataset')]
                available_datasets = ['input_dataset',]+available_datasets

        input_dataset_name = st.sidebar.selectbox('Select input dataset',options=available_datasets)
        st.sidebar.write('----------------------')
        inapp_3d_plot = not st.sidebar.checkbox('Show 3d plots on separate window',
                                                help='Reduce computational requirements for visualization of large '
                                                     '3d dataset',
                                                value=True)
        if group_name != '':

            trace = Trace()
            trace.link(trace_folder=trace_folder_path,group_name=group_name)

            st.write('#### Trace graph visualization')
            with st.expander('See trace {} graph for group \'{}\''.format(trace_code,group_name)):

                net = build_trace_graph(trace.trace_graph_dict,max_width=650)
                stvis.pv_static(net)

            st.write('--------------------')
            st.write('#### Dataset visualization')

            opt = [elem.split('.')[-1] for elem in trace.pandas_variables_traced_on_hd+trace.dict_variables_traced_on_hd+
                   trace.numpy_variables_traced_on_hd if elem.count('.') == 1]
            to_visualize = st.selectbox('Select dataset:', options=['',]+opt)
            if to_visualize != '':

                if list(inspect_graph(to_visualize,trace.trace_graph_dict).values())[0] == '':

                    fullname = '{}.{}'.format(trace.group_name,to_visualize)
                    if fullname in trace.pandas_variables_traced_on_hd:

                        st.dataframe(cached_load(trace,to_visualize))

                    if fullname in trace.dict_variables_traced_on_hd:

                        dictionary_res = cached_load(trace,to_visualize)
                        selected_key = st.selectbox('Select data to visualize:',options=['',]+list(dictionary_res.keys()))
                        if selected_key != '':

                            fullname2 = '{}.{}.{}'.format(trace.group_name,to_visualize,selected_key)
                            if fullname2 in trace.pandas_variables_traced_on_hd:

                                st.dataframe(dictionary_res[selected_key])

                            else:

                                st.write('No specific visualization method available.')

                elif PatchDiscreteFourierTransform3D.__name__ in list(inspect_graph(to_visualize,
                                                                      trace.trace_graph_dict).values())[0]:

                    # get plottable keys
                    with h5py.File(trace.hdf5_trace_path, 'r') as file:

                        av_keys = list(file['segmenter/post_pdft3d_inference_dataset'].keys())
                        try:

                            av_keys.remove('patch_space_coordinates')

                        except:

                            pass

                    key_to_vis = st.selectbox('Select data to visualize',options=['',]+av_keys)
                    if key_to_vis != '':

                        # load necessary data
                        coords_df = cached_load_key_from_dict(trace,to_visualize,'patch_space_coordinates')
                        with st.form('visualize 3d dft'):

                            # visualization inputs
                            col1, col2, col3 = st.columns(3)
                            z0 = col1.number_input('Z in patch space', min_value=0, max_value=int(max(coords_df['Z'])),
                                                   value=0)
                            y0 = col1.number_input('Y in patch space', min_value=0, max_value=int(max(coords_df['Y'])),
                                                   value=0)
                            x0 = col1.number_input('X in patch space', min_value=0, max_value=int(max(coords_df['X'])),
                                                   value=0)
                            index0 = col1.number_input('Patch index', min_value=0, max_value=len(coords_df), value=0)
                            read_index = col2.radio('Select patch via', options=['index', 'ZYX coordinates in patch space'])
                            if read_index == 'ZYX coordinates in patch space':

                                index = coords_df[(coords_df['X'] == x0) &
                                                  (coords_df['Y'] == y0) &
                                                  (coords_df['Z'] == z0)].index.to_numpy()
                                if len(index) > 0:

                                    index = int(index)

                                else:

                                    index = 0
                                    st.write('No index found corresponding to the ZYX coordinates: index 0 used instead.')

                            else:

                                index = index0

                            plot_kind = col2.selectbox('Plot kind', options=['volume', 'isosurface'])
                            use_log_scale = col2.checkbox('Use logarithmic scale', value=True)
                            smooth = col2.checkbox('Use gaussian filter', value=True)

                            ISmin = col3.slider('Isomin', min_value=0., max_value=.99, value=0.7)
                            ISmax = col3.slider('Isomax', min_value=ISmin, max_value=1., value=1.)
                            sigma = col3.slider('Sigma gaussian filter', min_value=0.1, max_value=5.0, value=0.5)

                            show = st.form_submit_button('Show')
                            if show:

                                with h5py.File(trace.hdf5_trace_path, 'r') as file:

                                    vol = file[trace._hdf5_inner_path+'{}/{}'.format(to_visualize,key_to_vis)][index,:,:,:]

                                fig = plot_volume(vol, plot_kind, ISmin, ISmax, use_log_scale, smooth, sigma,
                                                  return_figure=inapp_3d_plot)

                        if inapp_3d_plot:

                            st.plotly_chart(fig)

                elif Clusterer.__name__ in list(inspect_graph(to_visualize, trace.trace_graph_dict).values())[0] and \
                     ClusterValidator.__name__ not in list(inspect_graph(to_visualize, trace.trace_graph_dict).values())[0]:

                    result_df = cached_load(trace,to_visualize)
                    with open(trace.json_trace_path,'r') as f:

                        data = json.load(f)

                    try:

                        patch_shape0 = tuple(data[trace.group_name]['ops_parameters']['2']['p']['patch_shape']) # risky...find a better method to get the patch shape

                    except:

                        patch_shape0 = (50,50,50)

                    st.write('**Visualization in patch space**')
                    labels = list(set(result_df.loc[:, 'label'].to_list()))
                    color_palette = sns.color_palette("hls", np.max(labels) + 1)
                    color_palette = plotly_color_palette(color_palette)
                    n_xticks = int(result_df.loc[:,'X'].max()-result_df.loc[:,'X'].min())
                    n_yticks = int(result_df.loc[:,'Y'].max()-result_df.loc[:,'Y'].min())
                    n_zticks = int(result_df.loc[:,'Z'].max()-result_df.loc[:,'Z'].min())
                    layout = go.Layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
                                       scene=dict(xaxis=dict(nticks=n_xticks,
                                                             range=[result_df.loc[:,'X'].min(),result_df.loc[:,'X'].max()]),
                                                  yaxis=dict(nticks=n_yticks,
                                                             range=[result_df.loc[:,'Y'].min(),result_df.loc[:,'Y'].max()]),
                                                  zaxis=dict(nticks=n_zticks,
                                                             range=[result_df.loc[:,'Z'].min(),result_df.loc[:,'Z'].max()]),
                                                  ),
                                       scene_aspectmode='cube',
                                       legend=dict(yanchor="bottom",
                                                   y=0.01,
                                                   xanchor="left",
                                                   x=0.01)
                                       )
                    plot_figure = go.Figure(layout=layout)
                    selected_labels = st.multiselect('Select labels visualized',options=labels,default=labels)
                    for sel_label in selected_labels:

                        sel_result_df = result_df[result_df['label']==sel_label]
                        X = sel_result_df.loc[:,'X'].to_list()
                        Y = sel_result_df.loc[:,'Y'].to_list()
                        Z = sel_result_df.loc[:,'Z'].to_list()
                        t = go.Scatter3d(x=X,y=Y,z=Z,
                                         mode='markers',
                                         marker={'size': 7,
                                                 'opacity': 1,
                                                 'color': 'rgb{}'.format(color_palette[sel_label])},
                                         name='Label {}'.format(sel_label))
                        plot_figure.add_trace(t)

                    st.plotly_chart(plot_figure)

                    st.write('**Visualize raw labels on input data**')
                    fig = plt.figure()
                    with st.form('raw label on input dataset'):

                        col1,col2,col3 = st.columns(3)
                        axis = col1.radio('Select axis',('Z','Y','X'))
                        axis = ['Z','Y','X'].index(axis)
                        input_data_shape = trace.__getattribute__(input_dataset_name).shape
                        sl = col2.number_input('Select slice',min_value=0,value=input_data_shape[axis]//2,
                                               max_value=input_data_shape[axis])
                        flip_ud = col2.checkbox('Flip up-down',value=False)
                        patch_shape = col3.text_input('Patch shape', '{}'.format(patch_shape0))
                        flup_lr = col3.checkbox('Flip left-right',value=False)

                        show = st.form_submit_button('Show')
                        if show:

                            patch_shape = eval(patch_shape)
                            input_slice_slicing_num = []
                            for i in range(3):

                                if axis == i:

                                    input_slice_slicing_num.append(slice(sl,sl+1))

                                else:

                                    input_slice_slicing_num.append(slice(None,None))

                            slice_image = np.squeeze(trace.__getattribute__(input_dataset_name)
                                                     [tuple(input_slice_slicing_num)])
                            patch_space_shape = list(np.array(input_data_shape) // np.array(patch_shape))
                            K = sl // patch_shape[axis]

                            patch_shape_ = copy.copy(patch_shape)
                            patch_space_shape_ = copy.copy(patch_space_shape)
                            patch_space_shape_.remove(patch_space_shape_[axis])
                            patch_space_coords = []
                            for i in range(patch_space_shape_[0]):

                                for j in range(patch_space_shape_[1]):

                                    if axis == 0:

                                        patch_space_coords.append([K,i,j])

                                    elif axis == 1:

                                        patch_space_coords.append([i,K,j])

                                    else:

                                        patch_space_coords.append([i,j,K])

                            patch_shape_ = list(patch_shape_)
                            patch_shape_.remove(patch_shape_[axis])
                            mask = -np.ones(slice_image.shape)
                            for coords in patch_space_coords:

                                z, y, x = coords
                                try:

                                    sel = result_df[(result_df['Z'] == z) & (result_df['Y'] == y) & (result_df['X'] == x)]
                                    l = sel.reset_index().loc[0, 'label']
                                    if axis == 0:

                                        mask[y * patch_shape_[0]:(y + 1) * patch_shape_[0],
                                        x * patch_shape_[1]:(x + 1) * patch_shape_[1]] = l+1

                                    elif axis == 1:

                                        mask[z * patch_shape_[0]:(z + 1) * patch_shape_[0],
                                        x * patch_shape_[1]:(x + 1) * patch_shape_[1]] = l+1

                                    else:

                                        mask[z * patch_shape_[0]:(z + 1) * patch_shape_[0],
                                        y * patch_shape_[1]:(y + 1) * patch_shape_[1]] = l+1

                                except:

                                    continue

                            if flip_ud:

                                slice_image = np.flipud(slice_image)
                                mask = np.flipud(mask)

                            if flup_lr:

                                slice_image = np.fliplr(slice_image)
                                mask = np.fliplr(mask)

                            ax = fig.add_subplot()
                            ax.imshow(slice_image, cmap='Greys_r')
                            legend_elements = []
                            for L in selected_labels:

                                im_ma = np.ma.array(slice_image, mask=~(mask == L+1))
                                newcmp = ListedColormap([[0, 0, 0], list(color_palette[L])])
                                ax.imshow(im_ma, cmap=newcmp, alpha=1)
                                legend_elements.append(Patch(facecolor=color_palette[L], label='Label {}'.format(L)))

                            box = ax.get_position()
                            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                                             box.width, box.height * 0.9])
                            ax.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.05),
                                      loc='upper center', fancybox=True, shadow=True, ncol=5)

                    st.pyplot(fig)

                elif ClusterValidator.__name__ in list(inspect_graph(to_visualize, trace.trace_graph_dict).values())[0]:

                    result_df = cached_load(trace, to_visualize)
                    available_labels = [e for e in result_df.columns if e in ['label','RS_label']]
                    lab_selected = st.selectbox('Kind of label to visualize', options=available_labels)
                    available_columns = [e for e in result_df.columns if e not in ['Z','Y','X','label','RS_label']]
                    col_selected = st.selectbox('Label to visualize: ',options=available_columns)
                    result_df = result_df[result_df[col_selected] == 1]
                    with open(trace.json_trace_path, 'r') as f:

                        data = json.load(f)

                    try:

                        patch_shape0 = tuple(data[trace.group_name]['ops_parameters']['2']['p'][
                                                 'patch_shape'])  # risky...find a better method to get the patch shape

                    except:

                        patch_shape0 = (50, 50, 50)

                    st.write('**Visualization in patch space**')
                    labels = list(set(result_df.loc[:,lab_selected].to_list()))
                    color_palette = sns.color_palette("hls", np.max(labels) + 1)
                    color_palette = plotly_color_palette(color_palette)
                    n_xticks = int(result_df.loc[:, 'X'].max() - result_df.loc[:, 'X'].min())
                    n_yticks = int(result_df.loc[:, 'Y'].max() - result_df.loc[:, 'Y'].min())
                    n_zticks = int(result_df.loc[:, 'Z'].max() - result_df.loc[:, 'Z'].min())
                    layout = go.Layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
                                       scene=dict(xaxis=dict(nticks=n_xticks,
                                                             range=[result_df.loc[:, 'X'].min(),
                                                                    result_df.loc[:, 'X'].max()]),
                                                  yaxis=dict(nticks=n_yticks,
                                                             range=[result_df.loc[:, 'Y'].min(),
                                                                    result_df.loc[:, 'Y'].max()]),
                                                  zaxis=dict(nticks=n_zticks,
                                                             range=[result_df.loc[:, 'Z'].min(),
                                                                    result_df.loc[:, 'Z'].max()]),
                                                  ),
                                       scene_aspectmode='cube',
                                       legend=dict(yanchor="bottom",
                                                   y=0.01,
                                                   xanchor="left",
                                                   x=0.01)
                                       )
                    plot_figure = go.Figure(layout=layout)
                    selected_labels = st.multiselect('Select labels visualized', options=labels, default=labels)
                    for sel_label in selected_labels:

                        sel_result_df = result_df[result_df[lab_selected] == sel_label]
                        X = sel_result_df.loc[:, 'X'].to_list()
                        Y = sel_result_df.loc[:, 'Y'].to_list()
                        Z = sel_result_df.loc[:, 'Z'].to_list()
                        t = go.Scatter3d(x=X, y=Y, z=Z,
                                         mode='markers',
                                         marker={'size': 7,
                                                 'opacity': 1,
                                                 'color': 'rgb{}'.format(color_palette[sel_label])},
                                         name='Label {}'.format(sel_label))
                        plot_figure.add_trace(t)

                    st.plotly_chart(plot_figure)

                    st.write('**Visualize selected labels on input data**')
                    fig = plt.figure()
                    with st.form('valid label on input dataset'):

                        col1, col2, col3 = st.columns(3)
                        axis = col1.radio('Select axis', ('Z', 'Y', 'X'))
                        axis = ['Z', 'Y', 'X'].index(axis)
                        input_data_shape = trace.__getattribute__(input_dataset_name).shape
                        sl = col2.number_input('Select slice',
                                               min_value=0,
                                               value=input_data_shape[axis]//2,
                                               max_value=input_data_shape[axis])
                        flip_ud = col2.checkbox('Flip up-down',value=False)
                        patch_shape = col3.text_input('Patch shape', '{}'.format(patch_shape0))
                        flup_lr = col3.checkbox('Flip left-right',value=False)

                        show = st.form_submit_button('Show')
                        if show:

                            patch_shape = eval(patch_shape)
                            input_slice_slicing_num = []
                            for i in range(3):

                                if axis == i:

                                    input_slice_slicing_num.append(slice(sl,sl+1))

                                else:

                                    input_slice_slicing_num.append(slice(None,None))

                            slice_image = np.squeeze(trace.__getattribute__(input_dataset_name)
                                                     [tuple(input_slice_slicing_num)])
                            patch_space_shape = list(np.array(input_data_shape) // np.array(patch_shape))
                            K = sl // patch_shape[axis]

                            patch_shape_ = copy.copy(patch_shape)
                            patch_space_shape_ = copy.copy(patch_space_shape)
                            patch_space_shape_.remove(patch_space_shape_[axis])
                            patch_space_coords = []
                            for i in range(patch_space_shape_[0]):

                                for j in range(patch_space_shape_[1]):

                                    if axis == 0:

                                        patch_space_coords.append([K,i,j])

                                    elif axis == 1:

                                        patch_space_coords.append([i,K,j])

                                    else:

                                        patch_space_coords.append([i,j,K])

                            patch_shape_ = list(patch_shape_)
                            patch_shape_.remove(patch_shape_[axis])
                            mask = -np.ones(slice_image.shape)
                            for coords in patch_space_coords:

                                z, y, x = coords
                                try:

                                    sel = result_df[(result_df['Z'] == z) & (result_df['Y'] == y) & (result_df['X'] == x)]
                                    l = sel.reset_index().loc[0, 'label']
                                    if axis == 0:

                                        mask[y * patch_shape_[0]:(y + 1) * patch_shape_[0],
                                        x * patch_shape_[1]:(x + 1) * patch_shape_[1]] = l+1

                                    elif axis == 1:

                                        mask[z * patch_shape_[0]:(z + 1) * patch_shape_[0],
                                        x * patch_shape_[1]:(x + 1) * patch_shape_[1]] = l+1

                                    else:

                                        mask[z * patch_shape_[0]:(z + 1) * patch_shape_[0],
                                        y * patch_shape_[1]:(y + 1) * patch_shape_[1]] = l+1

                                except:

                                    continue

                            if flip_ud:

                                slice_image = np.flipud(slice_image)
                                mask = np.flipud(mask)

                            if flup_lr:

                                slice_image = np.fliplr(slice_image)
                                mask = np.fliplr(mask)

                            fig = plt.figure()
                            ax = fig.add_subplot()
                            ax.imshow(slice_image, cmap='Greys_r')
                            legend_elements = []
                            for L in selected_labels:

                                im_ma = np.ma.array(slice_image, mask=~(mask == L+1))
                                newcmp = ListedColormap([[0, 0, 0], list(color_palette[L])])
                                ax.imshow(im_ma, cmap=newcmp, alpha=1)
                                legend_elements.append(Patch(facecolor=color_palette[L], label='Label {}'.format(L)))

                            box = ax.get_position()
                            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                                             box.width, box.height * 0.9])
                            ax.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.05),
                                      loc='upper center', fancybox=True, shadow=True, ncol=5)

                    st.pyplot(fig)

                elif ArchetypeIdentifier.__name__ in list(inspect_graph(to_visualize, trace.trace_graph_dict).values())[0]:

                    st.write('**Visualize archetype coordinates**')
                    archetype_coords_df = cached_load_key_from_dict(trace,to_visualize,'archetype_patch_coordinates')
                    labels = list(set(archetype_coords_df.loc[:, 'label'].to_list()))
                    selected_labels = st.multiselect('Select labels visualized', options=labels, default=labels)

                    fig = plot_points3d(archetype_coords_df,labels,selected_labels,return_figure=True)
                    st.plotly_chart(fig)

                elif RotationalSimilarityIdentifier.__name__ in \
                       list(inspect_graph(to_visualize, trace.trace_graph_dict).values())[0]:

                    st.write('**Visualize cluster rotational similarity information**')
                    rs_dict = cached_load(trace,to_visualize)

                    radial_dist_keys = []
                    mean_archetypes_keys = []
                    rs_results = []
                    labels = []
                    for key in rs_dict.keys():

                        if 'label' in key:

                            labels.append(int(key.split('label_')[1].split('_')[0]))
                            if 'radial' in key:

                                radial_dist_keys.append(key)

                            else:

                                mean_archetypes_keys.append(key)

                        else:

                            rs_results.append(key)

                    labels = list(set(labels))

                    col0,col1 = st.columns(2)
                    col0.write('Suggested identification: ')
                    eq_mask = rs_dict['identification_df'].to_numpy()
                    if np.any(eq_mask != np.eye(len(eq_mask))):

                        eq_relation = RotationalSimilarityIdentifier._get_binary_relation_from_mask(eq_mask)
                        eq_relation = RotationalSimilarityIdentifier._split_in_subgroups(eq_relation)
                        for g in eq_relation:

                            str_res = ('- **Label {}** with '.format(g[0]))
                            for l in g[1:-1]:

                                str_res += '**label {}**, '.format(l)

                            if len(g)>2:

                                str_res = str_res[:-2]+' and '

                            str_res += '**label {}**.'.format(g[-1])
                            col0.write(str_res)

                        col0.write(' ')

                    else:

                        col0.write('None')

                    col0.write('Identification based on simmetrized [1 sample Hotelling t² test]'
                               '(https://en.wikipedia.org/wiki/Hotelling%27s_T-squared_distribution).')
                    show_stp = col0.checkbox('Show statistical test power',value=False)
                    show_rd = col0.checkbox('Show radial distribution',value=False)
                    show_ma = col0.checkbox('Show archetype mean 3d dft modulus',value=False)
                    if not show_stp:

                        fig,ax = plt.subplots()
                        ax.set_title('Identification probability')
                        img = ax.imshow(rs_dict['identification_probability'],cmap='Reds_r')
                        ax.set_yticks(range(len(labels)))
                        ax.set_yticklabels(['Label {}'.format(l) for l in labels])
                        ax.set_xticks(range(len(labels)))
                        ax.set_xticklabels(['Label {}'.format(l) for l in labels],rotation=45,horizontalalignment='right')
                        plt.colorbar(img)

                    else:

                        fig, ax = plt.subplots()
                        ax.set_title('Hotelling t² test power')
                        img = ax.imshow(rs_dict['test_statistical_power'], cmap='Greens')
                        ax.set_yticks(range(len(labels)))
                        ax.set_yticklabels(['Label {}'.format(l) for l in labels])
                        ax.set_xticks(range(len(labels)))
                        ax.set_xticklabels(['Label {}'.format(l) for l in labels], rotation=45, horizontalalignment='right')
                        plt.colorbar(img)

                    col1.pyplot(fig)

                    if show_rd:

                        st.write('**3d dft mean radial distribution**')
                        color_palette = sns.color_palette("hls", np.max(labels) + 1)
                        max_bin_num = len(rs_dict[radial_dist_keys[0]])
                        bins_selected = st.select_slider('Select radial distribution bin range to visualize',
                                                         options = list(range(max_bin_num)),
                                                         value = (1,max_bin_num-1))
                        fig = plt.figure()
                        for k in radial_dist_keys:

                            name = k.replace('_',' ').replace('archetype ','').replace('mean radial distribution','')
                            x = [True if 'label_{}'.format(l) in k else False for l in labels]
                            plt.plot(rs_dict[k][bins_selected[0]:bins_selected[1]],label = name,c = color_palette[x.index(True)])

                        plt.legend()
                        st.pyplot(fig)

                    if show_ma:

                        st.write('**Archetype mean 3d dft modulus**')
                        label_sel = st.selectbox('Select archetype label',options=labels)
                        for k in  mean_archetypes_keys:

                            if 'label_{}'.format(label_sel) in k:

                                mdft_arch = rs_dict[k]
                                with st.form('Archetype visualizer 3d dft module'):

                                    col0,col1 = st.columns(2)

                                    plot_kind = col0.selectbox('Plot kind', options=['volume', 'isosurface'])
                                    use_log_scale = col0.checkbox('Use logarithmic scale', value=True)
                                    smooth = col0.checkbox('Use gaussian filter', value=False)

                                    ISmin = col1.slider('Isomin', min_value=0., max_value=.99, value=0.5)
                                    ISmax = col1.slider('Isomax', min_value=ISmin, max_value=1., value=1.)
                                    sigma = col1.slider('Sigma gaussian filter', min_value=0.1, max_value=5.0, value=0.5)

                                    send = st.form_submit_button('Send')
                                    if send:

                                        fig = plot_volume(mdft_arch,plot_kind,ISmin,ISmax,use_log_scale,smooth,sigma,
                                                          return_figure=inapp_3d_plot)

                                if inapp_3d_plot:

                                    st.plotly_chart(fig)


                else:

                    st.write('No visualization method currently available')