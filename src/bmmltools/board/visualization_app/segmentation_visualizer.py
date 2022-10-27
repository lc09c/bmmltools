# Title: 'segmentation_visualizer.py'
# Author: Curcuraci L.
# Date: 19/10/2022
#
# Scope: visualization app for the  refined segmentation.

"""
Visualization app for RandomForestSegmenter operation.
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
    from bmmltools.operations.segmentation import RandomForestSegmenter
    from bmmltools.board.backend.visualization import plotly_color_palette


    #################
    #####   FUNCTIONS
    #################


    @st.cache
    def cached_load(trace, attribute):

        return trace.__getattribute__(attribute)

    @st.cache
    def cached_load_key_from_dict(trace, name, key):

        return trace.read_dictionary_key(name, key)

    @st.cache(allow_output_mutation=True, ttl=100)
    def plot_volume(vol, plot_kind='volume', ISmin=0.7, ISmax=1.0, use_log_scale=True, smooth=True, sigma=0.5,
                    return_figure=False):

        X, Y, Z = np.mgrid[:vol.shape[2], :vol.shape[1], :vol.shape[0]]
        if use_log_scale:
            vol = np.log(vol)

        if smooth:
            vol = ndimage.gaussian_filter(vol, sigma)

        vol /= vol.max()
        if plot_kind == 'volume':

            to_plot = go.Volume(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=vol.flatten(),
                                isomin=ISmin, isomax=ISmax, opacity=0.1, surface_count=25)

        else:

            to_plot = go.Isosurface(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=vol.flatten(),
                                    isomin=ISmin, isomax=ISmax, caps=dict(x_show=False, y_show=False))

        fig = go.Figure(data=to_plot)
        fig.update_layout(scene_xaxis_showticklabels=False,
                          scene_yaxis_showticklabels=False,
                          scene_zaxis_showticklabels=False)
        if return_figure:
            return fig

        fig.show()

    @st.cache(allow_output_mutation=True, ttl=100)
    def plot_points3d(df, labels, selected_labels, max_n_ticks=10, return_figure=False):

        color_palette = sns.color_palette("hls", np.max(labels) + 1)
        color_palette = plotly_color_palette(color_palette)
        n_xticks = np.minimum(df.loc[:, 'X'].max() - df.loc[:, 'X'].min(), max_n_ticks)
        n_yticks = np.minimum(df.loc[:, 'Y'].max() - df.loc[:, 'Y'].min(), max_n_ticks)
        n_zticks = np.minimum(df.loc[:, 'Z'].max() - df.loc[:, 'Z'].min(), max_n_ticks)
        layout = go.Layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
                           scene=dict(
                               xaxis=dict(nticks=int(n_xticks), range=[df.loc[:, 'X'].min(), df.loc[:, 'X'].max()]),
                               yaxis=dict(nticks=int(n_yticks), range=[df.loc[:, 'Y'].min(), df.loc[:, 'Y'].max()]),
                               zaxis=dict(nticks=int(n_zticks), range=[df.loc[:, 'Z'].min(), df.loc[:, 'Z'].max()])),
                           scene_aspectmode='cube',
                           legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01))
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


    # st.set_page_config(page_title='Segmentation visualizer', page_icon='🧫')

    st.sidebar.header('Segmentation visualizer')
    if 'trace_folder_path' in list(st.session_state.keys()):

        # extract trace information
        trace_folder_path = st.session_state['trace_folder_path']
        trace_code = st.session_state['trace_code']
        hdf5_trace_path = trace_folder_path + os.sep + 'trace_{}.hdf5'.format(trace_code)
        json_trace_path = trace_folder_path + os.sep + 'trace_{}.json'.format(trace_code)

        #
        st.sidebar.write('Current trace code: {}'.format(trace_code))
        with h5py.File(hdf5_trace_path, 'r') as h5file:

            available_groups = list(h5file.keys())

        group_name = st.sidebar.selectbox('Select the group containing the result the labelling tool: ',
                                          options=['', ] + available_groups)
        with h5py.File(hdf5_trace_path, 'r') as h5file:

            _inner_path = group_name
            if _inner_path == '':
                _inner_path = '/'

            available_datasets = list(h5file[_inner_path].keys())
            if 'input_dataset' in available_datasets:
                del available_datasets[available_datasets.index('input_dataset')]
                available_datasets = ['input_dataset', ] + available_datasets

        input_dataset_name = st.sidebar.selectbox('Select input dataset', options=available_datasets)
        if group_name != '':

            trace = Trace()
            trace.link(trace_folder=trace_folder_path, group_name=group_name)

            st.write('#### Trace graph visualization')
            with st.expander('See trace {} graph for group \'{}\''.format(trace_code, group_name)):

                net = build_trace_graph(trace.trace_graph_dict, max_width=650)
                stvis.pv_static(net)

            st.write('--------------------')
            st.write('#### Dataset visualization')
            with open(trace.json_trace_path, 'r') as f:

                data = json.load(f)

            try:

                patch_shape0 = tuple(data[trace.group_name]['ops_parameters']['2']['p'][
                                         'patch_shape'])  # risky...find a better method to get the patch shape

            except:

                patch_shape0 = (50, 50, 50)

            opt = [elem.split('.')[-1] for elem in
                   trace.pandas_variables_traced_on_hd + trace.dict_variables_traced_on_hd +
                   trace.numpy_variables_traced_on_hd if elem.count('.') == 1]
            to_visualize = st.selectbox('Select dataset:', options=['', ] + opt)
            if to_visualize != '':

                if list(inspect_graph(to_visualize, trace.trace_graph_dict).values())[
                    0] in RandomForestSegmenter.__name__:

                    st.write('**Visualize segmentation masks on input data**')

                    label_df_name = st.selectbox('Select labelling dataset', options=['', ] + opt)
                    if label_df_name != '':

                        label_df = cached_load(trace, label_df_name)
                        available_labels = [e for e in label_df.columns if e in ['label', 'RS_label']]
                        label_kind = st.selectbox('Kind of label to visualize', options=available_labels)
                        labels = list(set(label_df.loc[:, 'label'].to_list()))
                        color_palette = sns.color_palette("hls", np.max(labels) + 1)

                        # segmentation_result= cached_load(trace, to_visualize)
                        input_data_shape = trace.__getattribute__(input_dataset_name).shape
                        selected_labels = st.multiselect('Select labels visualized', options=labels, default=labels)

                        fig = plt.figure()
                        with st.form('raw label on input dataset'):

                            col1, col2 = st.columns(2)
                            axis = col1.radio('Select axis', ('Z', 'Y', 'X'))
                            axis = ['Z', 'Y', 'X'].index(axis)
                            sl = col2.number_input('Select slice', min_value=0, value=input_data_shape[axis] // 2,
                                                   max_value=input_data_shape[axis])
                            flip_ud = col2.checkbox('Flip up-down', value=False)
                            flup_lr = col2.checkbox('Flip left-right', value=False)
                            show = st.form_submit_button('Show')
                            if show:

                                input_slice_slicing_num = []
                                for i in range(3):

                                    if axis == i:

                                        input_slice_slicing_num.append(slice(sl, sl + 1))

                                    else:

                                        input_slice_slicing_num.append(slice(None, None))

                                slice_image = np.squeeze(trace.__getattribute__(input_dataset_name)
                                                         [tuple(input_slice_slicing_num)])
                                mask = np.squeeze(trace.__getattribute__(to_visualize)
                                                  [tuple(input_slice_slicing_num)])
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
                                    im_ma = np.ma.array(slice_image, mask=~(mask == L))
                                    newcmp = ListedColormap([[0, 0, 0], list(color_palette[L])])
                                    ax.imshow(im_ma, cmap=newcmp, alpha=1)
                                    legend_elements.append(
                                        Patch(facecolor=color_palette[L], label='Label {}'.format(L)))

                                box = ax.get_position()
                                ax.set_position([box.x0, box.y0 + box.height * 0.1,
                                                 box.width, box.height * 0.9])
                                ax.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.05),
                                          loc='upper center', fancybox=True, shadow=True, ncol=5)

                        st.pyplot(fig)

                else:

                    st.write('No visualization method currently available')