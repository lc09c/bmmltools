# Title: 'explainer_visualizer.py'
# Author: Curcuraci L.
# Date: 19/10/2022
#
# Scope: visualization app for explanation.

"""
Visualization app for MultiCollinearityReducer, InterpretPIandPD, and ExplainWithClassifier operations.
"""

def run():


    #################
    #####   LIBRARIES
    #################


    import os
    import streamlit as st
    import h5py
    import stvis
    import matplotlib.pyplot as plt

    from bmmltools.core.tracer import Trace
    from bmmltools.board.backend.tracegraph import build_trace_graph, inspect_graph
    from bmmltools.operations.explanation import MultiCollinearityReducer, InterpretPIandPD, ExplainWithClassifier


    #################
    #####   FUNCTIONS
    #################

    @st.cache
    def cached_load(trace, attribute):

        return trace.__getattribute__(attribute)


    ############
    #####   MAIN
    ############


    # st.set_page_config(page_title='Explainer visualizer', page_icon='📊')

    st.sidebar.header('Explainer visualizer')
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

        group_name = st.sidebar.selectbox('Select the group containing the result the explainer tool: ',
                                          options=['', ] + available_groups)
        if group_name != '':

            trace = Trace()
            trace.link(trace_folder=trace_folder_path, group_name=group_name)

            st.write('### Trace {} graph'.format(trace_code))
            net = build_trace_graph(trace.trace_graph_dict)
            stvis.pv_static(net)
            st.write('--------------------')

            opt = [elem.split('.')[-1] for elem in
                   trace.pandas_variables_traced_on_hd + trace.dict_variables_traced_on_hd
                   if elem.count('.') == 1]
            to_visualize = st.selectbox('Select dataset:', options=['', ] + opt)
            if to_visualize != '':

                if list(inspect_graph(to_visualize, trace.trace_graph_dict).values())[0] is '':

                    fullname = '{}.{}'.format(trace.group_name, to_visualize)
                    if fullname in trace.pandas_variables_traced_on_hd:
                        st.dataframe(cached_load(trace, to_visualize))

                    if fullname in trace.dict_variables_traced_on_hd:

                        dictionary_res = cached_load(trace, to_visualize)
                        selected_key = st.selectbox('Select data to visualize:',
                                                    options=['', ] + list(dictionary_res.keys()))
                        if selected_key != '':

                            fullname2 = '{}.{}.{}'.format(trace.group_name, to_visualize, selected_key)
                            if fullname2 in trace.pandas_variables_traced_on_hd:

                                st.dataframe(dictionary_res[selected_key])

                            else:

                                st.write('No specific visualization method available.')

                if list(inspect_graph(to_visualize, trace.trace_graph_dict).values())[
                    0] == MultiCollinearityReducer.__name__:
                    st.dataframe(cached_load(trace, to_visualize))

                if list(inspect_graph(to_visualize, trace.trace_graph_dict).values())[
                    0] == ExplainWithClassifier.__name__:

                    ewc_dict_res = cached_load(trace, to_visualize)

                    st.write('Classifiers F1 scores')
                    st.dataframe(ewc_dict_res['classifier_f1_scores'])

                    labels = [elem.split('_')[-1] for elem in ewc_dict_res if 'PI_Label_' in elem]
                    sel_label_to_vis = st.selectbox('Chose a label: ', options=['', ] + labels)
                    if sel_label_to_vis != '':

                        features = ewc_dict_res['PI_Label_{}'.format(sel_label_to_vis)].index.tolist()
                        pi_m = ewc_dict_res['PI_Label_{}'.format(sel_label_to_vis)].loc[:,
                               'estimated_permutation_importance_mean'].tolist()
                        pi_std = ewc_dict_res['PI_Label_{}'.format(sel_label_to_vis)].loc[:,
                                 'estimated_permutation_importance_std'].tolist()

                        fig, ax = plt.subplots()
                        ax.set_title('Permutation importance'
                                     '\nLabel: {}'.format(sel_label_to_vis))
                        ax.bar(list(range(len(features))), pi_m, yerr=pi_std, align='center', alpha=0.5, ecolor='black',
                               capsize=10)
                        ax.set_ylabel('permutation importance')
                        ax.set_xlabel('features')
                        plt.xticks(list(range(len(features))), labels=features, rotation=45)
                        ax.yaxis.grid(True)
                        st.pyplot(fig)

                        st.write('Visualize partial dependency plots.')
                        sel_features = st.multiselect('Select features:', options=features)
                        if sel_features != []:

                            for f in sel_features:
                                PD_Lf_df = ewc_dict_res['PD_Label_{}_feature_{}'.format(sel_label_to_vis, f)]
                                f_vals = PD_Lf_df.loc[:, 'feature_values']
                                pd_f_mean = PD_Lf_df.loc[:, 'estimated_partial_dependency_mean']
                                pd_f_std = PD_Lf_df.loc[:, 'estimated_partial_dependency_std']

                                fig, ax = plt.subplots()
                                ax.set_title('Partial dependence feature {}  '
                                             '\nLabel: {}'.format(f, sel_label_to_vis))
                                ax.plot(f_vals, pd_f_mean)
                                ax.fill_between(f_vals, pd_f_mean - pd_f_std, pd_f_mean + pd_f_std, color='b',
                                                alpha=0.1)
                                ax.set_ylabel('partial dependence')
                                ax.set_xlabel('{} value'.format(f))
                                ax.yaxis.grid(True)
                                st.pyplot(fig)

                if list(inspect_graph(to_visualize, trace.trace_graph_dict).values())[0] == InterpretPIandPD.__name__:

                    li_dict_res = cached_load(trace, to_visualize)

                    int_acc_df = li_dict_res['interpretation_accuracy']
                    labels = [e.split('_')[-1] for e in int_acc_df.index.tolist()]
                    col1, col2 = st.columns(2)

                    for L in labels:

                        col1.write('Label {} is definable as:'.format(L))
                        sel_f = [k.split('_')[-2] for k in li_dict_res if 'Label_{}_feature_'.format(L) in k]
                        l_int = [li_dict_res[e] for e in li_dict_res if 'Label_{}'.format(L) in e]
                        for i, _ in enumerate(l_int):

                            f = sel_f[i]
                            inter = l_int[i]
                            ths = inter.loc[:, 'thresholds'].tolist()
                            pths_vals = inter.loc[:, 'post_thresholds_value'].tolist()
                            if pths_vals == [1, 0, 0]:
                                col1.write('{} ≤ {}'.format(f, ths[1]))
                            elif pths_vals == [0, 1, 0]:
                                col1.write('{} ≥ {}'.format(f, ths[1]))
                            elif pths_vals == [0, 1, 0, 0]:
                                col1.write('{} ≤ {} ≤ {}'.format(ths[1], f, ths[2]))
                            elif pths_vals == [1, 0, 1, 0]:
                                col1.write('{} ≤ {} and {} ≥ {}'.format(f, ths[1], f, ths[2]))

                        col1.write('---------------')

                    #
                    sel_L = st.selectbox('Select label for partial dependency inspection', options=labels)
                    sel_f = [k.split('_')[-2] for k in li_dict_res if 'Label_{}_feature_'.format(sel_L) in k]
                    ewc_label_name = list(inspect_graph(InterpretPIandPD.__name__, trace.trace_graph_dict,
                                                        in_type='edge', out_type='link').values())[0][0]
                    ewc_dict_res = cached_load(trace, ewc_label_name)

                    col2.write('Interpretation reliability')
                    col2.dataframe(int_acc_df)
                    col2.write('-----------------')
                    col2.write('Classifier score')
                    col2.dataframe(ewc_dict_res['classifier_f1_scores'])

                    for i, f in enumerate(sel_f):

                        PD_Lf_df = ewc_dict_res['PD_Label_{}_feature_{}'.format(sel_L, f)]
                        f_vals = PD_Lf_df.loc[:, 'feature_values']
                        pd_f_mean = PD_Lf_df.loc[:, 'estimated_partial_dependency_mean']
                        pd_f_std = PD_Lf_df.loc[:, 'estimated_partial_dependency_std']

                        # thresholds to plot
                        ths_to_plot = li_dict_res['Label_{}_feature_{}_interval'.format(sel_L, f)].loc[:,
                                      'thresholds'].tolist()[1:-1]
                        kind_ths_to_plot = li_dict_res['Label_{}_feature_{}_interval'.format(sel_L, f)].loc[:,
                                           'post_thresholds_value'].tolist()

                        fig, ax = plt.subplots()
                        ax.set_title('Partial dependence feature {}  '
                                     '\nLabel: {}'.format(f, sel_L))
                        ax.plot(f_vals, pd_f_mean)
                        ax.fill_between(f_vals, pd_f_mean - pd_f_std, pd_f_mean + pd_f_std, color='b', alpha=0.1)
                        for th in ths_to_plot:
                            ax.axvline(th, linestyle='--', color='r')

                        if kind_ths_to_plot == [1, 0, 0]:

                            ax.axvspan(f_vals.min(), ths_to_plot[0], alpha=0.1, color='red')

                        elif kind_ths_to_plot == [0, 1, 0]:

                            ax.axvspan(ths_to_plot[0], f_vals.max(), alpha=0.1, color='red')

                        elif kind_ths_to_plot == [0, 1, 0, 0]:

                            ax.axvspan(ths_to_plot[0], ths_to_plot[1], alpha=0.1, color='red')

                        elif kind_ths_to_plot == [1, 0, 1, 0]:

                            ax.axvspan(f_vals.min(), ths_to_plot[0], alpha=0.1, color='red')
                            ax.axvspan(ths_to_plot[1], f_vals.max(), alpha=0.1, color='red')

                        ax.set_ylabel('partial dependence')
                        ax.set_xlabel('{} value'.format(f))
                        ax.yaxis.grid(True)
                        st.pyplot(fig)

            st.write('----------------')

    else:

        st.write('Specify a trace in the main page.')

