#
#
#
#
#

"""
Bmmltools visualization tool main page.
"""


#################
#####   LIBRARIES
#################


import sys
sys.path.append(r'\\HOME\udata\curcuraci\Desktop\package\src')

import streamlit as st
import os
from bmmltools.board.visualization_app import label_visualizer,segmentation_visualizer,explainer_visualizer


############
#####   MAIN
############


st.set_page_config(layout = "wide")
with st.sidebar:

    st.markdown('**Select the visualization app**')
    page = st.selectbox('Select:',['','Label visualizer','Segmentation visualizer','Explainer visualizer'])

if page == 'Label visualizer':

    label_visualizer.run()

elif page == 'Segmentation visualizer':

    segmentation_visualizer.run()

elif page == 'Explainer visualizer':

    explainer_visualizer.run()

else:

    st.title('bmmltools visualization board')
    st.write('This is the bmmltools visualization board. It can be used to inspect the partial results stored in a given '
             'trace.')
    st.write('-------------------')
    trace_folder_path = st.text_input(label = "Please insert the absolute path to a trace folder an press enter" )
    if trace_folder_path != '':

        last_folder_names = os.path.basename(trace_folder_path).split('_')
        if last_folder_names[0] == 'trace':

            st.session_state['trace_folder_path'] = os.path.normpath(trace_folder_path)
            st.session_state['trace_code'] = last_folder_names[1]

        else:

            try:

                del st.session_state['trace_folder_path']
                del st.session_state['trace_code']

            except:

                pass

    if hasattr(st.session_state,'trace_code'):

        st.write('Trace {} linked to the board.'.format(st.session_state['trace_code']))

    else:

        st.write('Currently no trace linked to the board.')

    st.write('-------------------')



