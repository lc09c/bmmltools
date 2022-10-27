# Title: 'tracegraph.py'
# Author: Curcuraci L.
# Date: 18/10/2022
#
# Scope: functions used to visualize the trace graph in bmmlboard

"""
Graph utils for bmmlboard.
"""

#################
#####   LIBRARIES
#################


import numpy as np
from pyvis.network import Network


#################
#####   FUNCTIONS
#################


def build_trace_graph(graph_dict,graph_name='',x0=0,y0=0,delta_x=100,delta_y=150,max_width=500,max_height= 1000):
    """
    Visualize the trace graph.


    :param graph_dict:
    :param graph_name:
    :param x0:
    :param y0:
    :param delta_x:
    :param delta_y:
    :param max_width:
    :param max_height:
    :return:
    """
    n_nodes = len(graph_dict)
    net = Network(height='{}px'.format(np.minimum((n_nodes + 1) * delta_y, max_height)),
                  width='{}px'.format(max_width),
                  heading=graph_name,
                  directed=True)
    n = 0
    w0 = x0
    idx_node_name_correspondence = {}
    for k in graph_dict:

        graph_line = graph_dict[k]
        for n_node, node in enumerate(graph_line['node']):

            if node == 'input_dataset':

                net.add_node(n,
                             label=node,
                             shape='square',
                             physics=False,
                             x=x0,
                             y=y0)
                idx_node_name_correspondence.update({node: n})

            else:

                if len(graph_line['link']) >= 2 and n_node != 0:

                    w = w0 - int(np.sign(w0)) * n_node * delta_x // (len(graph_line['link']) - 1)

                else:

                    w = x0 + int(np.power(-1, n + 1) * delta_x)

                net.add_node(n,
                             label=node,
                             shape='dot',
                             physics=False,
                             x=w,
                             y=y0 + n * delta_y)
                idx_node_name_correspondence.update({node: n})
                w0 = w

            n += n_node + 1

        for l in graph_line['link']:

            idx_node1 = idx_node_name_correspondence[l[0]]
            idx_node2 = idx_node_name_correspondence[l[1]]
            net.add_edge(source=idx_node1,
                         to=idx_node2,
                         label=graph_line['edge']+'_{}'.format(k),
                         smooth=True)

    return net
#
# def inspect_graph(x, graph_dict,in_type='output',out_time):
#     """
#     Inspect trace graph in order to find the operation producing a given output of the output produced by a
#     given operation.
#
#     :param x: (str) target output/ target operation.
#     :param graph_dict: (dict) graph dictionary.
#     :param type: (str) optional, if 'output' it looks for the operation producing the target output, while if
#                  'operation' it looks for the output produced by the target operation.
#     :return: (str) operation name/output name.
#     """
#     if type == 'output':
#
#         for k in graph_dict:
#
#             if x in graph_dict[k]['node']:
#
#                 return graph_dict[k]['edge']
#
#         return ''
#
#     elif type == 'operation':
#
#         for k in graph_dict:
#
#             if x == graph_dict[k]['edge']:
#
#                 return graph_dict[k]['node']
#
#         return []
#
#     elif type == 'key':
#
#         for k in graph_dict:
#
#             if x == graph_dict[k]['edge']:
#
#                 return k
#
#         return None
#
#     elif type == 'link':
#
#         for k in graph_dict:
#
#             if x == graph_dict[k]['edge']:
#
#                 return
#

def inspect_graph(x, graph_dict, in_type='node', out_type='edge'):
    """
    Inspect trace graph in order to find the operation producing a given output of the output produced by a
    given operation.

    :param x: (str) target output/ target operation.
    :param graph_dict: (dict) graph dictionary.
    :param type: (str) optional, if 'output' it looks for the operation producing the target output, while if
                 'operation' it looks for the output produced by the target operation.
    :return: (str) operation name/output name.
    """
    sel_keys = []
    for k in graph_dict:

        if x in graph_dict[k][in_type]:

            sel_keys.append(k)

    if len(sel_keys)>0:

        if out_type == 'key':

            return sel_keys

        result = {}
        for s_k in sel_keys:

            if out_type in ['node','edge']:

                result.update({s_k: graph_dict[s_k][out_type]})

            if out_type == 'link':

                tmp = []
                for l in graph_dict[s_k][out_type]:

                    tmp.append(l[0])

                result.update({s_k: tmp})

        return result

    else:

        return None