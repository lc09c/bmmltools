# Title: 'graph.py'
# Author: Curcuraci L.
# Date: 13/10/2022
#
# Scope: collect methods used to inspect the operation graphs in bmmltools.

"""
Graph related utils.
"""

#################
#####   LIBRARIES
#################


###############
#####   CLASSES
###############


class TraceGraphInspector:
    """
    Inspect a given operation graph.
    """

    def __init__(self,trace_graph):
        """
        Class used to get some information from the operation graph of a given trace.

        :param trace_graph: (dict) dictionary containing the operation graph of a trace.
        """
        self.trace_graph = trace_graph
        self._path = []

    def find_output(self,output_name):
        """
        Given the output name, find the position in the operation graph of the node having it as output.

        :param output_name: (str) name of the output to search.
        """
        for key,line in self.trace_graph.items():

            if output_name in line['node']:

                return key

        return None

    def find_operation(self, operation_name):
        """
        Given an operation name find the (first) position in the operation graph with that operation.

        :param operation_name: (str) name of the operation to search.
        """
        for key, line in self.trace_graph.items():

            if operation_name in line['edge']:

                return key

        return None

    def reconstruct_simple_path(self,output_name,key_position = True):
        """
        Inspect the operation graph in order to find all the operation needed to produce a given output from the input
        dataset.

        :param output_name: (str)
        :param key_position: (bool) optional, if False the list of operation names is returned, otherwise just the
                              list of positions in the operation graph.
        """
        out_key = self.find_output(output_name)
        if out_key is not None:

            if len(self.trace_graph[out_key]['link']) != 0:

                previous_output_name = self.trace_graph[out_key]['link'][0][0]
                if key_position:

                    self._path.append(list(self.trace_graph.keys()).index(out_key))

                else:

                    self._path.append(output_name)

                return self.reconstruct_simple_path(previous_output_name,key_position)

            else:

                if key_position:

                    return self._path+[0,]

                else:

                    return self._path+[self.trace_graph[list(self.trace_graph.keys())[0]]['node'][0],]

        else:

            return self._path

