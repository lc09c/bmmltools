# title: 'basic.py'
# author: Curcuraci L.
# date: 02/08/2022
#
# Scope: general utilities of bmmltools

"""
Generic utilities used in bmmltools.
"""


#################
#####   LIBRARIES
#################


import os
import itertools
import copy


#################
#####   FUNCTIONS
#################


def standard_number(number,n_digits=4):
    """
    Standardize the string containing numbers to avoid subsequent ordering problem

    :param number: (integer) number to standardize.
    :param n_digits: (integer) maximum number of digits used.
    :return: a standardized string containing the number.
    """
    n_zeros_in_front = n_digits - len(str(number))
    res = str(number)
    for _ in range(n_zeros_in_front):

        res = '0' + res

    return res

def manage_path(path):
    """
    Check if all the folders specified in a given path exist and if not it creates them. This is done in OS
    independent way.

    P.A.: the string in the path field must be RAW! You can easily do that just adding "r" in
    front of the string of the path, e.g.

             [naive string of the path]       ->       [raw string of the path]

        'home/folder1/folder2/file.extension' -> r'home/folder1/folder2/file.extension'

    If this is not done, escape characters may alter the path of the function.

    :param path: RAW string containing the path possibly with file (put an "r" in front of the string!)
    :return: the normalized path
    """
    path = os.path.normpath(path)
    path_to_check = path
    if path.split(os.sep)[-1].find('.')>-1:

        path_to_check = path[:path.find(os.sep+path.split(os.sep)[-1])]

    os.makedirs(path_to_check,exist_ok=True)
    return path

def get_parameters(cls,method = '__init__'):
    """
    Given a class, get the input parameters of a method with its current value.

    P.A.: The convention "method inputs names equal the class attributes names" for all the input variables is assumed.

    :param cls: (object) Python class to analyze.
    :param method: (str) optional, name of the method to analyze.
    :return: (dict) dictionary have the variables names as keys and the variables values as values.
    """
    input_variables = cls.__getattribute__(method).__code__.co_varnames
    params = {}
    for variable in input_variables:

        if variable != 'self':

            try:

                params.update({variable: cls.__getattribute__(variable)})

            except:

                params.update({variable: 'Not Available'})

    return params

def generate_parameter_space(params_dict):
    """
    Given the possible parameters, generate all the possible parameter combinations.

    :param params_dict: (dict) dictionary containing the name of the parameter as key and a list of possible parameter's
                        value as corresponding value.
    :return: (list[list],list) list with all the possible parameter combinations and the list of the corresponding
             parameter names.
    """
    parameter_space = params_dict[list(params_dict)[0]]
    if len(params_dict) > 1:

        for n, key in enumerate(list(params_dict)[1:]):

            parameter_space = itertools.product(parameter_space, params_dict[key])
            parameter_space = [list(it) for it in parameter_space]
            if n > 0:

                parameter_space = [it[0] + [it[1]] for it in parameter_space]

    return list(parameter_space),list(params_dict)

def get_capitals(x,keep_numeric=True,keep_sybolic=True):
    """
    Given an input string, return the string containing only the capital letters now lowered.

    :param x: (str) input string.
    :return: (str) output string.
    """
    q = ''
    for l in x:

        add_cond = False
        if keep_numeric:

            add_cond = l.isdigit()

        add_cond2 = False
        if keep_sybolic:

            add_cond2 = l in ['_','*','-','+','/','.']

        if l.isupper() or add_cond or add_cond2:

            q += l

    return q.lower()

def delete_dict_key(dictionary,key):
    """
    Delete a kex from dictionary

    :param dictionary:
    :param key:
    :return:
    """
    dictionary.pop(key)
    return {n:dictionary[k] for n,k in enumerate(dictionary.keys())}


###############
#####   CLASSES
###############


class ParametersTracker:
    """
    Class used to track parameters passed as input to a method of a class.
    """

    def __init__(self,class_to_track):
        """
        Initialization. Parameters are tracked if they are initialized in a with statement.

        Example:

        .. code::

            # example class
            class Foo:

                def __init__(self,a):

                    self.a = a

                    # initialize parameter tracking
                    self.pt = ParametersTracker(self)

                def method(self,b,c):

                    # to track parameters in this method initialize them in a with statement.
                    with self.pt:

                        self.b = b
                        self.c = c

                    print(self.b + self.c)

                    # increment b of 1
                    self.b += 1

            foo = Foo(1)
            foo.method(3,4)
            foo.pt.get_current_tracked_params()

        :param class_to_track: (class) Class whose input parameters (of some method) need to be tracked.
        """
        self.tracked_parameters = []
        self.class_to_track = class_to_track

    def __enter__(self):

        self.pre_params = list(self.class_to_track.__dict__.keys())

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.post_params = list(self.class_to_track.__dict__.keys())
        tmp = [p for p in self.post_params if  p not in self.pre_params]
        self.tracked_parameters = tmp

    def get_current_tracked_params(self):
        """
        Return the current status of the parameters under tracking.

        :return: (dict) dictionary with the current values of the parameters tracked.
        """
        ctp = {}
        for p in self.tracked_parameters:

            ctp.update({p: self.class_to_track.__dict__[p]})

        return ctp

