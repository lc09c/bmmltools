==============
Data and Trace
==============


bmmltools is organized around two key objects:

* :ref:`Data <data_section>`: which is used to load external data;

* :ref:`Trace <trace_section>`: which is used to store all the partials results, and more generally execute various
  task during the application of a series of operations to some input data.


.. _data_section:

Data
====


:py:mod:`Data <bmmltools.core.data.Data>` is the python module used to load data in bmmltools. Input data are loaded and
stored in an `hdf5 <https://docs.h5py.org/en/stable/index.html>`_ file, saved in a folder selected by the user.
:py:mod:`Data <bmmltools.core.data.Data>` is able to load may formats in different ways, which are listed below. Each
of these input type has its own specific input method in :py:mod:`Data <bmmltools.core.data.Data>` which is indicated
between parenthesis.

* *stacks* save in a multitiff (see :py:mod:`load_stack <bmmltools.core.data.Data.load_stack>`);

* *stacks* saved as tiff slice-by-slice in a folder (see :py:mod:`load_stack_from_folder
  <bmmltools.core.data.Data.load_stack_from_folder>`);

* *numpy arrays* which are homogenous in data-type and contains only numeric or boolean data (see :py:mod:`from_array
  <bmmltools.core.data.Data.from_array>`);

* *file .npy* containing numpy array homogenous in data-type and contains only numeric or boolean data (see
  :py:mod:`load_npy <bmmltools.core.data.Data.load_npy>`);

* *pandas dataframe* (see :py:mod:`from_pandas_df <bmmltools.core.data.Data.from_pandas_df>`);

* *json files* containing pandas dataframe (see :py:mod:`load_pandas_df_from_json
  <bmmltools.core.data.Data.load_pandas_df_from_json>`);

* *csv files* containing pandas dataframe (see :py:mod:`load_pandas_df_from_csv
  <bmmltools.core.data.Data.load_pandas_df_from_csv>`).

In addition to these, there are two main methods:

* :py:attr:`new <bmmltools.core.data.Data.new>`, to create a new hdf5 file to store that data loaded. The hdf5 file will
  be create in a folder, specified in the ``working_folder`` field of this function. The code below show how to create
  a new :py:mod:`Data <bmmltools.core.data.Data>` object from numpy array

  .. code::

     import numpy as np
     from bmmltools.core.data import Data


     # initialize a Data object
     d = Data()
     d.create(working_folder = r'[PATH TO SOME FOLDER]')

     # load data from a numpy array
     arr = np.random.uniform(0,1,size=10)
     d.from_array(arr,'x')

  With the code above, the content of the numpy array ``arr`` is stored in the hdf5 file in a dataset called ``x``.

  There is also the possibility to specify the working folder directly in the data initialization. Keeping this in mind
  the initialization lines in the code above can be reduced to

  .. code::

     d = Data(working_folder = r'[PATH TO SOME FOLDER]')

  .. note::

     The name of the hdf5 file created by :py:mod:`Data <bmmltools.core.data.Data>` has a standard structure, which is
     ``data_XXXX.hdf5``. ``XXXX`` is a 4 digits code which is randomly generated once the file is created in order to
     uniquely identify this file: these 4 digits are called *trace code*.


* :py:attr:`link <bmmltools.core.data.Data.link>`, to link the data object initialized to an already existing hdf5 file
  (typically already containing some dataset previously loaded). To link an initialized :py:mod:`Data
  <bmmltools.core.data.Data>` object to an existing hdf5 file, one need to specify the folder where the file is in the
  field ``working_folder``, and the data code (as string) in the field ``data_code``.

  The code lines below show how this can be done.

  .. code::

     from bmmltools.core.data import Data


     # initialize a Data object
     d = Data()

     # link an existing hdf5 file.
     d.link(working_folder=r'[PATH TO FOLDER WITH data_XXXX.hdf5 FILE]',data_code='XXXX')

  .. note::

     :py:attr:`infodata <bmmltools.core.data.Data.infodata>` is method of a :py:mod:`Data <bmmltools.core.data.Data>`
     object which can be used to print the current datasets present in the hdf5 file linked to it.

     .. code::

        d.infodata()

     This command is particularly useful to inspect a :py:mod:`Data <bmmltools.core.data.Data>` objects after linking
     to check the content of the hdf5 file linked.

Once :py:mod:`Data <bmmltools.core.data.Data>` object are created and filled with some input method or linked to some
hdf5, the dataset can be used by specifying the its name within square parenthesis, as showed in the example before.
After these square parenthesis one can use the slicing notation of
`h5py <https://docs.h5py.org/en/stable/high/dataset.html>`_, which mimic the numpy slicing notation.

.. code::

   print(d['x'][0])

The line above print the 0-th element of the dataset called ``x`` present in the hdf5 file linked to the
:py:mod:`Data <bmmltools.core.data.Data>` object. Alternatively one can use :py:attr:`use_dataset
<bmmltools.core.data.Data.use_dataset>` , which can be particularly useful if the dataset have to be used many times.
Consider the example below

.. code::

   # ...
   # [creation and filling of a Data object or linking to an hdf5 file]
   # ...

   # select a dataset
   d.use_dataset('x')
   print(d[0])
   print(d[1])

   # unselect a dataset
   d.use_dataset(None)
   print(d['x'][0])     # <- This should work.
   print(d[1])          # <- This should give rise to error.

In the code above, the dataset ``x`` is first specified, and then every time the data object is called the use of this
particular dataset is assumed: the first two prints will print the elements 0 and 1 of the dataset without the need of
specifying the dataset name two times. To "unselect" a dataset ``None`` should be given as argument of
:py:attr:`use_dataset <bmmltools.core.data.Data.use_dataset>`. As showed in the example above, in this case one have to
proceed in the standard way, as the two last line of code above should show.



.. _trace_section:

Trace
=====


:py:mod:`Trace <bmmltools.core.tracer.Trace>` is the core class of bmmltoools. It is used to track all the intermediate
results during the application of a series of operation, in automatic manner and *without keeping these results in the
computer RAM*. :py:mod:`Trace <bmmltools.core.tracer.Trace>` produces a series of file in a folder called
*trace folder*, which is a folder created at a path specified by the user (see below). The trace folder has a standard
name: ``trace_XXXX``, where ``XXXX`` is a random 4 digits number (called *trace code*) uniquely identifying the trace.
The files generated by :py:mod:`Trace <bmmltools.core.tracer.Trace>` are listed and explained below.

* *trace hdf5*: here the intermediate results are stored. This file is produced once the trace is created (see below)
  and is the file which can be linked to a :py:mod:`Trace <bmmltools.core.tracer.Trace>` object.

* *trace json*: here the trace graph, i.e. all the information to reconstruct the sequence of operations applied on a
  given trace, and the parameters of the various operations applied on the trace are stored in a dictionary-like format.

* *trace dill*: here the initialized operation applied on a trace are saved as dill object once the application of them
  terminate (i.e. they are saved in the state they have at the end of the application on a dataset contained in the
  trace). This file is produced only if ``enable_trace_graph = True`` when the
  :py:mod:`Trace <bmmltools.core.tracer.Trace>` object is initialized (this is the default setting).

When operations act on a trace they can produce a series of folders where various files are saved during the application
on the trace. The :py:mod:`Trace <bmmltools.core.tracer.Trace>` object is also responsible for the creation and
organization in a standard way of these folder. These folders are organized as follows.

* *trace file folder*, to save the intermediate quantities produced during the application of an operation. The path
  to this folder is standard (it is a folder called ``trace_files`` inside the trace folder) and can be obtained
  calling the method :py:func:`trace_file_path <bmmltools.core.tracer.Trace.trace_file_path>`.

* *trace readings folder*, to save the final result one has at the end of the application of an operation (i.e. possibly
  an intermediate result of the application of a series of operations). The path to this folder is standard (it is a
  folder called ``trace_readings`` inside the trace folder)and can be obtained calling the method
  :py:func:`trace_readings_path <bmmltools.core.tracer.Trace.trace_readings_path>`.

* *trace outputs folder*, the :ref:`output operations <output_raw_labels_section>` store the files produced in this
  folder when they are applied. The path to this folder is standard (it is a folder called ``trace_outputs`` inside
  the trace folder) and can be obtained calling the method
  :py:func:`trace_outputs_path <bmmltools.core.tracer.Trace.trace_outputs_path>`.

Form a practical point of view, :py:mod:`Trace <bmmltools.core.tracer.Trace>` works similarly to `Data <data_section>`_.
More precisely, a :py:mod:`Trace <bmmltools.core.tracer.Trace>` object once initialized needs to create or to be linked
to an hdf5 file. Two methods are used for that:

* :py:mod:`create <bmmltools.core.tracer.Trace.create>` is used to create an hdf5 file (and a trace json too). To create
  an hdf5 file one needs to specify a folder where the trace folder is created. This is done by specifying the path in
  the ``working_folder`` field. It is also possible to specify the
  `group <https://docs.h5py.org/en/stable/high/group.html>`_ where all the intermediate results are stored in the field
  ``group_name``. By default the group is the root of the hdf5 file, i.e. the intermediate results are stored in the
  dataset ``/[variable_name]``. When the group is specified, the intermediate results are saved at
  ``/[group_name]/[variable_name]`` . The code below show how to initialize a new trace.

  .. code::

     from bmmltools.core.trace import Trace


     # initialization with creation of necessary file of a trace
     t = trace()
     t.create(working_folder=r'[SOME FOLDER PATH]', group_name='[GROUP NAME]')

  It is not mandatory to used groups inside a trace but they can be useful: groups can used to give some internal
  organization to the hdf5 trace file, keeping separated intermediate results coming from different pipelines of
  operations, for example.

* :py:mod:`link <bmmltools.core.tracer.Trace.link>` is used to link an initialized
  :py:mod:`Trace <bmmltools.core.tracer.Trace>` object to an already existing  hdf5 file (and json file) containing the
  trace. To do that one needs to specify the path to the trace folder in the ``trace_folder`` field, and the name of
  the group (if any) in the ``group_name`` field

  .. code::

     from bmmltools.core.trace import Trace


     # initialization of a trace object with link to an existing trace folder
     t = trace()
     t.link(trace_folder=r'[TRACE FOLDER PATH]', group_name='[GROUP NAME]')

Since a trace can be organized in groups, one can create a new group or change the group used to store the data. This
can be done using the methods :py:mod:`change_group <bmmltools.core.tracer.Trace.change_group>` and
:py:mod:`create_group <bmmltools.core.tracer.Trace.create_group>` whose meaning is self-explaining.

.. attention::

   It is possible to specify in the trace the seed used for all the random steps of the various operations applied
   on the trace. This can be done right ater the creation/linking of an hdf5 file simply as showed below

   .. code::

      #...
      t.seed = 5

Given a :py:mod:`Trace <bmmltools.core.tracer.Trace>` object linked to some hdf5 file, one can initialize a new variable
in the trace, recover the content of a variable tracked on the trace, or delete a variable using the python's standard
ways. The example below shows the basic usage of a :py:mod:`Trace <bmmltools.core.tracer.Trace>` object.

.. code::

   from bmmltools.core.trace import Trace


   # initialize a trace creating all necessary trace files
   t = trace()
   t.create(working_folder=r'[SOME FOLDER PATH]')

   # add an initialized variable to the trace
   t.x = 4

   # recover a variable from the trace
   print(t.x)

   # change value to a variable on the trace
   t.x = 5
   print(t.x)

   # remove a variable from the trace
   del t.x
   print(t.x) # <- this should give rise to error.

It is important to keep in mind that the variable ``x`` is in RAM only the time necessary to print it: for the rest of
the time the variable is stored in an hdf5 file. This is particularly useful when one has to use many different
variables containing data occupying a lot of RAM. Note that in the example above the whole content of ``x`` is loaded
in RAM.

Finally, also :py:mod:`Trace <bmmltools.core.tracer.Trace>` has a method to get information over the trace content,
which is :py:mod:`infotrace <bmmltools.core.tracer.Trace.infotrace>`. This method can be used to get the names of the
variables that are currently under tracking *on the hard disk*, the variable type, the groups available on the trace,
and group currently used to store the variables.

.. code::

   t.infotrace()


.. _supported_variable_type_subsection:

Supported variable types
------------------------

Trace is able to automatically store-read-delete variables on the Hard Disk (i.e. inside the trace hdf5 file) only if
they are of specific formats. These formats are listed below.


* **Homogenous numpy array**: namely numpy arrays of any shape and dimension hose elements are numbers and all of the
  same type, i.e. only boolean,integer,float or complex.

  .. code::

     import numpy as np
     ...

     # ...
     # [initialization and linking to an hdf5 file of a trace object]
     # ...

     # create an nd array
     arr = np.random.uniform(0,1,size=(10,10,10))

     # store value of arr in x then erase from the RAM
     trace.x = arr
     del arr

     # read the whole x and print the content
     print(trace.x)


* **Homogenous numeric dataframe**: namely pandas dataframe whose elements are all of the same numeric type. The numeric
  types supported are the same of the previous data format.

  .. code::

     import pandas as pd
     ...

     # ...
     # [initialization and linking to an hdf5 file of a trace object]
     # ...

     # create an pandas dataframe
     df = pd.DataFrame({'X':[1,2,3,4],'Y':[5,6,7,8],'Z':[9,10,11,12]})

     # store value of arr in y then erase from the RAM
     trace.y = df
     del df

     # read the whole y and print the content
     trace.y


* **Dictionary of the two variable types listed above**: namely a dictionary whose keys are homogenous numpy arrays
  and/or homogenous numeric dataframe. One can read and write individual keys of the dictionary by using the methods
  :py:mod:`read_dictionary_key <bmmltools.core.tracer.Trace.read_dictionary_key>` and
  :py:mod:`write_dictionary_key <bmmltools.core.tracer.Trace.write_dictionary_key>`.

  .. code::

     import numpy as np
     import pandas as pd
     ...

     # ...
     # [initialization and linking to an hdf5 file of a trace object]
     # ...

     # create a dictionary to save
     dictionary_to_trace = {'x': np.random.uniform(0,1,size=(10,10,10)),
                            'y': pd.DataFrame({'X':[1,2,3,4],'Y':[5,6,7,8],'Z':[9,10,11,12]})}

     # store value of arr in x then erase from the RAM
     trace.dictionary = dictionary_to_trace
     del dictionary_to_trace

     # read the whole 'dictionary' and print the content
     print(trace.dictionary)

     # read just one key of 'dictionary'
     trace.read_dictionary_key('dictionary','x')

     # write just one key of 'dictionary'
     trace.read_dictionary_key('dictionary','x',np.array([1,2,3]))


* **External link to dataset in other hdf5 files**: it is used to avoid to copy the content of the input dataset which
  is present in :py:mod:`Data <bmmltools.core.data.Data>` object, saving space on the Hard Disk.

  .. note::

     This external link depends on the path to the :py:mod:`Data <bmmltools.core.data.Data>` object. Therefore if the
     content of the folder created by :py:mod:`Data <bmmltools.core.data.Data>`, where  its hdf5 file is created, is
     changed, the external link would not work (see `external links <https://docs.h5py.org/en/stable/high/group.html>`_
     in h5py).

What does not fall in these categories can be added to a trace but its content remain in RAM.

.. note::

   The decision on where the variables are stored (in the hdf5 file or in RAM) is done automatically by
   :py:mod:`Trace <bmmltools.core.tracer.Trace>` and cannot be selected by the user.