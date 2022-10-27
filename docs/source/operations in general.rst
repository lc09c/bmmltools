=====================
Operations in general
=====================


In bmmltools operations are the collective name of all the transformations that one can apply to some input dataset to
produce some other dataset stored in the same trace of the input. They have all the same structure, which is discussed
here, while the details about the specific operations can be found in the "Operations" section of the side-menu'.


**Initialization**

Operations are always initialized passing to them a trace object. This is the trace object on which they acts, i.e.
from which they take the input dataset and save the operation result. For some operation one needs to specify additional
information, like python function or classes having a specific structure.

As example consider the code below.

.. code::

  # ...
  from bmmltools.operations.featore import Binarizer

  # ...
  trace = ...  # this is some trace object
  # ...

  binariezer = Binarizer(trace)


**Setting inputs and outputs**

Right after initialization typically one has to specify the inputs and or the outputs of the operation. In particular,
one has to specify the names of the dataset in the trace used as input, and (possibly) the name(s) given to the
dataset(s) created in the trace hdf5 file where the operation results are saved. When just one input or output is
required one may simply specify the name in a string, while when there are more a list of string need to be used. Every
operation has 3 methods to deal with these setting:

- method ``i``, to specify the inputs;

- method ``o``, to specify the outputs;

- methods ``io``, to specify at the same time both the inputs (in the first argument) and outputs (in the second
  argument).

It is not necessary to call the methods every time a new operation is initialized. If the output is not specified, every
operation has a default name for the output (i.e. the output is always already specified). Clearly the input method
``i`` is needed for all the operations  one want to :ref:`"apply" <apply_subsection>` (see below), but is not needed when
one want just :ref:`"read" <read_subsection>` the output dataset. It is important to know that these methods returns the
operation class itself. In reference to the example above, the specification of the inputs and outputs can be done with
the line of code below

.. code::

   #...
   binarizer.io('input_dataset','binarized_dataset')

.. note::

   The previous line of code is equivalent to

   .. code::

      #...
      binarizer.i('input_dataset').o('binarized_dataset')

.. _apply_subsection:

**Apply the operation**

Every operation has an ``apply`` methods, which execute the operation on the inputs when is called. the results
Operations typically depends on parameters. All the parameters need to be specified as argument of the ``apply`` method.
Differently to the other two situations above, this method does not return the class itself but the list of the
outputs. The example below show how to use it, continuing the example above

.. code::

   #...
   x = binarizer.apply(threshold=0.2)

   print(type(x))
   print(x)

Sometime during the application of the operation certain files are produced. These files are saved in the trace reading
folder of the trace, creating a folder having the name `` [OPERATION NAME]_X`` where ``X`` is a number counting the
number of operations applied on that trace since the input layer.

.. note::

   Each operation has an attribute called ``pt``, where one can find a dictionary containing all the parameters given
   to the apply method. These parameters are stored in the trace json file.

.. _read_subsection:

**Read operation results**


The final result of the application of the operation is stored in the hdf5 file of the trace. Despite the user is free
to open this file with traditional methods and take the operation result, each operation is equipped with a ``read``
methods whose goal is to make the operation output "more readable" to the user. When ``read`` is called, a folder
having the name `` [OPERATION NAME]_X`` (``X`` is a number counting the number of operations applied on that trace
since the input layer) is created in the reading folder of the trace: in this folder all the outputs of the operation
are saved in a suitable format. This method return None.

It is important to observe that one do not need to use exactly the same operation used to produce the output dataset
(despite the class *has to* be  the same). this should be clarified in the continuation of the previous example which
can be found below

.. code::

   #...
   binarizer2 = Binarizer(trace).o('binarized_dataset') # note that this is a new Binarizer object
                                                        # which is used to read the result of 'binarizer'.
   binarizer2.read()



Example of operation usage
==========================


The code below is a realistic example of how one should use operations (together
:py:mod:`Trace <bmmltools.core.tracer.Trace>` and :py:mod:`Data <bmmltools.core.data.Data>` objects). In particular
the :py:mod:`Input <bmmltools.operations.io.Input>` (see :ref:`input operation page <input_section>`) and
:py:mod:`PatchTransform3D <bmmltools.operations.feature.PatchTransform3D>`  (see :ref:`patch transform operation page
<patch_transform_section>`)

.. code::

   import numpy as np
   from bmmltools.core.data import Data
   from bmmltools.core.tracer import Trace
   from bmmltools.operations.feature import PatchTransform3d


   # input data
   data = Data()
   data.new(working_folder=r'SOME PATH')
   data.from_array(np.arange(0,27).reshape((3,3,3)),'INPUT DATASET NAME')

   # initialize the trace
   trace = Trace()
   trace.create(working_folder=r'SOME OTHER PATH',group_name='SOME GROUP NAME')

   # some operation
   x = Input(trace).i('INPUT DATASET NAME').apply(data)

   f = lambda x: x**2+x+1                                 # function applied to a patch
   x = PatchTransform3d(trace,f).io(x,'output_data').apply()


.. attention::

    Alternatively the last 3 lines can be replaced with the two lines below

    .. code::

       f = lambda x: x**2+x+1
       PT3D = PatchTransform3d(trace,f).io(x,'output_data')
       x = PT3D.apply()

    The first version can be considered as a sort of "RAM-efficient" version of the code, since the operation class
    remains in RAM just the time necessary to execute the initialization, setting and the apply method. On the other
    hand this second approach can be better suited to have access to internal parameters of the operation which can
    be of some interest.