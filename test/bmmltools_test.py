# Title: 'bmmltools_test.py'
# Author: Curcuraci L.
# Date: 20/10/2022
#
# Scope: Supersimple unit test for bmmltools.

#################
#####   LIBRARIES
#################


import unittest
import warnings


###############
#####   CLASSES
###############


class BmmltoolsTest(unittest.TestCase):

    def setUp(self):

        warnings.simplefilter('ignore', category=ImportWarning)
        warnings.simplefilter('ignore', category=DeprecationWarning)
        warnings.simplefilter('ignore', category=RuntimeWarning)

    def test_imports(self):

        print('\nRunning imports test...')

        import_list = ['import bmmltools',
                       'import bmmltools.core.data',
                       'import bmmltools.core.tracer',
                       'import bmmltools.operations.clustering',
                       'import bmmltools.operations.feature',
                       'import bmmltools.operations.io',
                       'import bmmltools.operations.segmentation',
                       'import bmmltools.features.dft',
                       'import bmmltools.features.dish',
                       'import bmmltools.utils.basic',
                       'import bmmltools.utils.io_utils',
                       'import bmmltools.utils.graph',
                       'import bmmltools.utils.geometric']
        import_failure = []
        for elem in import_list:

            try:

                exec(elem)
                import_failure.append(False)

            except:

                import_failure.append(True)

        failure = sum(import_failure) > 0
        idx = -1
        if failure:

            idx = import_failure.index(True)

        self.assertEqual(failure,False,'Import command \'{}\' failed!'.format(import_list[idx]))

        print('...DONE!')

############
#####   MAIN
############


if __name__ == '__main__':

    unittest.main()
