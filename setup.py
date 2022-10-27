import setuptools
import os

with open('README.md','r') as readme:

    long_description = readme.read()

with open(os.path.dirname(os.path.abspath(__file__))+os.sep+'requirements.txt') as reqfile:
# with open('requirements.txt','r') as reqfile:

    requirements = reqfile.read().splitlines()

setuptools.setup(
    name='bmmltools',
    version='0.2.8',
    author='Luca Curcuraci',
    author_email='Luca.Curcuraci@mpikg.mpg.de',
    description = 'BioMaterial Machine Learning tools (bmmltools), package to do machine learning with large binary '
                  '3d images',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url='https://gitlab.mpikg.mpg.de/curcuraci/bmmltools',
    project_urls={'Bug tracker': 'https://gitlab.mpikg.mpg.de/curcuraci/bmmltools/-/issues/new',
                  'Documentation': 'https://bmmltools.readthedocs.io/en/latest/'},
    classifiers=['Programming Language :: Python :: 3.8',
                 'License :: OSI Approved :: Apache Software License',
                 'Operating System :: OS Independent',
                 'Topic :: Software Development :: Libraries :: Python Modules',
                 'Topic :: Scientific/Engineering :: Information Analysis',
                 'Topic :: Scientific/Engineering :: Bio-Informatics',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Intended Audience :: Science/Research'],
    package_dir={'': "src"},
    package_data={'': ['*.txt']},
    packages=setuptools.find_packages(where="src"),
    install_requires=requirements,
    include_package_data=True,
    python_requires='>=3.8',
)
