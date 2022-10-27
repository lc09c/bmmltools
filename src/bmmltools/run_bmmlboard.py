# Title: 'run_bmmlboard.py'
# Author: Curcuraci L.
# Date: 13/10/2022
#
# Scope: run bmmlboard from terminal.

"""
Script used to run the bmmlboard from terminal once the bmmltools library is installed.
"""


#################
#####   LIBRARIES
#################


import os
import subprocess

from pathlib import Path


############
#####   MAIN
############


if __name__ == '__main__':

    bmmltools_dir = str(Path(__file__).parent.absolute())
    path = bmmltools_dir+os.sep+'board'+os.sep+'home.py'
    subprocess.run(['streamlit','run',path])