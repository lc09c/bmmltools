# Title: 'tracegraph.py'
# Author: Curcuraci L.
# Date: 18/10/2022
#
# Scope: functions used for visualization in bmmlboard

"""
Visualization tools used in bmmlboard.
"""

def plotly_color_palette(color_palette):
    """
    Eliminate the 1.0 value from seaborn-generated color palette, which is not compatible with plotly colomaps. Every
    1.0 value is replaced with 0.9999.

    :param color_palette: seaborn color palette
    :return: plotly compatible color palette
    """
    ctmp = []
    for c in color_palette:

        color = []
        for n in c:

            if n == 1.0:

                n = 0.99999

            color.append(n)

        ctmp.append(tuple(color))

    return ctmp