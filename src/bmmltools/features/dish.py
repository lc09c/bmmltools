# title: 'dish_descriptor.py'
# date: 06/05/2021
# author: Curcuraci L.
#
# Scope: Class computing the dish descriptor.

"""
DISH descriptor.
"""

#################
#####   LIBRARIES
#################


import numpy as np
import pyshtools as psh
from scipy.ndimage import map_coordinates


###############
####    CLASSES
###############

class DISH:                 #TODO Commment this class!!!!!!!!!!!!!!!!!!!!!

    def __init__(self,L_max,R_min,R_max,n_different_radii,center=(0,0,0),interpolation_order=5,normalize=False):

        self.L_max = L_max
        self.R_min = R_min
        self.R_max = R_max
        self.n_different_radii = n_different_radii
        self.center = np.array(center)                  # in xyz convention
        self.interpolation_order = interpolation_order
        self.normalize = normalize

        self._setup()

    def _setup(self):

        self._compute_GLQ_grid()


    def _compute_GLQ_grid(self):

        # find (r,theta,phi) points of the GLQ grid
        rs_GLQ = np.linspace(self.R_min,self.R_max,self.n_different_radii)
        phis_GLQ = np.linspace(0,2*np.pi,2*self.L_max+1)
        self._zeros,self._ws = psh.expand.SHGLQ(self.L_max)
        thetas_GLQ = self._zeros*np.pi/2+np.pi/2

        # produce the interpolation mesh
        Rs,Thetas,Phis = np.meshgrid(rs_GLQ,thetas_GLQ,phis_GLQ, indexing='ij')
        Rs,Thetas,Phis = Rs.reshape(-1),Thetas.reshape(-1),Phis.reshape(-1)

        # spherical to cartesian
        self._Xs = Rs*np.cos(Phis)*np.sin(Thetas)+self.center[0]
        self._Ys = Rs*np.sin(Phis)*np.sin(Thetas)+self.center[1]
        self._Zs = Rs*np.cos(Thetas)+self.center[2]

        # save grid information
        self.grid = {'rs': rs_GLQ,
                     'thetas': thetas_GLQ,
                     'phis': phis_GLQ}
        self.grid_shape = (len(rs_GLQ),len(thetas_GLQ),len(phis_GLQ))

    def _evaluate_3dvolume_on_GLQ_grid(self,data):

        # evaluation of the 3d volume on the GLQ grid
        data_on_GLQ_grid = map_coordinates(data, [self._Zs,self._Ys,self._Xs],order=self.interpolation_order)
        return data_on_GLQ_grid.reshape(self.grid_shape)

    def compute(self,data,radius_refactor=1):

        # evaluate data on GLQ grid
        data_on_GLQ_grid = self._evaluate_3dvolume_on_GLQ_grid(data)

        # compute the SH fingerprint
        SH_fingerprint = []
        if self.normalize:

            weights = []

        for R_selected in range(0,self.n_different_radii,radius_refactor):

            if radius_refactor>1:

                data_to_expand = data_on_GLQ_grid[R_selected:R_selected+radius_refactor,...].mean(axis=0)

            else:

                data_to_expand = data_on_GLQ_grid[R_selected,...]

            grid_GLQ = psh.SHGrid.from_array(data_to_expand, grid='GLQ')
            coefficients_GLQ = grid_GLQ.expand(normalization='ortho')
            sh_coeff_GLQ = psh.SHCoeffs.from_array(coefficients_GLQ.coeffs, normalization='ortho')
            SH_fingerprint_line = sh_coeff_GLQ.spectrum(unit='per_l')
            if self.normalize:

                SH_fingerprint_line = SH_fingerprint_line/np.sum(SH_fingerprint_line)
                dte_x,dte_y = np.gradient(data_to_expand)
                weights.append(np.sqrt(np.mean(dte_x**2+dte_y**2)))

            SH_fingerprint.append(SH_fingerprint_line)

        SH_fingerprint = np.array(SH_fingerprint)
        if self.normalize:

            weights = np.array(weights)/np.sum(weights)
            SH_fingerprint = np.expand_dims(weights,axis=-1)*SH_fingerprint

        return SH_fingerprint