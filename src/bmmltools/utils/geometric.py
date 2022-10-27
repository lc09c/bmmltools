# Title: 'geometric.py
# Author: Curcuraci L.
# Date: 17/10/2022
#
# Scope: Collect geometric realted utils.

"""
Utilities for geometric transformations in bmmltools.
"""


#################
#####   LIBRARIES
#################


import numpy as np
import scipy.ndimage as spnd


#################
#####   FUNCTIONS
#################


# UOM Conversion

def grad_to_rad(angle):
    """
    Convert an angle in grad into an angle in rad.

    :param angle: (float) angle in grad
    :return: (float) angle in rad
    """
    return angle/180*np.pi


def rad_to_grad(angle):
    """
    Convert an angle in rad into an angle in grad.

    :param angle: (float) angle in rad
    :return: (float) angle in grad
    """
    return angle/np.pi*180

# Coordinates

def cartesian_to_spherical(x,y,z,grad=True):
    """
    (x,y,z) -> (r,theta,phi)

    :param x: x
    :param y: y
    :param z: z
    :param grad: (boolean) if True theta and phi are converted to grads
    :return: r,theta,phi
    """
    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y/x)
    if grad:

        theta = rad_to_grad(theta)
        phi = rad_to_grad(phi)

    return r,theta,phi

def spherical_to_cartesian(r,theta,phi,grad=True):
    """
    (r,theta,phi) -> (x,y,z)

    :param r: r
    :param theta: theta
    :param phi: phi
    :param grad: (boolean) if True theta and phi are converted to radiants.
    :return: x,y,z
    """
    if grad:

        theta = grad_to_rad(theta)
        phi = grad_to_rad(phi)

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x,y,z

def cartesian_to_cylindircal(x,y,z,grad=True):
    """
    (x,y,z) -> (r,theta,z)

    :param x: x
    :param y: y
    :param z: z
    :param grad: (boolean) if True theta is converted to grad.
    :return: r,theta,z
    """
    r = np.sqrt(x**2+y**2)
    theta = np.arctan2(y,x)
    if grad:

        theta = rad_to_grad(theta)

    return r,theta,z

def cylindrical_to_cartesian(r,theta,z,grad=True):
    """
    (r,theta,z) -> (x,y,z)

    :param r: r
    :param theta: theta in grad
    :param z: z
    :param grad: (boolean) if True theta is converted to radiants
    :return: x,y,z
    """
    if grad:

        theta = grad_to_rad(theta)

    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x,y,z


###############
#####   CLASSES
###############


class ChangeCoordinateSystem:
    """
    This class can be used to change the reference frame in which a volume is visualized.
    """

    def __init__(self,reference_frame_origin,xyz_to_XYZ_inv_map,xyz_to_XYZ_specs,use_xyz_ordering=True):
        """
        The new reference frame is specified by using the inverse of the map from the old to the new coordinated and
        from a specification dictionary (called 'xyz_to_XYZ_specs') which has to be the following structure:

            {'new_shape': (tuple) Shape of the volume in the new reference frame.

             'X_bounds': (tuple/list/numpy array) Min and max value of the X coordinate.

             'Y_bounds': (tuple/list/numpy array) Min and max value of the Y coordinate.

             'Z_bounds': (tuple/list/numpy array) Min and max value of the Z coordinate.

             'XYZ_ordering': (None or tuple/list/numpy array of int) Ordering of the coordinates in the new reference
             frame. If None the order is the one implicitly specified in the above fields.

             }


        :param reference_frame_origin: (tuple/list/numpy array) Position of the origin in the new reference frame.
        :param xyz_to_XYZ_inv_map: (python function with 3 inputs) Inverse mapping between the reference frame
                                   'xyz' and the reference frame 'XYZ'.
        :param xyz_to_XYZ_specs: (dict) Dictionary containing the specifications of the new reference frame.
        :param use_xyz_ordering: (bool) if True, the default ordering of the axis in the stack (which is 'zyx') is
                                 converted to the cartesian ordering (i.e. 'xyz').
        """
        self.reference_frame_origin = reference_frame_origin
        self.xyz_to_XYZ_inv_map = xyz_to_XYZ_inv_map
        self.new_shape = xyz_to_XYZ_specs['new_shape']
        self.X_min = xyz_to_XYZ_specs['X_bounds'][0]
        self.X_max = xyz_to_XYZ_specs['X_bounds'][1]
        self.n_X_steps = xyz_to_XYZ_specs['new_shape'][0]
        self.Y_min = xyz_to_XYZ_specs['Y_bounds'][0]
        self.Y_max = xyz_to_XYZ_specs['Y_bounds'][1]
        self.n_Y_steps = xyz_to_XYZ_specs['new_shape'][1]
        self.Z_min = xyz_to_XYZ_specs['Z_bounds'][0]
        self.Z_max = xyz_to_XYZ_specs['Z_bounds'][1]
        self.n_Z_steps = xyz_to_XYZ_specs['new_shape'][2]
        self.XYZ_ordering = None
        if xyz_to_XYZ_specs['XYZ_ordering'] is not None:

            self.XYZ_ordering = xyz_to_XYZ_specs['XYZ_ordering']

        self.use_xyz_ordering = use_xyz_ordering

    def transform(self,volume,**kwargs):
        """
        Apply the initialized transformation.

        :param volume: (numpy.ndarray) volume on which the transformation is applied.
        """

        # adopt the cartesian-like convention if needed
        if self.use_xyz_ordering:

            volume = volume.transpose(2,1,0)

        # create a a grid of points in the new reference frame
        Xs,Ys,Zs = np.meshgrid(np.linspace(self.X_min,self.X_max,self.n_X_steps),
                               np.linspace(self.Y_min,self.Y_max,self.n_Y_steps),
                               np.linspace(self.Z_min,self.Z_max,self.n_Z_steps),indexing='ij')

        # maps the point of the grid back to the original points in the initial reference frame
        xs,ys,zs = self.xyz_to_XYZ_inv_map(Xs,Ys,Zs)

        # Center the reference frame in the image center (z axis is already in the right position)
        xs = xs + self.reference_frame_origin[0]
        ys = ys + self.reference_frame_origin[1]
        zs = zs + self.reference_frame_origin[2]

        # create teh list of all the coordinates in which the image need to be evaluated
        xs, ys, zs = xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)
        coords = np.vstack((xs, ys, zs))

        # compute the volume in the new reference frame
        volume_in_new_coordinates = spnd.map_coordinates(volume, coords,**kwargs)
        volume_in_new_coordinates = np.reshape(volume_in_new_coordinates,self.new_shape)

        # change coordinate ordering if required
        if self.XYZ_ordering is not None:

            volume_in_new_coordinates = volume_in_new_coordinates.transpose(self.XYZ_ordering)

        return volume_in_new_coordinates
