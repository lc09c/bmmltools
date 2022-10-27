# Title: 'dft.py'
# Author: Curcuraci L.
# Date: 18/10/2022
#
# Scope: collect DFt related functions.

"""
DFT related functions.
"""

#################
#####   LIBRARIES
#################


import numpy as np
import mkl_fft


########################
#####   GLOBAL CONSTANTS
########################


eps = 1e-9                      # epsilon for numerical regularization


#################
#####   FUNCTIONS
#################


def dft(x,backend='mkl'):
    """
    Discrete fourier transform.

    :param x:
    :param backend:
    :return:
    """
    if backend == 'mkl':

        return mkl_fft.fftn(x)

    else:

        return np.fft.fftn(x)

def idft(x,backend='mkl'):
    """
    Inverse discrete fourier transform.

    :param x:
    :param backend:
    :return:
    """
    if backend == 'mkl':

        return mkl_fft.ifftn(x)

    else:

        return np.fft.ifftn(x)

def periodic_smooth_decomposition_dft2(x,backend='mkl'):
    """
    Periodic-smooth decomposition of the the discrete fourier transform of 2d data.

    :param x:
    :return:
    """
    x = x.astype(np.float64)
    x_hat = dft(x,backend)

    # compute data borders
    B = np.zeros(x.shape, dtype=np.float64)
    B[0,:] = x[-1,:]-x[0,:]
    B[-1,:] = x[0,:]-x[-1,:]
    B[:,0] = B[:,0]+(x[:,-1]-x[:,0])
    B[:,-1] = B[:,-1]+(x[:,0]-x[:,-1])
    B_hat = dft(B,backend)

    # compute smooth components
    M, N = B_hat.shape
    q, r = np.meshgrid(np.arange(M), np.arange(N),indexing='ij')
    den = (-4 + 2 * np.cos(2 * np.pi * q / M) + 2 * np.cos(2 * np.pi * r / N)).astype(B_hat.dtype)
    S_hat = B_hat / (den + eps)
    S_hat[0, 0] = 0

    return x_hat - S_hat, S_hat

def periodic_smooth_decomposition_dft3(x,backend='mkl'):
    """
    Periodic-smooth decomposition of the the discrete fourier transform of 3d data.

    :param x:
    :return:
    """
    x = x.astype(np.float64)
    x_hat = dft(x,backend)

    # compute data borders
    B = np.zeros(x.shape, dtype=np.float64)
    B[0,:,:] = x[-1,:,:]-x[0,:,:]
    B[-1,:,:] = x[0,:,:]-x[-1,:,:]
    B[:,0,:] = B[:,0,:]+(x[:,-1,:]-x[:,0,:])
    B[:,-1,:] = B[:,-1,:]+(x[:,0,:]-x[:,-1,:])
    B[:,:,0] = B[:,:,0]+(x[:,:,-1]-x[:,:,0])
    B[:,:,-1] = B[:,:,-1]+(x[:,:,0]-x[:,:,-1])
    B_hat = dft(B,backend)

    # compute smooth components
    P,M,N = B_hat.shape
    p,m,n = np.meshgrid(np.arange(P),np.arange(M),np.arange(N),indexing='ij')
    den = (-6+2*np.cos(2*np.pi*p/P)+2*np.cos(2*np.pi*m/M)+2*np.cos(2*np.pi*n/N)).astype(B_hat.dtype)
    S_hat = B_hat/(den+eps)
    S_hat[0,0,0] = 0

    return x_hat-S_hat, S_hat