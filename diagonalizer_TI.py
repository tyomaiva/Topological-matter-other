#!/usr/bin/env python

from __future__ import division

# kwant
import kwant

# standard imports
import numpy as np
from scipy.sparse import linalg as lag

import matplotlib as mpl
from matplotlib import pyplot as plt

from tinyarray import array as ta
from numpy import kron, sign, sqrt, fabs, sin, cos

from copy import copy

# Pauli matrices
sigma0 = ta([[1, 0], [0, 1]])
sigma1 = ta([[0, 1], [1, 0]])
sigma2 = ta([[0, -1j], [1j, 0]])
sigma3 = ta([[1, 0], [0, -1]])

# products of Pauli matrices
s0s0 = kron( sigma0,sigma0 )
s0s1 = kron( sigma0,sigma1 )
s0s2 = kron( sigma0,sigma2 )
s0s3 = kron( sigma0,sigma3 )

s1s0 = kron( sigma1,sigma0 )
s1s1 = kron( sigma1,sigma1 )
s1s2 = kron( sigma1,sigma2 )
s1s3 = kron( sigma1,sigma3 )

s2s0 = kron( sigma2,sigma0 )
s2s1 = kron( sigma2,sigma1 )
s2s2 = kron( sigma2,sigma2 )
s2s3 = kron( sigma2,sigma3 )

s3s0 = kron( sigma3,sigma0 )
s3s1 = kron( sigma3,sigma1 )
s3s2 = kron( sigma3,sigma2 )
s3s3 = kron( sigma3,sigma3 )



# Simple Namespace
class SimpleNamespace(object):
    """A simple container for parameters."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs) 



def onsite( site,p ):
    x,=site.pos
    pyLong = p.py + float(x-p.x_shift)*float(p.lBinv2)
    Onsite = (p.M0 - 2.*p.B1*(1-cos(p.pz)) - 2.*p.B2*(1-cos(pyLong)+1)) * s0s3 + p.A1*sin(p.pz) * s3s1 \
        + p.A2*sin(pyLong)* s2s1 + (p.C + 2.*p.D1*(1-cos(p.pz)) + 2.*p.D2*(1-cos(pyLong)+1)) * s0s0
    if hasattr(p, 'b0'):
        Onsite = Onsite + p.b0/2.*s2s3
    return Onsite
    
    
    
def hop( s1,s2,p ):
    return -p.D2 * s0s0 + p.B2 * s0s3 + 0.5j * p.A2 * s1s1



#Remark from https://kwant-project.org/doc/1.0/tutorial/tutorial1:
#sys[lat(i1, j1), lat(i2, j2)] = ... the hopping matrix element FROM point (i2, j2) TO point (i1, j1).
def FinalizedSystem( SitesCount_X ):
    # lattices
    lat = kwant.lattice.general( ( (1,), ) )
    # builder
    sys = kwant.Builder()
    # first, define all sites and their onsite energies
    for nx in range(SitesCount_X):
        sys[lat(nx,)]=onsite
    # hoppings
    for nx in range(SitesCount_X-1):
        sys[lat(nx+1,), lat(nx,)] = hop
    # finalize the system
    return sys.finalized()



def diagonalize( FinalizedSystem, Parameters ):
    ham_sparse_coo = FinalizedSystem.hamiltonian_submatrix( args=([Parameters]), sparse=True )
    #Conversion to some "compressed sparse" format
    ham_sparse = ham_sparse_coo.tocsc()
    #Finding k=EigenvectorsCount eigenvalues of the Hamiltonian H, which are located around sigma=0, 
    #so that they are the closest (wrt to the Hermitian norm) to this value. (The largest-in-magnitude label ('LM') is misleading,
    #since here we work in the "shift-invert" mode, so that actually the largest eigenvalues of the 1 / (H - omega) matrix 
    #are sought for. The numerical check is in agreement with this picture.)
    EigenValues, EigenVectors = lag.eigsh( ham_sparse, k=Parameters.EigenvectorsCount, return_eigenvectors=True, \
                                          which='LM', sigma=Parameters.FermiEnergy, tol=Parameters.EnergyPrecision )
    EigenVectors = np.transpose(EigenVectors)
    #Sorting the wavefunctions by eigenvalues, so that the states with the lowest energies come first
    idx = EigenValues.argsort()
    EigenValues = EigenValues[idx]
    EigenVectors = EigenVectors[idx]
    
    SitesCount_X = len(FinalizedSystem.sites)
    EigenVectors = np.reshape(EigenVectors, (Parameters.EigenvectorsCount, SitesCount_X, Parameters.WavefunctionComponents) )
    return EigenValues, EigenVectors



def pSweep( FinalizedSystem, Parameters, pMin, pMax, pCount, yORzSweep ):
    p = copy(Parameters)
    pSweep = np.linspace(pMin, pMax, pCount)
    EigenValuesSweep = []
    EigenVectorsSweep = []
    for index in pSweep:
        if yORzSweep == 'pzSweep':
            p.pz = index
        elif yORzSweep == 'pySweep':
            p.py = index
        else:
            raise ValueError("Only two values are possible for the parameter yORzSweep: either 'pzSweep' or 'pySweep'")
        
        EigenValues, EigenVectors = diagonalize( FinalizedSystem, p )
        EigenValuesSweep.append(EigenValues)
        EigenVectorsSweep.append(EigenVectors)
    
    return EigenValuesSweep, EigenVectorsSweep



def spectrum_plot( EigenValues, pMin, pMax, pCount, ShowPlot = True ):
    pSweep = np.linspace(pMin, pMax, pCount)
    mpl.rcParams['font.size'] = 16
    plt.plot(pSweep, EigenValues,"b.",markersize=3)
    plt.xlim(pMin,pMax)
    if ShowPlot == True:
        plt.show()    
    else:
        return plt