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




# Simple Namespace
class SimpleNamespace(object):
    """A simple container for parameters."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs) 
        
        
        
        
def onsite( site,p ):
    x,=site.pos
    pyLong = p.py + float(x-p.x_shift)*float(p.lBinv2)
    Onsite = sin(pyLong) * sigma1 + p.Gap * sigma2 # + cos(p.pz) * sigma0 
    if (x == 0):
        if hasattr(p, 'Rescale_onsite_0'):
            Onsite = Onsite * p.Rescale_onsite_0
    return Onsite
    
    
    
def hop( s1,s2,p ):
    Hop = 0.5j * sigma3
    return Hop



#Remark from https://kwant-project.org/doc/1.0/tutorial/tutorial1:
#sys[lat(i1, j1), lat(i2, j2)] = ... the hopping matrix element FROM point (i2, j2) TO point (i1, j1).
def FinalizedSystem_trivial( SitesCount_X ):
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
    
    

def density_plot( FinalizedSystem, Parameters, EigenVectors ):
    position_1D = realspace_position(FinalizedSystem)
    SitesCount_X = len(FinalizedSystem.sites)    
    
    density = [[np.vdot(EigenVectors[i][position_1D[j]],EigenVectors[i][position_1D[j]]) for j in range(SitesCount_X)] \
               for i in range(Parameters.EigenvectorsCount)]
    density = np.real(density)

    plt.pcolor( np.linspace(0,SitesCount_X,SitesCount_X), np.linspace(0,1,Parameters.EigenvectorsCount), density, \
               vmin = 0, vmax= 3./SitesCount_X)
    plt.colorbar()
    plt.show()
    
    
    
def realspace_position( FinalizedSystem ):
    SitesCount_X = len(FinalizedSystem.sites)
    return [[j for j in range(SitesCount_X) if FinalizedSystem.sites[j].pos[0] == i][0] for i in range(SitesCount_X)]    