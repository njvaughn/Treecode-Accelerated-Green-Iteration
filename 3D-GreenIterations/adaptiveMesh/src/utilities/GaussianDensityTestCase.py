'''
Mesh utilities for the adaptive mesh refinement.

@author: nathanvaughn
'''
from numpy import pi, cos, arccos, sin, sqrt, exp 
from scipy.special import erf
import numpy as np
from scipy.special import factorial, comb
import vtk



def gaussianDensity(r,alpha):
        return alpha**3 / pi**(3/2) * exp(-alpha**2 * r**2)
    
    
def gaussianHartree(r,alpha):
        return erf(alpha*r)/r

def hartreeEnergy(alpha):
    return sqrt(2/pi)*alpha



def setDensityToGaussian(tree,alpha):
    
    for _,cell in tree.masterList:
        if cell.leaf == True:
            # set density on the primary mesh
            for i,j,k in cell.PxByPyByPz:
                gp = cell.gridpoints[i,j,k]
                r = sqrt( gp.x**2 + gp.y**2 + gp.z**2 )
                gp.rho = gaussianDensity(r,alpha)
                
        
def setIntegrand(tree,helmholtzShift):
    
    for _,cell in tree.masterList:
        if cell.leaf == True:
            # set density on the primary mesh
            for i,j,k in cell.PxByPyByPz:
                gp = cell.gridpoints[i,j,k]
                gp.f = gp.rho +  (helmholtzShift**2/(4*pi)) *gp.trueHartree
                

def setTrueHartree(tree,alpha):
    
    for _,cell in tree.masterList:
        if cell.leaf == True:
            for i,j,k in cell.PxByPyByPz:
                gp = cell.gridpoints[i,j,k]
                r = sqrt( gp.x**2 + gp.y**2 + gp.z**2 )
                gp.trueHartree = gaussianHartree(r,alpha)
                
def integrateCellDensityAgainst__(cell,integrand):
            rho = np.empty((cell.px,cell.py,cell.pz))
            pot = np.empty((cell.px,cell.py,cell.pz))
            
            for i,j,k in cell.PxByPyByPz:
                gp = cell.gridpoints[i,j,k]
                rho[i,j,k] = gp.rho  
                
                pot[i,j,k] = getattr(gp,integrand)
            
            return np.sum( cell.w * rho * pot)
        
def computeHartreeEnergyFromAnalyticPotential(tree):
    E = 0.0
    for _,cell in tree.masterList:
        if cell.leaf == True:
            E += integrateCellDensityAgainst__(cell,'trueHartree') 
    return E

def computeHartreeEnergyFromNumericalPotential(tree):
    E = 0.0
    for _,cell in tree.masterList:
        if cell.leaf == True:
            E += integrateCellDensityAgainst__(cell,'v_coulomb') 
    return E


     
    
    