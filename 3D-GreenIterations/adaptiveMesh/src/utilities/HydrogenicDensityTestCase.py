'''
Mesh utilities for the adaptive mesh refinement.

@author: nathanvaughn
'''
from numpy import pi, cos, arccos, sin, sqrt, exp 
from scipy.special import erf
import numpy as np
from scipy.special import factorial, comb
import vtk



# def hydrogenicDensity(r,k):
#     return k**2 / (4*pi) * exp(-k * r)/r
#     
#     
# def hydrogenicHartreePotential(r,k):
#     return -exp(-k * r)/r
# 
# def hydrogenicHartreeEnergy(alpha):
#     return -k/2


def hydrogenicDensity(r,alpha):
    return alpha**2  * exp(-sqrt(4*pi*alpha**2) * r)/r
    
    
def hydrogenicHartreePotential(r,alpha):
    return exp(-sqrt(4*pi*alpha**2) * r)/r

def hydrogenicHartreeEnergy(alpha):
    return sqrt(pi)*alpha



def setDensityToHydrogenic(tree,alpha):
    
    for _,cell in tree.masterList:
        if cell.leaf == True:
            # set density on the primary mesh
            for i,j,k in cell.PxByPyByPz:
                gp = cell.gridpoints[i,j,k]
                r = sqrt( gp.x**2 + gp.y**2 + gp.z**2 )
                gp.rho = hydrogenicDensity(r,alpha)
                
                
def setTrueHydrogenicHartree(tree,alpha):
    
    for _,cell in tree.masterList:
        if cell.leaf == True:
            for i,j,k in cell.PxByPyByPz:
                gp = cell.gridpoints[i,j,k]
                r = sqrt( gp.x**2 + gp.y**2 + gp.z**2 )
                gp.trueHartree = hydrogenicHartreePotential(r,alpha)
                

    
    