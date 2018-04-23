'''
Mesh utilities for the adaptive mesh refinement.

@author: nathanvaughn
'''
import numpy as np

def meshDensity(N,r):
    '''
    Mesh density function from Wilkinson and Levine for order 2, total gridpoints roughly N
    :param N:
    :param r:
    '''
    
    ''' for order = 1 '''
#     return N/25.191*(np.exp(-2*r)* (4 - 2/r + 9/r**2) )**(3/5)
    
    ''' for order = 2 '''
#     return N/412.86*(np.exp(-2*r)* (64 - 78/r + 267/r**2 + 690/r**3 + 345/r**4) )**(3/7)


    ''' for order = 3 '''
    return N/648.82*(np.exp(-2*r)* (52 - 102/r + 363/r**2 + 1416/r**3 + 4164/r**4 + 5184/r**5 + 2592/r**6) )**(3/9)