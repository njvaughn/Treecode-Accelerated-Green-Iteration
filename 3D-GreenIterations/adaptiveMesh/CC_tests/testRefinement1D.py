'''
Created on Jul 24, 2018

@author: nathanvaughn
'''

import numpy as np
from scipy.special import erf
import itertools
import matplotlib.pyplot as plt

from meshUtilities import ChebyshevPoints, unscaledWeights, weights, ChebDerivative, interpolator1Dchebyshev

def Phi(x):
    return np.exp(-np.abs(x))

def V(x):
    return -1/np.sqrt(np.abs(x))

 

class Cell(object):
    '''
    1D Cells.  Minimal structure to test refinement
    '''
    def __init__(self, xmin, xmax, px, depth, masterList=None):
        self.xmin = xmin
        self.xmax = xmax
        self.xmid = (xmax+xmin)/2
        self.px = px
    
        self.x = ChebyshevPoints(xmin, xmax, px)
        self.w = weights(xmin, xmax, order)
        self.phi = Phi(self.x)
        self.gradPhi = ChebDerivative(xmin,xmax,px,self.phi)
        self.v = V(self.x)
        
        self.leaf = True
        self.masterList = masterList
        
        self.depth = depth
        
    def computeKinetic(self):
        self.Kinetic = -np.dot(self.gradPhi,self.gradPhi*self.w)
        return self.Kinetic
#         return np.dot(self.phi,self.phi*self.w)

    def computeKineticError(self):
        if not hasattr(self, 'Kinetic'):
            self.computeKinetic()
        if self.xmid < 0:
            trueKinetic = -( np.exp(-2*abs(self.xmax)) - np.exp(-2*abs(self.xmin)) )/2
        else:
            trueKinetic = ( np.exp(-2*abs(self.xmax)) - np.exp(-2*abs(self.xmin)) )/2

        self.kineticError = (self.Kinetic - trueKinetic)*(self.xmax-self.xmin)
    
    def computePotential(self):
        self.Potential = np.dot(self.phi,self.phi*self.v*self.w)
        return self.Potential
    
    def computePotentialError(self):
        if not hasattr(self, 'Potential'):
            self.computePotential()
        
        truePotential = -np.sqrt(np.pi/2)*abs( ( erf(np.sqrt(2*abs(self.xmax))) - erf(np.sqrt(2*abs(self.xmin))) ) )
#         print(truePotential)
        self.potentialError = (self.Potential - truePotential)*(self.xmax-self.xmin)
    
    def divideIntoTwoChildren(self):
        self.leaf=False
        children = np.empty(2,dtype=object)
        
        children[0] = Cell(self.xmin,self.xmid,self.px,self.depth+1,self.masterList)
        children[1] = Cell(self.xmid,self.xmax,self.px,self.depth+1,self.masterList)
        
        self.children = children
        children[0].parent = self
        children[1].parent = self
        
        if self.masterList is not None:
            self.masterList = np.append(self.masterList, children[0]) 
            self.masterList = np.append(self.masterList, children[1]) 
#             print(self.masterList)   
            
    def refine_scheme1(self, depth=4):
        '''
        Refine if you are the closest child to the cusp.  
        '''
        if self.depth<depth:
            xmid = (self.xmax + self.xmin)/2
            dx = self.xmax-self.xmin
            if abs(xmid) < dx: # you are touching the cusp
#                 print('Cell at depth %i centered at %f.4 with width %.4f gets divided.' %(self.depth,xmid,dx))
                self.divideIntoTwoChildren()
#                 print('Parent:   ', self)
#                 print('Parent list: ', self.masterList)
#                 print('Children: ', self.children[0], self.children[1])
                
                for child in self.children:
                    child.refine_scheme1(depth)
                    
    def setInterpolator(self, interpolant):
        return interpolator1Dchebyshev(self.x, interpolant)
         


def computeEnergy(node, Kinetic = 0.0, Potential = 0.0):
    if node.leaf==True:
        Kinetic += node.computeKinetic()
        Potential += node.computePotential()
        
        node.computeKineticError()
        node.computePotentialError()
        
#         print('Leaf Cell at depth %i centered at %f has Kinetic %e and Potential %e.' %(node.depth,node.xmid,node.computeKinetic(),node.computePotential()))
#         print('Kinetic error: %e, Potential error: %e' %(node.kineticError, node.potentialError), '\n')
        return [Kinetic, Potential]

    else:
        for child in node.children:
            Kinetic,Potential = computeEnergy(child, Kinetic, Potential)
    
    return [Kinetic, Potential]

def computeCellErrors(node, midpoints=[], kineticErrors=[], potentialErrors=[]):
    
    if node.leaf==True:
        if not hasattr(node, 'kineticError'):
            node.computeKineticError()
        if not hasattr(node, 'potentialError'):
            node.computePotentialError()
        
        midpoints.append(node.xmid)
        kineticErrors.append(node.kineticError)
        potentialErrors.append(node.potentialError)
        
        return midpoints, kineticErrors, potentialErrors
    
    else:
        for child in node.children:
            midpoints, kineticErrors, potentialErrors = computeCellErrors(child,midpoints,kineticErrors,potentialErrors)
           
    return midpoints, kineticErrors, potentialErrors

def plotCellErrors(midpoints,kineticErrors,potentialErrors, order, maxDepth):
    plt.figure()
    plt.semilogy(midpoints,np.abs(kineticErrors),'go',label='Kinetic Errors')       
    plt.semilogy(midpoints,np.abs(potentialErrors),'bo',label='Potential Errors')
    plt.legend()
    plt.title('Cell-wise Errors using Cell Order %i with max Depth %i' %(order,maxDepth))
    plt.xlabel('Domain')
    plt.ylabel('Energy Error')
    plt.show() 
    
def interpolateOnEachCell(node, testpoints=[], interpolationValuesPhi=[], interpolationValuesPhiVPhi=[], interpolationValuesGradPhiSq=[], ptspercell = 10):
    if node.leaf==True:
        cellTestPoints = np.linspace(node.xmin,node.xmax, ptspercell)
        PhiInterpolator = node.setInterpolator(node.phi)
        PhiVPhiInterpolator = node.setInterpolator(node.phi*node.phi*node.v)
        GradPhiSqInterpolator = node.setInterpolator(node.gradPhi*node.gradPhi)
        
        interpolationValuesPhi = np.append(interpolationValuesPhi,PhiInterpolator(cellTestPoints)) 
        interpolationValuesPhiVPhi = np.append(interpolationValuesPhiVPhi,PhiVPhiInterpolator(cellTestPoints)) 
        interpolationValuesGradPhiSq = np.append(interpolationValuesGradPhiSq,GradPhiSqInterpolator(cellTestPoints)) 

        testpoints = np.append(testpoints, cellTestPoints)
        
        return testpoints, interpolationValuesPhi, interpolationValuesPhiVPhi, interpolationValuesGradPhiSq
    else:
        for child in node.children:
            testpoints, interpolationValuesPhi, interpolationValuesPhiVPhi, interpolationValuesGradPhiSq = interpolateOnEachCell(
                child,testpoints, interpolationValuesPhi, interpolationValuesPhiVPhi, interpolationValuesGradPhiSq,ptspercell)
        return testpoints, interpolationValuesPhi, interpolationValuesPhiVPhi, interpolationValuesGradPhiSq
    
def plotInterpolations(testpoints, interpolationValuesPhi, interpolationValuesPhiVPhi, interpolationValuesGradPhiSq, ptspercell):
    f1, (ax1, ax2, ax3) = plt.subplots(3, 1,figsize=(12,6))
    numcells = len(testpoints%ptspercell)
    ax1.plot(testpoints,Phi(testpoints),'k-')
    ax2.plot(testpoints,Phi(testpoints)*Phi(testpoints)*V(testpoints),'k-')
    ax3.plot(testpoints,Phi(testpoints)*Phi(testpoints),'k-')
    
    for i in range(numcells):
        cellPoints = testpoints[i*ptspercell:(i+1)*ptspercell]
        ax1.plot(cellPoints,interpolationValuesPhi[i*ptspercell:(i+1)*ptspercell],'--')
        ax2.plot(cellPoints,interpolationValuesPhiVPhi[i*ptspercell:(i+1)*ptspercell],'--')
        ax3.plot(cellPoints,interpolationValuesGradPhiSq[i*ptspercell:(i+1)*ptspercell],'--')

    ax1.set_title('Interpolated Phi')
    
    ax2.set_title('Interpolated Phi*V*Phi')
    
    ax3.set_title('Interpolated Grad Phi Squared')
    
    plt.tight_layout()
    plt.show()
    
 
def computePminus1Derivatives(node, midpoints=[], PhiDerivatives=[], PhiVPhiDerivatives=[], GradPhiSqDerivatives=[]):  
    if node.leaf == True:
#         print('Cell at depth %i centered at %f.4.' %(node.depth,node.xmid))
#         temp = node.phi**2*node.v
        tempPhi = node.phi
        tempPhiVPhi = node.phi**2*node.v
        tempGradPhiSq = node.gradPhi**2
        for i in range(node.px-1):
            tempPhi = ChebDerivative(node.xmin, node.xmax, node.px, tempPhi) 
            tempPhiVPhi = ChebDerivative(node.xmin, node.xmax, node.px, tempPhiVPhi) 
            tempGradPhiSq = ChebDerivative(node.xmin, node.xmax, node.px, tempGradPhiSq) 
            if i==node.px-2:
#                 print('%i derivative' %(i+1))
#                 print(temp)
#                 print()
                midpoints.append(node.xmid)

#                 PhiDerivatives.append(max(abs(tempPhi)) * (node.xmax-node.xmin))
#                 PhiVPhiDerivatives.append(max(abs(tempPhiVPhi)) * (node.xmax-node.xmin))
#                 GradPhiSqDerivatives.append(max(abs(tempGradPhiSq)) * (node.xmax-node.xmin))

                PhiDerivatives.append(max(abs(tempPhi)) )
                PhiVPhiDerivatives.append(max(abs(tempPhiVPhi)))
                GradPhiSqDerivatives.append(max(abs(tempGradPhiSq)))
        return midpoints, PhiDerivatives, PhiVPhiDerivatives, GradPhiSqDerivatives
    
    else:
        for child in node.children:
            midpoints, PhiDerivatives, PhiVPhiDerivatives, GradPhiSqDerivatives = computePminus1Derivatives(child)
            
        return midpoints, PhiDerivatives, PhiVPhiDerivatives, GradPhiSqDerivatives

def plotMaxDerivatives(midpoints, PhiDerivatives, PhiVPhiDerivatives, GradPhiSqDerivatives, order):
    plt.figure()
    plt.semilogy(midpoints,PhiDerivatives,'go', label='Phi')       
    plt.semilogy(midpoints,PhiVPhiDerivatives,'bo', label='PhiVPhi')       
    plt.semilogy(midpoints,GradPhiSqDerivatives,'ro', label='GradPhiSq')       
    plt.title('Maximum of %ith derivative in each order %i cell' %(order-1, order))
    plt.legend()
    plt.show()
    
    
if __name__ == '__main__':
    xmin = -10
    xmid =   0
    xmax =  10
    order = 8
    maxDepth=2
    masterList = np.empty(0, dtype=object)

    root = Cell(xmin,xmax,order,0)
    root.refine_scheme1(depth=maxDepth)
    print('tree built')
#     Kinetic, Potential = computeEnergy(root, Kinetic=0.0, Potential=0.0)
#     print('Kinetic Error:   %e' %(Kinetic + 1.0) )
#     print('Potential Error: %e' %(Potential+np.sqrt(2*np.pi)) )
#     midpoints, kineticErrors, potentialErrors = computeCellErrors(root)
#     plotCellErrors(midpoints, kineticErrors, potentialErrors, order, maxDepth)
    
#     ptspercell=200
#     testpoints, interpolationValuesPhi, interpolationValuesPhiVPhi, interpolationValuesGradPhiSq = interpolateOnEachCell(root, ptspercell=ptspercell)
#     plotInterpolations(testpoints, interpolationValuesPhi, interpolationValuesPhiVPhi, interpolationValuesGradPhiSq, ptspercell=ptspercell)
    
    
    midpoints, PhiDerivatives, PhiVPhiDerivatives, GradPhiSqDerivatives = computePminus1Derivatives(root)
    
    plotMaxDerivatives(midpoints, PhiDerivatives, PhiVPhiDerivatives, GradPhiSqDerivatives,order)

    

    
    