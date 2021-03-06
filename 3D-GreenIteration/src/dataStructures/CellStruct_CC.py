'''
CellStruct_CC.py

Trees are composed of cells.
Cells are composed of gridpoints, in this case using the clenshaw curtis quadrature.
Cell structures are used in the initialization, but not throughout the calculation,
although many features would benefit from retaining the cell structure.

Created on Mar 5, 2018

@author: nathanvaughn
'''
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import sph_harm
import itertools
import bisect

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size() 

from mpiUtilities import rprint
from meshUtilities import meshDensity, weights3DFirstKind, unscaledWeightsFirstKind, weights3DSecondKind,unscaledWeightsSecondKind, \
  ChebGradient3D, ChebyshevPointsFirstKind,ChebyshevPointsSecondKind,computeDerivativeMatrix,\
    computeLaplacianMatrix, ChebLaplacian3D, sumChebyshevCoefficicentsGreaterThanOrderQ, sumChebyshevCoefficicentsEachGreaterThanOrderQ, sumChebyshevCoefficicentsAnyGreaterThanOrderQ,\
    sumChebyshevCoefficicentsGreaterThanOrderQZeroZero
from GridpointStruct import GridPoint
# from mpmath import psi

ThreeByThreeByThree = [element for element in itertools.product(range(3),range(3),range(3))]
TwoByTwoByTwo = [element for element in itertools.product(range(2),range(2),range(2))]
TwoByTwo = [element for element in itertools.product(range(2),range(2))]
FiveByFiveByFive = [element for element in itertools.product(range(5),range(5),range(5))]
 
 
 
class Cell(object):
    '''
    Cell object.  Cells are composed of gridpoint objects.  Trees are composed of Cells (as the nodes).
    '''
    
    """
    INITIALIZATION FUNCTIONS
    """
    def __init__(self, kind, xmin, xmax, px, ymin, ymax, py, zmin, zmax, pz, gridpoints=None, fine_p=None, fine_gridpoints=None, densityPoints=None, tree=None, atomAtCorner=False):
        '''
        Cell Constructor.  Cell composed of gridpoint objects
        '''
        if tree is not None:
            self.tree = tree
        self.px = px
        self.py = py
        self.pz = pz
        self.numCoarsePoints=(self.px+1)*(self.py+1)*(self.pz+1)
        self.pxf=fine_p
        self.pyf=fine_p
        self.pzf=fine_p
        self.fineMesh = False
        self.pxd = self.px+1 # Points for the density
        self.pyd = self.py+1 # Points for the density secondary mesh
        self.pzd = self.pz+1
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.gridpoints = gridpoints
        self.fine_gridpoints = fine_gridpoints
        self.densityPoints = densityPoints
        self.leaf = True
        self.atomAtCorner = atomAtCorner
        if kind=='first':
#             rprint(rank, "CELL IS FIRST KIND")
            W = unscaledWeightsFirstKind(px)  # assumed px=py=pz
            self.unscaledW = W
            Wf = unscaledWeightsFirstKind(self.pxf)  # assumed px=py=pz
            self.w = weights3DFirstKind(xmin, xmax, px, ymin, ymax, py, zmin, zmax, pz, W)
            self.wf = weights3DFirstKind(xmin, xmax, self.pxf, ymin, ymax, self.pyf, zmin, zmax, self.pzf, Wf)
#             self.PxByPyByPz = [element for element in itertools.product(range(self.px),range(self.py),range(self.pz))]
        elif kind=='second':
            W = unscaledWeightsSecondKind(px)  # assumed px=py=pz
            self.w = weights3DSecondKind(xmin, xmax, px, ymin, ymax, py, zmin, zmax, pz, W)
#             self.PxByPyByPz = [element for element in itertools.product(range(self.px+1),range(self.py+1),range(self.pz+1))]
        else:
            rprint(rank, "Which kind of Chebyshev?")
            return
        self.PxByPyByPz = [element for element in itertools.product(range(self.px+1),range(self.py+1),range(self.pz+1))]
        self.PxfByPyfByPzf = [element for element in itertools.product(range(self.pxf+1),range(self.pyf+1),range(self.pzf+1))]
        
        self.setCellMidpointAndVolume()
        self.setNearestAtom()
        self.kind = kind
        
        
        if hasattr(self, "tree"):
#             rprint(rank, 'Cell has attribute tree')
            self.orbitalPE = np.zeros(self.tree.nOrbitals)
            self.orbitalKE = np.zeros(self.tree.nOrbitals)

    def switchKindsSecondToFirst(self):
        assert self.kind=='second'
        self.kind='first'
        rprint(rank, "CELL SWITCHED TO FIRST KIND")
        W = unscaledWeightsFirstKind(px)  # assumed px=py=pz
        self.w = weights3DFirstKind(xmin, xmax, px, ymin, ymax, py, zmin, zmax, pz, W)
        
    def switchKindsFirstSecond(self):
        assert self.kind=='first'
        self.kind='second'
        rprint(rank, "CELL SWITCHED TO SECOND KIND")
        W = unscaledWeightsSecondKind(px)  # assumed px=py=pz
        self.w = weights3DSecondKind(xmin, xmax, px, ymin, ymax, py, zmin, zmax, pz, W)
        
    def locateAtomAtCorner(self,Atom):
#         assert self.atomAtCorner==True
        if Atom.x==self.xmax:
            x='1'
        else:
            x='0'
        if Atom.y==self.ymax:
            y='1'
        else:
            y='0'
        if Atom.z==self.zmax:
            z='1'
        else:
            z='0'
            
        self.atomAtCorner=x+y+z
        rprint(rank, 'Cell %s has atom at corner %s' %(self.uniqueID, self.atomAtCorner))
        
      
    def setGridpoints(self,gridpoints):
        self.gridpoints = gridpoints
        
        if  (   (self.gridpoints[0,0,0].x    < self.xmin) or
                (self.gridpoints[-1,-1,-1].x > self.xmax) or
                (self.gridpoints[0,0,0].y    < self.ymin) or
                (self.gridpoints[-1,-1,-1].y > self.ymax) or
                (self.gridpoints[0,0,0].z    < self.zmin) or
                (self.gridpoints[-1,-1,-1].z > self.zmax)    ):
            
            rprint(rank, 'WARNING: Gridpoints arent contained within cell bounds.')
            
    def setFineGridpoints(self,fine_gridpoints):
        self.fine_gridpoints = fine_gridpoints
        
        if  (   (self.fine_gridpoints[0,0,0].x    < self.xmin) or
                (self.fine_gridpoints[-1,-1,-1].x > self.xmax) or
                (self.fine_gridpoints[0,0,0].y    < self.ymin) or
                (self.fine_gridpoints[-1,-1,-1].y > self.ymax) or
                (self.fine_gridpoints[0,0,0].z    < self.zmin) or
                (self.fine_gridpoints[-1,-1,-1].z > self.zmax)    ):
            
            rprint(rank, 'WARNING: Fine_Gridpoints arent contained within cell bounds.')
            
    
    def setDensityPoints(self,densityPoints):
        self.densityPoints = densityPoints
             
    def setCellMidpointAndVolume(self):

        self.volume = (self.xmax-self.xmin)*(self.ymax-self.ymin)*(self.zmax-self.zmin)
        self.xmid = 1/2*(self.xmin + self.xmax)
        self.ymid = 1/2*(self.ymin + self.ymax)
        self.zmid = 1/2*(self.zmin + self.zmax)
        
        if self.volume==0.0:
            rprint(rank, 'warning: cell has zero volume')
            rprint(rank, 'Dx = ', self.xmax-self.xmin)
            rprint(rank, 'xmin = ', self.xmin)
            rprint(rank, 'xmax = ', self.xmax)
            rprint(rank, 'Dy = ', self.ymax-self.ymin)
            rprint(rank, 'ymin = ', self.ymin)
            rprint(rank, 'ymax = ', self.ymax)
            rprint(rank, 'Dz = ', self.zmax-self.zmin)
            rprint(rank, 'zmin = ', self.zmin)
            rprint(rank, 'zmax = ', self.zmax)
            rprint(rank, " ")
            
        
        if abs(np.sum(self.w) - self.volume) / self.volume > 1e-10:
            try:
                rprint(rank, 'warning, cell weights dont sum to cell volume for cell ', self.uniqueID)
            except:
                rprint(rank, 'warning, cell weights dont sum to cell volume, no uniqueID')
            rprint(rank, 'Volume: ', self.volume)
            rprint(rank, 'Weights: ', self.w)

    def getAspectRatio(self):
        
        dx = (self.xmax-self.xmin)
        dy = (self.ymax-self.ymin)
        dz = (self.zmax-self.zmin)
        L = max(dx, max(dy,dz))
        l = min(dx, min(dy,dz))
                
        return L/l
    
    
    """ 
    NEAREST ATOM(S)
    """
    
    def setNearestAtom(self):
        minDistSq = np.inf
        for atom in self.tree.atoms:
            distSq = (self.xmid - atom.x)**2 + (self.ymid - atom.y)**2 + (self.zmid - atom.z)**2 
            
            if distSq < minDistSq:
                self.nearestAtom = atom
                minDistSq = distSq
    
    """
    NEIGHBOR IDENTIFICATION AN LABELING FUNCTIONS
    """ 
    def setUniqueID(self,i,j,k):
        self.uniqueID = "".join( list(self.parent.uniqueID) + list([str(i+1),str(j+1),str(k+1)]) )
        
    def setNeighborList(self):
        def getNeighbors3D(self):
            xLowXHighID = list(self.uniqueID)[::3]
            yLowYHighID = list(self.uniqueID)[1::3]
            zHighZLowID = list(self.uniqueID)[2::3]
            
            
            def recursiveDigitFlipForGreaterNeighbor(identifierList, targetIndex, noNeighborFlag):
                neighborID = np.copy(identifierList)
                if ( (targetIndex == 0) and (neighborID[targetIndex] == '2') ):
                    noNeighborFlag = True
                    return (list('This cell has no greater neighbor'), noNeighborFlag)
                if neighborID[targetIndex] == '0':
                    neighborID, noNeighborFlag = recursiveDigitFlipForGreaterNeighbor(neighborID, targetIndex-1, noNeighborFlag)
                    return (neighborID, noNeighborFlag)
                
                if neighborID[targetIndex] == '1':
                    neighborID[targetIndex] = '2'
                    return (neighborID, noNeighborFlag)
                
                if neighborID[targetIndex] == '2':
                    neighborID[targetIndex] = '1'
                    neighborID, noNeighborFlag = recursiveDigitFlipForGreaterNeighbor(neighborID, targetIndex-1, noNeighborFlag)
                    return (neighborID, noNeighborFlag)
                
            def recursiveDigitFlipForLesserNeighbor(identifierList, targetIndex, noNeighborFlag):
                neighborID = np.copy(identifierList)
                if ( (targetIndex == 0) and (neighborID[targetIndex]) ) == '1':
                    noNeighborFlag = True
                    return (list('This cell has no lesser neighbor'), noNeighborFlag)
                if neighborID[targetIndex] == '0':
                    neighborID, noNeighborFlag = recursiveDigitFlipForLesserNeighbor(neighborID, targetIndex-1, noNeighborFlag)
                    return (neighborID, noNeighborFlag)
                
                if neighborID[targetIndex] == '2':
                    neighborID[targetIndex] = '1'
                    return (neighborID, noNeighborFlag)
                
                if neighborID[targetIndex] == '1':
                    neighborID[targetIndex] = '2'
                    neighborID, noNeighborFlag = recursiveDigitFlipForLesserNeighbor(neighborID, targetIndex-1, noNeighborFlag)
                    return (neighborID, noNeighborFlag)
        
    
            xHighID, xHighIDNoNeighborFlag = recursiveDigitFlipForGreaterNeighbor(xLowXHighID, len(xLowXHighID)-1, noNeighborFlag=False)
            xLowID, xLowIDNoNeighborFlag = recursiveDigitFlipForLesserNeighbor(xLowXHighID, len(xLowXHighID)-1, noNeighborFlag=False)
            zHighID, zHighIDNoNeighborFlag = recursiveDigitFlipForGreaterNeighbor(zHighZLowID, len(zHighZLowID)-1, noNeighborFlag=False)
            zLowID, zLowIDNoNeighborFlag = recursiveDigitFlipForLesserNeighbor(zHighZLowID, len(zHighZLowID)-1, noNeighborFlag=False)
            yHighID, yHighIDNoNeighborFlag = recursiveDigitFlipForGreaterNeighbor(yLowYHighID, len(yLowYHighID)-1, noNeighborFlag=False)
            yLowID, yLowIDNoNeighborFlag = recursiveDigitFlipForLesserNeighbor(yLowYHighID, len(yLowYHighID)-1, noNeighborFlag=False)
        
        
            xHighNeighbor = []
            xLowNeighbor = []
            zHighNeighbor = []
            zLowNeighbor = []
            yHighNeighbor=[]
            yLowNeighbor=[]
        
            for i in range(int(len(list(self.uniqueID))/3)):
                
                zHighNeighbor.append(xLowXHighID[i])
                zHighNeighbor.append(yLowYHighID[i])
                zHighNeighbor.append(zHighID[i])
                zLowNeighbor.append(xLowXHighID[i])
                zLowNeighbor.append(yLowYHighID[i])
                zLowNeighbor.append(zLowID[i])
                
                xHighNeighbor.append(xHighID[i])
                xHighNeighbor.append(yLowYHighID[i])
                xHighNeighbor.append(zHighZLowID[i])
                xLowNeighbor.append(xLowID[i])
                xLowNeighbor.append(yLowYHighID[i])
                xLowNeighbor.append(zHighZLowID[i])
                
                yHighNeighbor.append(xLowXHighID[i])
                yHighNeighbor.append(yHighID[i])
                yHighNeighbor.append(zHighZLowID[i])
                yLowNeighbor.append(xLowXHighID[i])
                yLowNeighbor.append(yLowID[i])
                yLowNeighbor.append(zHighZLowID[i])
            

            neighborList = []
            if xLowIDNoNeighborFlag == True: pass
            else: neighborList.append(['xLow', "".join(xLowNeighbor)])
            if xHighIDNoNeighborFlag == True: pass 
            else: neighborList.append(['xHigh',"".join(xHighNeighbor)])
            if yLowIDNoNeighborFlag == True: pass 
            else: neighborList.append(['yLow',"".join(yLowNeighbor)])
            if yHighIDNoNeighborFlag == True: pass
            else: neighborList.append(['yHigh',"".join(yHighNeighbor)])
            if zLowIDNoNeighborFlag == True: pass
            else: neighborList.append(['zLow',"".join(zLowNeighbor)])
            if zHighIDNoNeighborFlag == True: pass 
            else: neighborList.append(['zHigh',"".join(zHighNeighbor)])
        
 
            return neighborList

        self.neighbors = getNeighbors3D(self)
     
    def checkIfChildrenInSaveList(self, saveList): 
        
#         rprint(rank, 'Enertering cell.checkIfChildrenInSaveList...')
#         rprint(rank, 'saveList = ', saveList[0:10])
        
        
        def findStr(a, x):
#             'Locate the leftmost value exactly equal to x'
            i = bisect.bisect_left(a, x)
            if i != len(a) and a[i].startswith(x) == True:
                return i
            else:
                return-1
#             raise ValueError
        
        
        self.divideFlag=False
#         rprint(rank, 'calling bisect')
#         i = bisect.bisect_left(saveList, self.uniqueID+'111')
#         rprint(rank, 'completed bisect')
        i = findStr(saveList, self.uniqueID+'111')  # search for x child.  Could also search for self ID, then see if next thing in list is longer.
#         i = findStr(saveList, self.uniqueID)  # search for x child.  Could also search for self ID, then see if next thing in list is longer.
        
#         if ( ( i<(len(saveList)-1) )  and (i!=-1) ):
#             if len(saveList[i+1]) > len(saveList[i]):
#                 if saveList[i+1].startswith(self.uniqueID):
#                     self.divideFlag=True

        if i!=-1:
            self.divideFlag=True
#         rprint(rank, 'Returning: divideFlag = ', self.divideFlag)
        return
                
    def fillInNeighbors(self, gridpoints): 
        '''
        For all 6 possible neighbors, check if they occur in the cell's neighbor list, meaning that neighbor *could* exist.  Cells along boundaries will not have all 6.
        If the cell could exist, check if it does already exist.  This depends on whether other regions of the domain have divided this far or not.  
        If the neighbor *DOES* exist, and IF the neighbor has already been fully created, meaning its gridpoints are defined, then copy the appropriate face
        of gridpoints.  Notice, sibling cells from the same parent will exist, but won't yet have gridpoints set up.  
        :param gridpoints: input the sub-array of gridpoints that will be used to construct the child.
        Modify this sub-array of gridpoints (if neighbors exist), then output the sub-array. 
        '''


        def find(a, x):
            'Locate the leftmost value exactly equal to x'
            i = bisect.bisect_left(a, [x,])
            if i != len(a) and a[i][0] == x:
                return i
            raise ValueError
        
        
        printNeighborResults = False
        if printNeighborResults == True: rprint(rank, '\nTarget Cell ID      ', self.uniqueID)
    
        '''fill in any gridpoints coming from X neighbors'''
        try: 
            xLowID =   [element[1] for element in self.neighbors if element[0] == 'xLow'][0]
            xLowCell = self.tree.masterList[ find(self.tree.masterList, xLowID) ][1]
#             xLowID =   [element[1] for element in self.neighbors if element[0] == 'xLow'][0]
#             xLowCell = [element[1] for element in self.tree.masterList if str(element[0]) == [element[1] for element in self.neighbors if element[0] == 'xLow'][0]][0]
            if hasattr(xLowCell, 'gridpoints'): gridpoints[0,:,:] = xLowCell.gridpoints[2,:,:] # this is failing
            if printNeighborResults == True: rprint(rank, 'found xLowCell:   ', xLowCell, 'whose ID is ', xLowCell.uniqueID)
        except: pass
        try: 
            xHighID =  [element[1] for element in self.neighbors if element[0] == 'xHigh'][0]
            xHighCell = self.tree.masterList[ find(self.tree.masterList, xHighID) ][1]
#             xHighCell = [element[1] for element in self.tree.masterList if str(element[0]) == [element[1] for element in self.neighbors if element[0] == 'xHigh'][0]][0]
            if hasattr(xHighCell, 'gridpoints'): gridpoints[2,:,:] = xHighCell.gridpoints[0,:,:] # this is failing
            if printNeighborResults == True: rprint(rank, 'found xHighCell:  ', xHighCell, 'whose ID is ', xHighCell.uniqueID)
        except: pass
        
        '''fill in any gridpoints coming from Y neighbors'''
        try: 
            yLowID =     [element[1] for element in self.neighbors if element[0] == 'yLow'][0]
            yLowCell = self.tree.masterList[ find(self.tree.masterList, yLowID) ][1]
#             yLowCell = [element[1] for element in self.tree.masterList if str(element[0]) == [element[1] for element in self.neighbors if element[0] == 'yLow'][0]][0]
            if hasattr(yLowCell, 'gridpoints'): gridpoints[:,0,:] = yLowCell.gridpoints[:,2,:] # this is failing
            if printNeighborResults == True: rprint(rank, 'found yLowCell:     ', yLowCell, 'whose ID is ', yLowCell.uniqueID)
        except: pass
        try: 
            yHighID =    [element[1] for element in self.neighbors if element[0] == 'yHigh'][0]
            yHighCell = self.tree.masterList[ find(self.tree.masterList, yHighID) ][1]
#             yHighCell = [element[1] for element in self.tree.masterList if str(element[0]) == [element[1] for element in self.neighbors if element[0] == 'yHigh'][0]][0]
            if hasattr(yHighCell, 'gridpoints'): gridpoints[:,2,:] = yHighCell.gridpoints[:,0,:] # this is failing
            if printNeighborResults == True: rprint(rank, 'found yHighCell:    ', yHighCell, 'whose ID is ', yHighCell.uniqueID)
        except: pass
        
        '''fill in any gridpoints coming from Z neighbors'''
        try: 
            zLowID = [element[1] for element in self.neighbors if element[0] == 'zLow'][0]
            zLowCell = self.tree.masterList[ find(self.tree.masterList, zLowID) ][1]
#             zLowCell = [element[1] for element in self.tree.masterList if str(element[0]) == [element[1] for element in self.neighbors if element[0] == 'zLow'][0]][0]
            if hasattr(zLowCell, 'gridpoints'): gridpoints[:,:,0] = zLowCell.gridpoints[:,:,2] # this is failing
            if printNeighborResults == True: rprint(rank, 'found zLowCell: ', zLowCell, 'whose ID is ', zLowCell.uniqueID)
        except: pass 
        try: 
            zHighID = [element[1] for element in self.neighbors if element[0] == 'zHigh'][0]
            zHighCell = self.tree.masterList[ find(self.tree.masterList, zHighID) ][1]
#             zHighCell = [element[1] for element in self.tree.masterList if str(element[0]) == [element[1] for element in self.neighbors if element[0] == 'zHigh'][0]][0]
            if hasattr(zHighCell, 'gridpoints'): gridpoints[:,:,2] = zHighCell.gridpoints[:,:,0] # this is failing
            if printNeighborResults == True: rprint(rank, 'found zHighCell:    ', zHighCell, 'whose ID is ', zHighCell.uniqueID)
        except: pass  
        
        ''' return the (potentially) modified sub-array of gridpoints'''
        return gridpoints


    """
    REFINEMENT CHECKING FUNCTIONS
    """
    def getPhiVariation(self):
        minPhi = self.gridpoints[0,0,0].phi
        maxPhi = self.gridpoints[0,0,0].phi
        for i,j,k in self.PxByPyByPz:
            if self.gridpoints[i,j,k].phi < minPhi: minPhi = self.gridpoints[i,j,k].phi
            if self.gridpoints[i,j,k].phi > maxPhi: maxPhi = self.gridpoints[i,j,k].phi
        
        self.psiVariation = maxPhi - minPhi
    
    def getRhoVariation(self):
        minRho = self.gridpoints[0,0,0].rho
        maxRho = self.gridpoints[0,0,0].rho
        for i,j,k in self.PxByPyByPz:
            if self.gridpoints[i,j,k].rho < minRho: minRho = self.gridpoints[i,j,k].rho
            if self.gridpoints[i,j,k].rho > maxRho: maxRho = self.gridpoints[i,j,k].rho
        
        return maxRho - minRho
    
    def getWavefunctionVariation(self,m):
        minPsi = self.gridpoints[0,0,0].phi[m]
        maxPsi = self.gridpoints[0,0,0].phi[m]
        maxAbsPsi = 0.0
        for i,j,k in self.PxByPyByPz:
            if self.gridpoints[i,j,k].phi[m] < minPsi: minPsi = self.gridpoints[i,j,k].phi[m]
            if self.gridpoints[i,j,k].phi[m] > maxPsi: maxPsi = self.gridpoints[i,j,k].phi[m]
#             if abs(self.gridpoints[i,j,k].phi[m]) > maxAbsPsi: maxAbsPsi = self.gridpoints[i,j,k].phi[m]
        
        return (maxPsi - minPsi)

    def getWavefunctionIntegral(self,m):
        absIntegral = 0.0
        for i,j,k in self.PxByPyByPz:
            absIntegral += abs( self.gridpoints[i,j,k].phi[m]*self.w[i,j,k] )
        return absIntegral
    
    
#         return (maxPsi - minPsi)/maxAbsPsi
#         return min( (maxPsi - minPsi)/maxAbsPsi, (maxPsi - minPsi))
        
    def getTestFunctionVariation(self):
        minValue = self.gridpoints[0,0,0].testFunctionValue
        maxValue = self.gridpoints[0,0,0].testFunctionValue
        for i,j,k in self.PxByPyByPz:
            if self.gridpoints[i,j,k].testFunctionValue < minValue: minValue = self.gridpoints[i,j,k].testFunctionValue
            if self.gridpoints[i,j,k].testFunctionValue > maxValue: maxValue = self.gridpoints[i,j,k].testFunctionValue
        
        self.testFunctionVariation = maxValue - minValue
               
    def getPhiVPhiVariation(self):
        minPhiVPhi = self.gridpoints[0,0,0].phi
        maxPhiVPhi = self.gridpoints[0,0,0].phi
        for i,j,k in self.PxByPyByPz:
            if self.gridpoints[i,j,k].phi < minPhiVPhi: minPhiVPhi = self.gridpoints[i,j,k].phi
            if self.gridpoints[i,j,k].phi > maxPhiVPhi: maxPhiVPhi = self.gridpoints[i,j,k].phi
         
        self.psiVariation = maxPhiVPhi - minPhiVPhi
        
    def getPhiGradVariation(self):
        
        phi = np.empty((3,3,3))
        for i,j,k in self.PxByPyByPz:
            phi[i,j,k] = self.gridpoints[i,j,k].phi
        gradient = np.gradient(phi, self.dx, self.dy, self.dz, edge_order=2)
        
        maxXgrad = np.max(gradient[0])
        maxYgrad = np.max(gradient[1])
        maxZgrad = np.max(gradient[2])
        
        minXgrad = np.min(gradient[0])
        minYgrad = np.min(gradient[1])
        minZgrad = np.min(gradient[2])
         
        self.phiGradVariation = max( (maxXgrad-minXgrad), max( (maxYgrad-minYgrad), (maxZgrad-minZgrad)) )
            
    
    
#     def checkIntegralsAndDivideIfNeeded(self, divideTolerance):
#         self.computePotential()
#         parentIntegral = self.PE
#         childrenIntegral = 0.0
#         self.divideInto8()
#         for i,j,k in TwoByTwoByTwo:
#             self.children[i,j,k].computePotential()
#             childrenIntegral += self.children[i,j,k].PE
#             
#         if abs(parentIntegral-childrenIntegral) > divideTolerance:
#             self.divideFlag = True
#         else:
#             self.children=None
#             self.leaf = True
#             self.divideFlag = False
    
    def checkIfCellShouldDivide(self, divideTolerance):
        '''
        Perform midpoint method for integral(testFunction) for parent cell
        If parent integral differs from the sum of children integrals over the test function
        then set divideFlag to true.
        :param divideTolerance:
        '''

        self.computePotential()
        parentIntegral = self.PE
        childrenIntegral = 0.0
        children = self.divideButJustReturnChildren()
        for i,j,k in TwoByTwoByTwo:
            children[i,j,k].computePotential()
            childrenIntegral += children[i,j,k].PE
            
        if abs((parentIntegral-childrenIntegral)/parentIntegral) > divideTolerance:
            self.divideFlag = True
        else:
            self.divideFlag = False
    
    def checkIfCellShouldDivideTwoConditions(self, divideTolerance1, divideTolerance2):
        '''
        Check two divideInto8 conditions.
        '''

        self.divideFlag = False # initialize

        # check variation in phi
#         self.getPhiVariation()
#         if self.psiVariation > divideTolerance1:
#             self.divideFlag = True
            
        # check variation in gradient of phi
        self.getPhiGradVariation()
        if self.phiGradVariation > divideTolerance1:
            self.divideFlag = True
            
#         mid = self.gridpoints[1,1,1]
#         r = np.sqrt(mid.x**2 + mid.y**2 + mid.z**2)
#         VolumeDivR3 = self.volume/r**3
#         if VolumeDivR3 > divideTolerance2:
#             self.divideFlag = True
            
#         mid = self.gridpoints[1,1,1]
# #         Rsq = mid.x**2 + mid.y**2 + mid.z**2
# #         VolumeDivRsq = self.volume/Rsq
# #         if VolumeDivRsq > divideTolerance2:
# #             self.divideFlag = True
#         PsiSqTimesVolume = mid.phi**2*self.volume
#         if PsiSqTimesVolume > divideTolerance2:
#             self.divideFlag = True
        
    
                
        self.gridpoints[1,1,1].setTestFunctionValue()
        parentIntegral = self.gridpoints[1,1,1].testFunctionValue*self.volume
#          
        childrenIntegral = 0.0
        xmids = np.array([(3*self.xmin+self.xmax)/4, (self.xmin+3*self.xmax)/4])
        ymids = np.array([(3*self.ymin+self.ymax)/4, (self.ymin+3*self.ymax)/4])
        zmids = np.array([(3*self.zmin+self.zmax)/4, (self.zmin+3*self.zmax)/4])
        for i,j,k in TwoByTwoByTwo:
            tempChild = GridPoint(xmids[i],ymids[j],zmids[k], self.tree.gaugeShift, self.tree.atoms, self.tree.coreRepresentation, self.tree.nOrbitals)
            tempChild.setTestFunctionValue()
            childrenIntegral += tempChild.testFunctionValue*(self.volume/8)
          
        if abs(parentIntegral-childrenIntegral) > divideTolerance2:
            self.divideFlag = True

    def checkIfAboveMeshDensity(self,divideParameter,divideCriterion):
        self.divideFlag = False
        for atom in self.tree.atoms:

            r = np.sqrt( (self.xmid-atom.x)**2 + (self.ymid-atom.y)**2 + (self.zmid-atom.z)**2 )
            if 1/self.volume < meshDensity(r,divideParameter,divideCriterion):
                self.divideFlag=True
    
    def compareMeshDensityToDensity(self,divideParameter):
        self.divideFlag = False
        for atom in self.tree.atoms:

            r = np.sqrt( (self.xmid-atom.x)**2 + (self.ymid-atom.y)**2 + (self.zmid-atom.z)**2 )
            if 1/self.volume < divideParameter*np.sqrt( atom.interpolators['density'](r) ):
                self.divideFlag=True
                
            if 1/self.volume < 10*divideParameter*atom.interpolators['density'](r):
                self.divideFlag=True
     
    def wavefunctionVariationAtCorners_Vext(self):   
        
        xmm = [self.xmin, self.xmax]
        ymm = [self.ymin, self.ymax]
        zmm = [self.zmin, self.zmax] 
        xm = (xmm[0]+xmm[1])/2
        ym = (ymm[0]+ymm[1])/2
        zm = (zmm[0]+zmm[1])/2       
        
        aufbauList = ['10',                                     # n+ell = 1
                      '20',                                     # n+ell = 2
                      '21', '30',                               # n+ell = 3
                      '31', '40', 
                      '32', '41', '50'
                      '42', '51', '60'
                      '43', '52', '61', '70']

        orbitalIndex=0
        
        maxVariation = 0.0
        maxRelDensityVariation = 0.0
        maxPsiVextVariation = 0.0
        maxSqVariation = 0.0
        maxAbsIntegral = 0.0
        sqVariationCause=-2
        densityIntegralCause=-2
        
        VextVariation=0.0
        VextVariationCause = -1
        
        VextIntegral = 0.0
        VextIntegralCause=-1
        
        
        ### Compute Density terms based on all atoms
        
        densityIntegral = 0.0
        sqrtDensityIntegral = 0.0
        for i,j,k in TwoByTwoByTwo:
            Vext = 0.0
            density=0.0
            for atom in self.tree.atoms:
                xm = (xmm[0]+xmm[1])/2
                Rsq =  (xm-atom.x)**2 + (ym-atom.y)**2 + (zm-atom.z)**2  # distance between atom and cell midpoint
                dx = xmm[i]-atom.x
                dy = ymm[j]-atom.y
                dz = zmm[k]-atom.z
                r = np.sqrt( dx**2 + dy**2 + dz**2 )
                if r==0:  # nudge the point 10% in towrads the cell center, just to avoid 1/0 cases
                    dx = 0.99*(xmm[i]-atom.x) + 0.01*(xmm[(i+1)%2]-atom.x)
                    dy = 0.99*(ymm[j]-atom.y) + 0.01*(ymm[(j+1)%2]-atom.y)
                    dz = 0.99*(zmm[k]-atom.z) + 0.01*(zmm[(k+1)%2]-atom.z)
                    
                    xtemp = 0.99*(xmm[i]) + 0.01*(xmm[(i+1)%2])
                    ytemp = 0.99*(ymm[j]) + 0.01*(ymm[(j+1)%2])
                    ztemp = 0.99*(zmm[k]) + 0.01*(zmm[(k+1)%2])
                    r = np.sqrt( dx**2 + dy**2 + dz**2 )
                
                if Rsq < 0.1: Vext += atom.V(xmm[i],ymm[j],zmm[k])  # only increment Vext if cell is with R=2 ball of the atom
                density +=  atom.interpolators['density'](r)
                
            densityIntegral += density/8*self.volume
            VextIntegral += np.abs(Vext)/8*self.volume
            sqrtDensityIntegral += np.sqrt(density)/8*self.volume
             
            if ( (i==0) and (j==0) and (k==0)): 
                maxDensity = density
                minDensity = density
                maxVext = Vext
                minVext = Vext
            else:
                if density>maxDensity: maxDensity = density
                if density<minDensity: minDensity = density
                
                if Vext > maxVext: maxVext = Vext
                if Vext < minVext: minVext = Vext
             
            relDensityVariation = (maxDensity-minDensity)/maxDensity
            if relDensityVariation>maxRelDensityVariation:
                maxRelDensityVariation = relDensityVariation
                relDensityVariationCause = -1
            
            VextVariation = maxVext-minVext
            
    
#         for atom in self.tree.atoms:
        atom = self.nearestAtom
#         nAtomicOrbitals = atom.nAtomicOrbitals
        if atom.atomicNumber <= 4:
            nAtomicOrbitals = 2
        elif atom.atomicNumber <= 10:
            nAtomicOrbitals = 5
        elif atom.atomicNumber <= 12:
            nAtomicOrbitals = 6
        elif atom.atomicNumber <= 18:
            nAtomicOrbitals = 9
        elif atom.atomicNumber <= 20:
            nAtomicOrbitals = 10
        else:
            rprint(rank, 'Atom with atomic number %i.  How many wavefunctions should be used in mesh refinement scheme?' %atom.nAtomicOrbitals)
        
        
       
        singleAtomOrbitalCount=0
        for nell in aufbauList:
            
            if singleAtomOrbitalCount< nAtomicOrbitals:  
                n = int(nell[0])
                ell = int(nell[1])
                psiID = 'psi'+str(n)+str(ell)
                for m in range(-ell,ell+1):
                    
                    
                    absIntegral = 0.0

                    for i,j,k in TwoByTwoByTwo:
                        dx = xmm[i]-atom.x
                        dy = ymm[j]-atom.y
                        dz = zmm[k]-atom.z
                        
                        xtemp = xmm[i]
                        ytemp = ymm[j]
                        ztemp = zmm[k]
                        r = np.sqrt( dx**2 + dy**2 + dz**2 )
                        if r==0:  # nudge the point 10% in towrads the cell center, just to avoid 1/0 cases
                            dx = 0.99*(xmm[i]-atom.x) + 0.01*(xmm[(i+1)%2]-atom.x)
                            dy = 0.99*(ymm[j]-atom.y) + 0.01*(ymm[(j+1)%2]-atom.y)
                            dz = 0.99*(zmm[k]-atom.z) + 0.01*(zmm[(k+1)%2]-atom.z)
                            
                            xtemp = 0.99*(xmm[i]) + 0.01*(xmm[(i+1)%2])
                            ytemp = 0.99*(ymm[j]) + 0.01*(ymm[(j+1)%2])
                            ztemp = 0.99*(zmm[k]) + 0.01*(zmm[(k+1)%2])
                            r = np.sqrt( dx**2 + dy**2 + dz**2 )
#                         inclination = np.arccos(dz/r)
#                         azimuthal = np.arctan2(dy,dx)
#                     
#                         
#                     
#                         if m<0:
#                             Y = (sph_harm(m,ell,azimuthal,inclination) + (-1)**m * sph_harm(-m,ell,azimuthal,inclination))/np.sqrt(2) 
#                         if m>0:
#                             Y = 1j*(sph_harm(m,ell,azimuthal,inclination) - (-1)**m * sph_harm(-m,ell,azimuthal,inclination))/np.sqrt(2)
#                         if ( m==0 ):
#                             Y = sph_harm(m,ell,azimuthal,inclination)
#
                        Y = sph_harm(0,ell,0,0)
                        psi = atom.interpolators[psiID](r)*np.real(Y)
                        psiSq = atom.interpolators[psiID](r)*np.real(Y)

#                         psi = atom.interpolators[psiID](r)
#                         psiSq = atom.interpolators[psiID](r)
                        Vext = atom.V(xtemp,ytemp,ztemp)
                        
                        if ( (i==0) and (j==0) and (k==0)): 
                            maxPsi = psi
                            maxPsiSq = psiSq
                            minPsi = psi
                            minPsiSq = psiSq
                            maxPsiVext = psi*Vext
                            minPsiVext = psi*Vext
                        else:
                            if psi>maxPsi: maxPsi = psi
                            if psiSq>maxPsiSq: maxPsiSq = psiSq
                            if psi<minPsi: minPsi = psi
                            if psiSq<minPsiSq: minPsiSq = psiSq
                            if psi*Vext>maxPsiVext: maxPsiVext = psi*Vext
                            if psi*Vext<minPsiVext: minPsiVext = psi*Vext

                        
                        absIntegral += abs(psi)*self.volume/8
                           

                    variation = maxPsi-minPsi
                    psiVextVariation = maxPsiVext - minPsiVext
                    maxabs = max(abs(maxPsi), abs(minPsi))
                    sqVariation = (maxPsiSq-minPsiSq) 
                    
                    
                    if variation>maxVariation:
                        maxVariation = variation
                        variationCause = orbitalIndex
                    if psiVextVariation>maxPsiVextVariation:
                        maxPsiVextVariation = psiVextVariation
                        maxVariation = variation
                        psiVextVariationCause = orbitalIndex
                    if sqVariation>maxSqVariation:
                        maxSqVariation = sqVariation
                        sqVariationCause = orbitalIndex
                    if absIntegral > maxAbsIntegral:
                        maxAbsIntegral = absIntegral
                        absIntegralCause = orbitalIndex
                    orbitalIndex += 1
                    singleAtomOrbitalCount += 1
                    
#         return maxVariation, maxRelDensityVariation, maxAbsIntegral, maxSqVariation, variationCause, relDensityVariationCause, absIntegralCause, sqVariationCause
#         return maxVariation, maxPsiVextVariation, maxAbsIntegral, maxSqVariation, variationCause, psiVextVariationCause, absIntegralCause, sqVariationCause
        return maxVariation, maxAbsIntegral, VextIntegral, densityIntegral, variationCause, absIntegralCause,VextIntegralCause, densityIntegralCause
    
    
    
    def wavefunctionVariationAtCorners(self):   
        
        xmm = [self.xmin, self.xmax]
        ymm = [self.ymin, self.ymax]
        zmm = [self.zmin, self.zmax]        
        
        aufbauList = ['10',                                     # n+ell = 1
                      '20',                                     # n+ell = 2
                      '21', '30',                               # n+ell = 3
                      '31', '40', 
                      '32', '41', '50'
                      '42', '51', '60'
                      '43', '52', '61', '70']

        orbitalIndex=0
        
        maxVariation = 0.0
        maxRelDensityVariation = 0.0
        maxPsiVextVariation = 0.0
        maxSqVariation = 0.0
        maxAbsIntegral = 0.0
        sqVariationCause=-2
        densityIntegralCause=-2
        
        VextVariation=0.0
        VextVariationCause = -1
        
        
        ### Compute Density terms based on all atoms
        
        densityIntegral = 0.0
        sqrtDensityIntegral = 0.0
        for i,j,k in TwoByTwoByTwo:
            Vext = 0.0
            density=0.0
            for atom in self.tree.atoms:
                dx = xmm[i]-atom.x
                dy = ymm[j]-atom.y
                dz = zmm[k]-atom.z
                r = np.sqrt( dx**2 + dy**2 + dz**2 )
                if r==0:  # nudge the point 10% in towrads the cell center, just to avoid 1/0 cases
                    dx = 0.99*(xmm[i]-atom.x) + 0.01*(xmm[(i+1)%2]-atom.x)
                    dy = 0.99*(ymm[j]-atom.y) + 0.01*(ymm[(j+1)%2]-atom.y)
                    dz = 0.99*(zmm[k]-atom.z) + 0.01*(zmm[(k+1)%2]-atom.z)
                    
                    xtemp = 0.99*(xmm[i]) + 0.01*(xmm[(i+1)%2])
                    ytemp = 0.99*(ymm[j]) + 0.01*(ymm[(j+1)%2])
                    ztemp = 0.99*(zmm[k]) + 0.01*(zmm[(k+1)%2])
                    r = np.sqrt( dx**2 + dy**2 + dz**2 )
    
                Vext += atom.V(xmm[i],ymm[j],zmm[k])
                density +=  atom.interpolators['density'](r)
                
            densityIntegral += density*(1 + 1/r)/8*self.volume
            sqrtDensityIntegral += density**(4/5)/8*self.volume
             
            if ( (i==0) and (j==0) and (k==0)): 
                maxDensity = density
                minDensity = density
                maxVext = Vext
                minVext = Vext
            else:
                if density>maxDensity: maxDensity = density
                if density<minDensity: minDensity = density
                
                if Vext > maxVext: maxVext = Vext
                if Vext < minVext: minVext = Vext
             
#             relDensityVariation = (maxDensity-minDensity)/maxDensity
#             relDensityVariation = (maxDensity-minDensity)
            relDensityVariation = ( np.sqrt(maxDensity)-np.sqrt(minDensity) ) / np.sqrt(maxDensity)
            if relDensityVariation>maxRelDensityVariation:
                maxRelDensityVariation = relDensityVariation
                relDensityVariationCause = -1
            
            VextVariation = maxVext-minVext
            
    
#         for atom in self.tree.atoms:
        atom = self.nearestAtom
#         nAtomicOrbitals = atom.nAtomicOrbitals
        if atom.atomicNumber <= 4:
            nAtomicOrbitals = 2
        elif atom.atomicNumber <= 10:
            nAtomicOrbitals = 5
        elif atom.atomicNumber <= 12:
            nAtomicOrbitals = 6
        elif atom.atomicNumber <= 18:
            nAtomicOrbitals = 9
        elif atom.atomicNumber <= 20:
            nAtomicOrbitals = 10
        else:
            rprint(rank, 'Atom with atomic number %i.  How many wavefunctions should be used in mesh refinement scheme?' %atom.nAtomicOrbitals)
        
        
       
        singleAtomOrbitalCount=0
        for nell in aufbauList:
            
            if singleAtomOrbitalCount< nAtomicOrbitals:  
                n = int(nell[0])
                ell = int(nell[1])
                psiID = 'psi'+str(n)+str(ell)
                for m in range(-ell,ell+1):
                    
                    
                    absIntegral = 0.0

                    for i,j,k in TwoByTwoByTwo:
                        dx = xmm[i]-atom.x
                        dy = ymm[j]-atom.y
                        dz = zmm[k]-atom.z
                        
                        xtemp = xmm[i]
                        ytemp = ymm[j]
                        ztemp = zmm[k]
                        r = np.sqrt( dx**2 + dy**2 + dz**2 )
                        if r==0:  # nudge the point 10% in towrads the cell center, just to avoid 1/0 cases
                            dx = 0.99*(xmm[i]-atom.x) + 0.01*(xmm[(i+1)%2]-atom.x)
                            dy = 0.99*(ymm[j]-atom.y) + 0.01*(ymm[(j+1)%2]-atom.y)
                            dz = 0.99*(zmm[k]-atom.z) + 0.01*(zmm[(k+1)%2]-atom.z)
                            
                            xtemp = 0.99*(xmm[i]) + 0.01*(xmm[(i+1)%2])
                            ytemp = 0.99*(ymm[j]) + 0.01*(ymm[(j+1)%2])
                            ztemp = 0.99*(zmm[k]) + 0.01*(zmm[(k+1)%2])
                            r = np.sqrt( dx**2 + dy**2 + dz**2 )
#                         inclination = np.arccos(dz/r)
#                         azimuthal = np.arctan2(dy,dx)
#                     
#                         
#                     
#                         if m<0:
#                             Y = (sph_harm(m,ell,azimuthal,inclination) + (-1)**m * sph_harm(-m,ell,azimuthal,inclination))/np.sqrt(2) 
#                         if m>0:
#                             Y = 1j*(sph_harm(m,ell,azimuthal,inclination) - (-1)**m * sph_harm(-m,ell,azimuthal,inclination))/np.sqrt(2)
#                         if ( m==0 ):
#                             Y = sph_harm(m,ell,azimuthal,inclination)
#
                        Y = sph_harm(0,ell,0,0)
                        psi = atom.interpolators[psiID](r)*np.real(Y)
                        psiSq = atom.interpolators[psiID](r)*np.real(Y)

#                         psi = atom.interpolators[psiID](r)
#                         psiSq = atom.interpolators[psiID](r)
                        Vext = atom.V(xtemp,ytemp,ztemp)
                        
                        if ( (i==0) and (j==0) and (k==0)): 
                            maxPsi = psi
                            maxPsiSq = psiSq
                            minPsi = psi
                            minPsiSq = psiSq
                            maxPsiVext = psi*Vext
                            minPsiVext = psi*Vext
                        else:
                            if psi>maxPsi: maxPsi = psi
                            if psiSq>maxPsiSq: maxPsiSq = psiSq
                            if psi<minPsi: minPsi = psi
                            if psiSq<minPsiSq: minPsiSq = psiSq
                            if psi*Vext>maxPsiVext: maxPsiVext = psi*Vext
                            if psi*Vext<minPsiVext: minPsiVext = psi*Vext

                        
                        absIntegral += abs(psi)*self.volume/8
                           

                    variation = maxPsi-minPsi
                    psiVextVariation = maxPsiVext - minPsiVext
                    maxabs = max(abs(maxPsi), abs(minPsi))
                    sqVariation = (maxPsiSq-minPsiSq) 
                    
                    
                    if variation>maxVariation:
                        maxVariation = variation
                        variationCause = orbitalIndex
                    if psiVextVariation>maxPsiVextVariation:
                        maxPsiVextVariation = psiVextVariation
                        maxVariation = variation
                        psiVextVariationCause = orbitalIndex
                    if sqVariation>maxSqVariation:
                        maxSqVariation = sqVariation
                        sqVariationCause = orbitalIndex
                    if absIntegral > maxAbsIntegral:
                        maxAbsIntegral = absIntegral
                        absIntegralCause = orbitalIndex
                    orbitalIndex += 1
                    singleAtomOrbitalCount += 1
                    
#         return maxVariation, maxRelDensityVariation, maxAbsIntegral, maxSqVariation, variationCause, relDensityVariationCause, absIntegralCause, sqVariationCause
#         return maxVariation, maxPsiVextVariation, maxAbsIntegral, maxSqVariation, variationCause, psiVextVariationCause, absIntegralCause, sqVariationCause
#         return maxVariation, VextVariation, maxAbsIntegral, maxSqVariation, variationCause, VextVariationCause, absIntegralCause, sqVariationCause
#         return maxVariation, densityIntegral, maxAbsIntegral, maxSqVariation, variationCause, densityIntegralCause, absIntegralCause, sqVariationCause
#         return maxVariation, sqrtDensityIntegral, maxAbsIntegral, maxSqVariation, variationCause, densityIntegralCause, absIntegralCause, sqVariationCause
    
#         return maxVariation, sqrtDensityIntegral, densityIntegral, VextVariation, variationCause, densityIntegralCause, densityIntegralCause, VextVariationCause
        return maxVariation, sqrtDensityIntegral, relDensityVariation, VextVariation, variationCause, densityIntegralCause, densityIntegralCause, VextVariationCause
#         return maxVariation, maxAbsIntegral, densityIntegral, VextVariation, variationCause, absIntegralCause, densityIntegralCause, VextVariationCause
    
    
    
    def computeLogDensityVariation(self):   
        
        xmm = [self.xmin, self.xmax]
        ymm = [self.ymin, self.ymax]
        zmm = [self.zmin, self.zmax]        
        
 
        logDensityVariation = 0.0
        
        
        
        ### Compute Density terms based on all atoms

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    density=0.0
                    for atom in self.tree.atoms:
                        dx = xmm[i]-atom.x
                        dy = ymm[j]-atom.y
                        dz = zmm[k]-atom.z
                        r = np.sqrt( dx**2 + dy**2 + dz**2 )
        
            
                        density +=  atom.interpolators['density'](r)
                        
                
                    if ( (i==0) and (j==0) and (k==0)): 
                        maxDensity = density
                        minDensity = density
                        
                    else:
                        if density>maxDensity: maxDensity = density
                        if density<minDensity: minDensity = density
                        

            
        logDensityVariation = np.log(maxDensity)-np.log(minDensity)
            
    

        return logDensityVariation
    
    def computeDensitySplitByNearAndFar(self):   
        
        xmm = [self.xmin, self.xmax]
        ymm = [self.ymin, self.ymax]
        zmm = [self.zmin, self.zmax]        
        
 
        densityVariation = 0.0
        sqrtDensityIntegral=0.0
        densityIntegral=0.0
        
        
        ### Compute Density terms based on all atoms

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    density=0.0
                    for atom in self.tree.atoms:
                        dx = xmm[i]-atom.x
                        dy = ymm[j]-atom.y
                        dz = zmm[k]-atom.z
                        r = np.sqrt( dx**2 + dy**2 + dz**2 )
        
            
                        density +=  atom.interpolators['density'](r)
                        
                        
                    sqrtDensityIntegral += np.sqrt(density)/8*self.volume   
                    densityIntegral += density/8*self.volume   
                
                    if ( (i==0) and (j==0) and (k==0)): 
                        maxDensity = density
                        minDensity = density
                        
                    else:
                        if density>maxDensity: maxDensity = density
                        if density<minDensity: minDensity = density
                        

        diagdist = np.sqrt( (xmm[0]-xmm[1])**2 + (ymm[0]-ymm[1])**2 + (zmm[0]-zmm[1])**2 )    
#         densityVariation = (maxDensity-minDensity)*diagdist

#         densityVariation = (maxDensity-minDensity)*self.volume
        densityVariation = (maxDensity-minDensity)
        
        density = 0.0
        for atom in self.tree.atoms:
            dx = self.xmid-atom.x
            dy = self.ymid-atom.y
            dz = self.zmid-atom.z
            r = np.sqrt( dx**2 + dy**2 + dz**2 )


            density +=  atom.interpolators['density'](r)
        # apply one criteria if rho>1, other if <1    
        if density>=1.0:
            sqrtDensityIntegral = 0.0
        else:
            densityVariation = 0.0
            densityIntegral = 0.0

#         return densityVariation, sqrtDensityIntegral
        return densityIntegral, sqrtDensityIntegral
    
    
    
    def initializeCellWavefunctionsForSingleAtom(self,atom):           
        
        aufbauList = ['10',                                     # n+ell = 1
                      '20',                                     # n+ell = 2
                      '21', '30',                               # n+ell = 3
                      '31', '40', 
                      '32', '41', '50'
                      '42', '51', '60'
                      '43', '52', '61', '70']

        orbitalIndex=0

        nAtomicOrbitals = atom.nAtomicOrbitals
                  
        singleAtomOrbitalCount=0
        for nell in aufbauList:
            
            if singleAtomOrbitalCount< nAtomicOrbitals:  
                n = int(nell[0])
                ell = int(nell[1])
                psiID = 'psi'+str(n)+str(ell)
#                     rprint(rank, 'Using ', psiID)
                for m in range(-ell,ell+1):
#                         for _,cell in self.masterList:
#                             if cell.leaf==True:

                    for i,j,k in self.PxByPyByPz:
                        gp = self.gridpoints[i,j,k]
                        dx = gp.x-atom.x
                        dy = gp.y-atom.y
                        dz = gp.z-atom.z
                        r = np.sqrt( dx**2 + dy**2 + dz**2 )
                        inclination = np.arccos(dz/r)
                        azimuthal = np.arctan2(dy,dx)
                    
                        
                    
                        if m<0:
                            Y = (sph_harm(m,ell,azimuthal,inclination) + (-1)**m * sph_harm(-m,ell,azimuthal,inclination))/np.sqrt(2) 
                        if m>0:
                            Y = 1j*(sph_harm(m,ell,azimuthal,inclination) - (-1)**m * sph_harm(-m,ell,azimuthal,inclination))/np.sqrt(2)
                        if ( m==0 ):
                            Y = sph_harm(m,ell,azimuthal,inclination)

                        try:
                            gp.phi[orbitalIndex] = atom.interpolators[psiID](r)*np.real(Y)
                        except ValueError:
                            gp.phi[orbitalIndex] = 0.0
                           
                    orbitalIndex += 1
                    singleAtomOrbitalCount += 1
    
    
    def initializeCellWavefunctions(self):           
        
        aufbauList = ['10',                                     # n+ell = 1
                      '20',                                     # n+ell = 2
                      '21', '30',                               # n+ell = 3
                      '31', '40', 
                      '32', '41', '50'
                      '42', '51', '60'
                      '43', '52', '61', '70']

        orbitalIndex=0

    
        for atom in self.tree.atoms:
            nAtomicOrbitals = atom.nAtomicOrbitals
            
#             dx=[]
#             dy=[]
#             dz=[]
#             for i,j,k in self.PxByPyByPz:
#                 gp = self.gridpoints[i,j,k]
#                 dx = np.append(dx, gp.x-atom.x)
#                 dy = np.append(dy,gp.y-atom.y)
#                 dz = np.append(dz,gp.z-atom.z)
#             r = np.sqrt( dx**2 + dy**2 + dz**2 )
#             inclination = np.arccos(dz/r)
#             azimuthal = np.arctan2(dy,dx)
                
            
            
#             rprint(rank, 'Initializing orbitals for atom Z = %i located at (x, y, z) = (%6.3f, %6.3f, %6.3f)' 
#                       %(atom.atomicNumber, atom.x,atom.y,atom.z))
#             rprint(rank, 'Orbital index = %i'%orbitalIndex)            
            singleAtomOrbitalCount=0
            for nell in aufbauList:
                
                if singleAtomOrbitalCount< nAtomicOrbitals:  
                    n = int(nell[0])
                    ell = int(nell[1])
                    psiID = 'psi'+str(n)+str(ell)
#                     rprint(rank, 'Using ', psiID)
                    for m in range(-ell,ell+1):
#                         for _,cell in self.masterList:
#                             if cell.leaf==True:

                        for i,j,k in self.PxByPyByPz:
                            gp = self.gridpoints[i,j,k]
                            dx = gp.x-atom.x
                            dy = gp.y-atom.y
                            dz = gp.z-atom.z
                            r = np.sqrt( dx**2 + dy**2 + dz**2 )
                            inclination = np.arccos(dz/r)
                            azimuthal = np.arctan2(dy,dx)
                        
                            
                        
#                                     Y = sph_harm(m,ell,azimuthal,inclination)*np.exp(-1j*m*azimuthal)
                            if m<0:
                                Y = (sph_harm(m,ell,azimuthal,inclination) + (-1)**m * sph_harm(-m,ell,azimuthal,inclination))/np.sqrt(2) 
                            if m>0:
                                Y = 1j*(sph_harm(m,ell,azimuthal,inclination) - (-1)**m * sph_harm(-m,ell,azimuthal,inclination))/np.sqrt(2)
    #                                     if ( (m==0) and (ell>1) ):
                            if ( m==0 ):
                                Y = sph_harm(m,ell,azimuthal,inclination)
    #                                     if ( (m==0) and (ell<=1) ):
    #                                         Y = 1
    #                         if abs(np.imag(Y)) > 1e-14:
    #                             rprint(rank, 'imag(Y) ', np.imag(Y))
    #                                     Y = np.real(sph_harm(m,ell,azimuthal,inclination))
                            try:
                                gp.phi[orbitalIndex] = atom.interpolators[psiID](r)*np.real(Y)
                            except ValueError:
                                gp.phi[orbitalIndex] = 0.0
                               
#                             count=0 
#                             for i,j,k in self.PxByPyByPz:
#                                 gp.phi[orbitalIndex] = phi[count]
#                                 count += 1
                                        
                        
                        
#                         rprint(rank, 'Cell %s Orbital %i filled with (n,ell,m) = (%i,%i,%i) ' %(self.uniqueID,orbitalIndex,n,ell,m))
                        orbitalIndex += 1
                        singleAtomOrbitalCount += 1
     
     
    def checkDensityInterpolation(self, divideParameter1, divideParameter2, divideParameter3, divideParameter4):                    
        self.divideFlag=False
        
        xmm = [self.xmin, self.xmax]
        ymm = [self.ymin, self.ymax]
        zmm = [self.zmin, self.zmax] 
        
        midpointDensity=0.0
        midpointSqrtDensity=0.0
        interpolatedDensity = 0.0
        interpolatedSqrtDensity=0.0
        cornerDensities = np.zeros((2,2,2))
        for atom in self.tree.atoms:
            densitySum = 0.0
            dx = self.xmid-atom.x
            dy = self.ymid-atom.y
            dz = self.zmid-atom.z
            r = np.sqrt( dx**2 + dy**2 + dz**2 )

            midpointDensity +=  atom.interpolators['density'](r)
            midpointSqrtDensity +=  np.sqrt( atom.interpolators['density'](r) )
            
            for i,j,k in TwoByTwoByTwo:
                dx = xmm[i]-atom.x
                dy = ymm[j]-atom.y
                dz = zmm[k]-atom.z
                
                r = np.sqrt( dx**2 + dy**2 + dz**2 )
                
                interpolatedDensity += atom.interpolators['density'](r) 
                interpolatedSqrtDensity += np.sqrt( atom.interpolators['density'](r) )
            
                
            
        interpolatedDensity /= 8 # for trilinear interpolation at midpoint, just use the average value.
        interpolatedSqrtDensity /= 8
        
        interpolationError = np.abs(midpointDensity - interpolatedDensity)
        relInterpolationError = np.abs(midpointDensity - interpolatedDensity)/midpointDensity
        sqrtInterpolationError = np.abs(midpointSqrtDensity - interpolatedSqrtDensity)
        relSqrtInterpolationError = np.abs(midpointSqrtDensity - interpolatedSqrtDensity)/midpointSqrtDensity
    

    
        if interpolationError > divideParameter1:

            self.divideFlag = True
            self.childrenRefineCause=1
            return
        
        if sqrtInterpolationError > divideParameter2:

            self.divideFlag = True
            self.childrenRefineCause=2
            return
        
        if relInterpolationError > divideParameter3:

            self.divideFlag = True
            self.childrenRefineCause=3
            return
        
        if relSqrtInterpolationError > divideParameter4:

            self.divideFlag = True
            self.childrenRefineCause=4
            return
        
     
    def checkDensityIntegral(self, divideParameter1, divideParameter2):                    
        self.divideFlag=False
        
        midpointDensity=0.0
        VextIntegral=0.0
        for atom in self.tree.atoms:
            dx = self.xmid-atom.x
            dy = self.ymid-atom.y
            dz = self.zmid-atom.z
            r = np.sqrt( dx**2 + dy**2 + dz**2 )

            midpointDensity +=  atom.interpolators['density'](r)
            VextIntegral += self.volume * np.abs(atom.V(dx,dy,dz))*midpointDensity**(1/2)
            
            x = min( abs(self.xmin - atom.x), abs(self.xmax-atom.x))
            X = max( abs(self.xmin - atom.x), abs(self.xmax-atom.x))
            y = min( abs(self.ymin - atom.y), abs(self.ymax-atom.y))
            Y = max( abs(self.ymin - atom.y), abs(self.ymax-atom.y))
            z = min( abs(self.zmin - atom.z), abs(self.zmax-atom.z))
            Z = max( abs(self.zmin - atom.z), abs(self.zmax-atom.z))
            
            r = np.sqrt( x*x + y*y + z*z )
            R = np.sqrt( X*X + Y*Y + Z*Z )

#         densityIntegral1 = self.volume * (midpointDensity * np.sqrt(r) )
#         VextIntegral = self.volume * np.abs(midpointVext)*midpointDensity


        densityIntegral2 = self.volume * (midpointDensity)**(1/2)
        
        

        
        if VextIntegral > divideParameter1:
#         if rsq >= Rsq: rprint(rank, 'Warning, rsq >= Rsq.  This should not happen.')
#         if (1/np.sqrt(r)-1/np.sqrt(R)) > divideParameter1:
#         if (R - r)*midpointDensity > divideParameter1:
            self.divideFlag = True
            self.childrenRefineCause=1
            return
        if densityIntegral2 > divideParameter2:
            self.divideFlag = True
            self.childrenRefineCause=2
            return
    
    def checkMeshDensity_Nathan(self, divideParameter1, divideParameter2):                    
        self.divideFlag=False
        
        for atom in self.tree.atoms:
            dx = self.xmid-atom.x
            dy = self.ymid-atom.y
            dz = self.zmid-atom.z
            r = np.sqrt( dx**2 + dy**2 + dz**2 )

            
            if self.volume > divideParameter1*np.exp(r)/r:
                self.divideFlag = True
                
            return
    
    def  checkLogDensityVariation(self, divideParameter1, divideParameter2, divideParameter3, divideParameter4):
        self.divideFlag = False
        LogDensityVariation = self.computeLogDensityVariation()
        
        if LogDensityVariation > divideParameter1:
            self.divideFlag=True
            self.childrenRefineCause=1
            
        return
    
    def splitNearAndFar(self, divideParameter1, divideParameter2, divideParameter3, divideParameter4):
        self.divideFlag = False
        LogDensityVariation = self.computeLogDensityVariation()
        
        densityVariation, sqrtDensityIntegral = self.computeDensitySplitByNearAndFar()
        
        if densityVariation > divideParameter1:
            self.divideFlag=True
            self.childrenRefineCause=1
            
        if sqrtDensityIntegral > divideParameter2:
            self.divideFlag=True
            self.childrenRefineCause=2
            
        return
                        
    def checkWavefunctionVariation(self, divideParameter1, divideParameter2, divideParameter3, divideParameter4):
#         rprint(rank, 'Working on Cell centered at (%f,%f,%f) with volume %f' %(self.xmid, self.ymid, self.zmid, self.volume))
#         rprint(rank, 'Working on Cell %s' %(self.uniqueID))
        self.divideFlag = False
#         self.initializeCellWavefunctions()
#         self.initializeCellWavefunctionsAtCorners()

#         waveVariation, relDensityVariation, absIntegral, sqWaveVariation, variationCause, densityVariationCause, absIntegralCause, sqVariationCause = self.wavefunctionVariationAtCorners()
#         waveVariation, psiVextVariation, absIntegral, sqWaveVariation, variationCause, psiVextVariationCause, absIntegralCause, sqVariationCause = self.wavefunctionVariationAtCorners()
#         waveVariation, VextVariation, absIntegral, sqWaveVariation, variationCause, VextVariationCause, absIntegralCause, sqVariationCause = self.wavefunctionVariationAtCorners()
#         waveVariation, waveIntegral, densityIntegral, VextVariation, variationCause, waveIntegralCause, densityIntegralCause, VextVariationCause = self.wavefunctionVariationAtCorners()
        waveVariation, sqrtDensityIntegral, densityIntegral, VextVariation, variationCause, sqrtDensityIntegralIntegralCause, densityIntegralCause, VextVariationCause = self.wavefunctionVariationAtCorners()
#         waveVariation, densityIntegral, absIntegral, sqWaveVariation, variationCause, densityIntegralCause, absIntegralCause, sqVariationCause = self.wavefunctionVariationAtCorners()
#         waveVariation, sqrtDensityIntegral, absIntegral, sqWaveVariation, variationCause, densityIntegralCause, absIntegralCause, sqVariationCause = self.wavefunctionVariationAtCorners()
        
        
#         if waveVariation > divideParameter1:
#             rprint(rank, 'Excluding criteria 1 for now...')
#             self.divideFlag=True
#             self.childrenRefineCause=1
#             rprint(rank, 'Dividing cell %s because of variation in wavefunction %i.' %(self.uniqueID,variationCause))
#             return
            
#         if waveIntegral > divideParameter2:
#             self.divideFlag=True
# #             rprint(rank, 'Dividing cell %s because of Integral for wavefunction %i.' %(self.uniqueID, waveIntegralCause))
#             return

        if densityIntegral > divideParameter3:
            self.divideFlag=True
            self.childrenRefineCause=3
#             rprint(rank, 'Dividing cell %s because of density integral.' %(self.uniqueID))
            return
        
#         if relDensityVariation > divideParameter4:
#             self.divideFlag=True
#             rprint(rank, 'Dividing cell %s because of variation in density.' %(self.uniqueID))
#             return
        
#         if psiVextVariation > divideParameter4:
#             self.divideFlag=True
#             rprint(rank, 'Dividing cell %s because of psi*Vext variation for wavefunction %i.' %(self.uniqueID, psiVextVariationCause))
#             return
        
#         if VextVariation > divideParameter4:
#             rprint(rank, 'Excluding criteria 4 for now...')

#             self.divideFlag=True
#             self.childrenRefineCause=4
#             rprint(rank, 'Dividing cell %s because of Vext variation.' %(self.uniqueID))
#             return

#         if densityIntegral > divideParameter4:
#             self.divideFlag=True
#             rprint(rank, 'Dividing cell %s because of density integral.' %(self.uniqueID))
#             return
        
        if sqrtDensityIntegral > divideParameter2:
            self.divideFlag=True
            self.childrenRefineCause=2
#             rprint(rank, 'Dividing cell %s because of sqrt(density) integral.' %(self.uniqueID))
            return

    def checkWavefunctionVariation_Vext(self, divideParameter1, divideParameter2, divideParameter3, divideParameter4):
        self.divideFlag = False

        waveVariation, absIntegral, VextIntegral, densityIntegral, variationCause, absIntegralCause, VextVariationCause, densityIntegralCause = self.wavefunctionVariationAtCorners_Vext()

        
        if waveVariation > divideParameter1:
            self.divideFlag=True
            rprint(rank, 'Dividing cell %s because of variation in wavefunction %i.' %(self.uniqueID,variationCause))
            return
            

        if absIntegral > divideParameter2:
            self.divideFlag=True
            rprint(rank, 'Dividing cell %s because of absIntegral for wavefunction %i.' %(self.uniqueID, absIntegralCause))
            return

        
        if VextIntegral > divideParameter3:
            self.divideFlag=True
            rprint(rank, 'Dividing cell %s because of Vext integral.' %(self.uniqueID))
            return

        if densityIntegral > divideParameter4:
            self.divideFlag=True
            rprint(rank, 'Dividing cell %s because of density integral.' %(self.uniqueID))
            return
        

            
                        
    def checkIfChebyshevCoefficientsAboveTolerance(self, divideParameter):
#         rprint(rank, 'Working on Cell centered at (%f,%f,%f) with volume %f' %(self.xmid, self.ymid, self.zmid, self.volume))
        self.divideFlag = False
        
        
        # intialize density on this cell
        rho = np.zeros((self.px+1,self.py+1,self.pz+1))
        for i,j,k in self.PxByPyByPz:
            gp = self.gridpoints[i,j,k]
            for atom in self.tree.atoms:
                r = np.sqrt( (gp.x-atom.x)**2 + (gp.y-atom.y)**2 + (gp.z-atom.z)**2 )
                try:
                    rho[i,j,k] += atom.interpolators['density'](r)
#                     phi0[i,j,k] += atom.interpolators['phi10'](r)
                except ValueError:
                    rho[i,j,k] += 0.0   # if outside the interpolation range, assume 0.
        
        # measure density first...
        densityCoefficientSum = sumChebyshevCoefficicentsGreaterThanOrderQ(rho,(self.px-1) + (self.py-1) + (self.pz-1) - 1  )
        
        
#         rprint(rank, 'Density Coefficient Sum = ', coefficientSum)
#         rprint(rank, " ")
        if densityCoefficientSum > divideParameter:
            self.divideFlag=True
            
    def checkIfChebyshevCoefficientsAboveTolerance_allIndicesAboveQ(self, divideParameter):
#         rprint(rank, 'Working on Cell centered at (%f,%f,%f) with volume %f' %(self.xmid, self.ymid, self.zmid, self.volume))
#         rprint(rank, 'Working on Cell %s' %(self.uniqueID))
        self.divideFlag = False
        
        
        # intialize density on this cell
        rho = np.zeros((self.px+1,self.py+1,self.pz+1))
        for i,j,k in self.PxByPyByPz:
            gp = self.gridpoints[i,j,k]
            for atom in self.tree.atoms:
                r = np.sqrt( (gp.x-atom.x)**2 + (gp.y-atom.y)**2 + (gp.z-atom.z)**2 )
                try:
                    rho[i,j,k] += atom.interpolators['density'](r)
#                     phi0[i,j,k] += atom.interpolators['phi10'](r)
                except ValueError:
                    rho[i,j,k] += 0.0   # if outside the interpolation range, assume 0.
        
        # measure density first...
        densityCoefficientSum = sumChebyshevCoefficicentsEachGreaterThanOrderQ(rho,(self.px-1)  )
        

        if densityCoefficientSum > divideParameter:
            self.divideFlag=True
            
    def checkIfChebyshevCoefficientsAboveTolerance_anyIndicesAboveQ(self, divideParameter):
#         rprint(rank, 'Working on Cell centered at (%f,%f,%f) with volume %f' %(self.xmid, self.ymid, self.zmid, self.volume))
#         rprint(rank, 'Working on Cell %s' %(self.uniqueID))
        self.divideFlag = False
        
        
        # intialize density on this cell
        rho = np.zeros((self.px+1,self.py+1,self.pz+1))
        for i,j,k in self.PxByPyByPz:
            gp = self.gridpoints[i,j,k]
            for atom in self.tree.atoms:
                r = np.sqrt( (gp.x-atom.x)**2 + (gp.y-atom.y)**2 + (gp.z-atom.z)**2 )
                try:
                    rho[i,j,k] += atom.interpolators['density'](r)
#                     phi0[i,j,k] += atom.interpolators['phi10'](r)
                except ValueError:
                    rho[i,j,k] += 0.0   # if outside the interpolation range, assume 0.
        
        # measure density first...
        densityCoefficientSum = sumChebyshevCoefficicentsGreaterThanOrderQZeroZero(rho,(self.px-1)  )
        

        if densityCoefficientSum > divideParameter:
            self.divideFlag=True
            
    def checkIfChebyshevCoefficientsAboveTolerance_anyIndicesAboveQ_sumOfWavefunctions(self, divideParameter):
#         rprint(rank, 'Working on Cell centered at (%f,%f,%f) with volume %f' %(self.xmid, self.ymid, self.zmid, self.volume))
#         rprint(rank, 'Working on Cell %s' %(self.uniqueID))
        self.divideFlag = False
        self.initializeCellWavefunctions()
        
        
        # intialize density on this cell
#         rho = np.zeros((self.px+1,self.py+1,self.pz+1))
        phi = np.zeros((self.px+1,self.py+1,self.pz+1,self.tree.nOrbitals))
        for i,j,k in self.PxByPyByPz:
            gp = self.gridpoints[i,j,k]
            
            for m in range(self.tree.nOrbitals):
                phi[i,j,k,m] = gp.phi[m]
        
        # measure density first...
#         densityCoefficientSum = sumChebyshevCoefficicentsAnyGreaterThanOrderQ(rho,(self.px-1)  )
        wavefunctionCoefficientSum = 0.0
        for m in range(self.tree.nOrbitals):
#             wavefunctionCoefficientSum = sumChebyshevCoefficicentsGreaterThanOrderQZeroZero(phi[:,:,:,m],(self.px-1)  )
            wavefunctionCoefficientSum = sumChebyshevCoefficicentsGreaterThanOrderQ(phi[:,:,:,m],(self.px)  )
        

            if wavefunctionCoefficientSum > divideParameter:
                self.divideFlag=True
                rprint(rank, 'Dividing cell %s because of wavefunction %i.' %(self.uniqueID,m))
                return
            
    def checkIfChebyshevCoefficientsAboveTolerance_anyIndicesAboveQ_psi_or_rho(self, divideParameter1, divideParameter2):
        self.divideFlag = False
        
        
        rho = np.zeros((self.px+1,self.py+1,self.pz+1))
        for i,j,k in self.PxByPyByPz:
            gp = self.gridpoints[i,j,k]
            for atom in self.tree.atoms:
                r = np.sqrt( (gp.x-atom.x)**2 + (gp.y-atom.y)**2 + (gp.z-atom.z)**2 )
                try:
                    rho[i,j,k] += atom.interpolators['density'](r)
#                     phi0[i,j,k] += atom.interpolators['phi10'](r)
                except ValueError:
                    rho[i,j,k] += 0.0   # if outside the interpolation range, assume 0.
        
#         densityCoefficientSum = sumChebyshevCoefficicentsGreaterThanOrderQZeroZero(rho,(self.px-1)  )
        densityCoefficientSum = sumChebyshevCoefficicentsAnyGreaterThanOrderQ(rho,(self.px-1)  )
        if densityCoefficientSum > divideParameter1:
            self.divideFlag=True
            rprint(rank, 'Cell %s dividing because of density coefficients.' %self.uniqueID)
            return
                    
        
        self.initializeCellWavefunctions()
          
        phi = np.zeros((self.px+1,self.py+1,self.pz+1,self.tree.nOrbitals))
        for i,j,k in self.PxByPyByPz:
            gp = self.gridpoints[i,j,k]                
            for m in range(self.tree.nOrbitals):
                phi[i,j,k,m] = gp.phi[m]
        
        # measure density first...
#         densityCoefficientSum = sumChebyshevCoefficicentsAnyGreaterThanOrderQ(rho,(self.px-1)  )
        wavefunctionCoefficientSum = 0.0
        for m in range(self.tree.nOrbitals):
#             wavefunctionCoefficientSum = sumChebyshevCoefficicentsGreaterThanOrderQZeroZero(phi[:,:,:,m],(self.px-1)  )
            wavefunctionCoefficientSum = sumChebyshevCoefficicentsAnyGreaterThanOrderQ(phi[:,:,:,m],(self.px-1)  )
        

            if wavefunctionCoefficientSum > divideParameter2:
                self.divideFlag=True
                rprint(rank, 'Cell %s dividing because of wavefunction %i coefficients.' %(self.uniqueID,m))
                return
            
    def checkIfChebyshevCoefficientsAboveTolerance_anyIndicesAboveQ_rho_sqrtRho(self, divideParameter1, divideParameter2):
        self.divideFlag = False
        
        
        rho = np.zeros((self.px+1,self.py+1,self.pz+1))
        sqrtrho = np.zeros((self.px+1,self.py+1,self.pz+1))
        for i,j,k in self.PxByPyByPz:
            gp = self.gridpoints[i,j,k]
            for atom in self.tree.atoms:
                r = np.sqrt( (gp.x-atom.x)**2 + (gp.y-atom.y)**2 + (gp.z-atom.z)**2 )
                try:
                    d = atom.interpolators['density'](r)
                    rho[i,j,k] += d
                    sqrtrho[i,j,k] += np.sqrt( d ) 
#                     phi0[i,j,k] += atom.interpolators['phi10'](r)
                except ValueError:
                    rho[i,j,k] += 0.0   # if outside the interpolation range, assume 0.
                    sqrtrho[i,j,k] += 0.0   # if outside the interpolation range, assume 0.
        
#         densityCoefficientSum = sumChebyshevCoefficicentsGreaterThanOrderQZeroZero(rho,(self.px-1)  )
        densityCoefficientSum = sumChebyshevCoefficicentsAnyGreaterThanOrderQ(rho,(self.px-1)  )
        if densityCoefficientSum > divideParameter1:
            self.divideFlag=True
            rprint(rank, 'Cell %s dividing because of density coefficients.' %self.uniqueID)
            return
        
        sqrtDensityCoefficientSum = sumChebyshevCoefficicentsAnyGreaterThanOrderQ(rho,(self.px-1)  )
        if sqrtDensityCoefficientSum > divideParameter2:
            self.divideFlag=True
            rprint(rank, 'Cell %s dividing because of sqrt(density) coefficients.' %self.uniqueID)
            return
                    
        
    
            
    def checkIfChebyshevCoefficientsAboveTolerance_anyIndicesAboveQ_psi_or_rho_or_v(self, divideParameter):
        self.divideFlag = False
        
        
        rho = np.zeros((self.px+1,self.py+1,self.pz+1))
        vext = np.zeros((self.px+1,self.py+1,self.pz+1))
        for i,j,k in self.PxByPyByPz:
            gp = self.gridpoints[i,j,k]
            gp.setExternalPotential(self.tree.atoms)
            vext[i,j,k] = gp.v_ext
            for atom in self.tree.atoms:
                r = np.sqrt( (gp.x-atom.x)**2 + (gp.y-atom.y)**2 + (gp.z-atom.z)**2 )
                try:
                    rho[i,j,k] += atom.interpolators['density'](r)
#                     phi0[i,j,k] += atom.interpolators['phi10'](r)
                except ValueError:
                    rho[i,j,k] += 0.0   # if outside the interpolation range, assume 0.
        
        externalPotentialCoefficientSum = sumChebyshevCoefficicentsGreaterThanOrderQZeroZero(vext,(self.px-1)  )
        if externalPotentialCoefficientSum > divideParameter:
            rprint(rank, 'Dividing cell %s because of the external potential.' %self.uniqueID)
            self.divideFlag=True
            return
        
        densityCoefficientSum = sumChebyshevCoefficicentsGreaterThanOrderQZeroZero(rho,(self.px-1)  )
        if densityCoefficientSum > divideParameter:
            rprint(rank, 'Dividing cell %s because of the density.' %self.uniqueID)
            self.divideFlag=True
            return
                    
        
        self.initializeCellWavefunctions()
          
        phi = np.zeros((self.px+1,self.py+1,self.pz+1,self.tree.nOrbitals))
        for i,j,k in self.PxByPyByPz:
            gp = self.gridpoints[i,j,k]                
            for m in range(self.tree.nOrbitals):
                phi[i,j,k,m] = gp.phi[m]
        
        # measure density first...
#         densityCoefficientSum = sumChebyshevCoefficicentsAnyGreaterThanOrderQ(rho,(self.px-1)  )
        wavefunctionCoefficientSum = 0.0
        for m in range(self.tree.nOrbitals):
            wavefunctionCoefficientSum = sumChebyshevCoefficicentsGreaterThanOrderQZeroZero(phi[:,:,:,m],(self.px-1)  )
        

            if wavefunctionCoefficientSum > divideParameter:
                rprint(rank, 'Dividing cell %s because of wavefunction %i.' %(self.uniqueID,m))
                self.divideFlag=True
                return
            
            
    def checkIfChebyshevCoefficientsAboveTolerance_DensityAndWavefunctions(self, divideParameter):
#         rprint(rank, 'Working on Cell centered at (%f,%f,%f) with volume %f' %(self.xmid, self.ymid, self.zmid, self.volume))
        self.divideFlag = False
        
        self.initializeCellWavefunctions()
        
        # intialize density on this cell
        rho = np.zeros((self.px+1,self.py+1,self.pz+1))
        phi = np.zeros((self.px+1,self.py+1,self.pz+1,self.tree.nOrbitals))
#         phi0 = np.zeros((self.px+1,self.py+1,self.pz+1))
#         phi1 = np.zeros((self.px+1,self.py+1,self.pz+1))
        for i,j,k in self.PxByPyByPz:
            gp = self.gridpoints[i,j,k]
            for atom in self.tree.atoms:
                r = np.sqrt( (gp.x-atom.x)**2 + (gp.y-atom.y)**2 + (gp.z-atom.z)**2 )
                try:
                    rho[i,j,k] += atom.interpolators['density'](r)
#                     phi0[i,j,k] += atom.interpolators['phi10'](r)
                except ValueError:
                    rho[i,j,k] += 0.0   # if outside the interpolation range, assume 0.
                    
            for m in range(self.tree.nOrbitals):
                phi[i,j,k,m] = gp.phi[m]
        
        # measure density first...
        densityCoefficientSum = sumChebyshevCoefficicentsGreaterThanOrderQ(rho,(self.px-1) + (self.py-1) + (self.pz-1) - 1  )
        
        
        # Now the wavefunctions
        wavefunctionCoefficientSum = 0.0
        for m in range(self.tree.nOrbitals):
            wavefunctionCoefficientSum += sumChebyshevCoefficicentsGreaterThanOrderQ(phi[:,:,:,m],(self.px-1) + (self.py-1) + (self.pz-1) - 1  )
        
#         rprint(rank, 'Cell ID: ', self.uniqueID)
#         rprint(rank, 'Density Coefficient Sum = ', densityCoefficientSum)
#         rprint(rank, 'Wavefunction Coefficient Sum = ', wavefunctionCoefficientSum)
#         rprint(rank, " ")
#         if wavefunctionCoefficientSum > densityCoefficientSum:
#             rprint(rank, 'wavefunctino sum greater than density sum.')
        if (densityCoefficientSum+wavefunctionCoefficientSum) > divideParameter:
            self.divideFlag=True
#         if (densityCoefficientSum) > divideParameter:
#             self.divideFlag=True
    
    def intializeAndIntegrateDensity(self):
        if self.kind=='second':
            rho = np.zeros((self.px+1,self.py+1,self.pz+1))
            r = np.zeros((self.px+1,self.py+1,self.pz+1))
            Vext = np.zeros((self.px+1,self.py+1,self.pz+1))
            weights = np.zeros((self.px+1,self.py+1,self.pz+1))
        elif self.kind=='first': 
            rho = np.zeros((self.px+1,self.py+1,self.pz+1))
            r = np.zeros((self.px+1,self.py+1,self.pz+1))
            Vext = np.zeros((self.px+1,self.py+1,self.pz+1))
            weights = np.zeros((self.px+1,self.py+1,self.pz+1))
        
        for i,j,k in self.PxByPyByPz:
            weights[i,j,k] = self.w[i,j,k]
                
        for atom in self.tree.atoms:
            dx = self.xmid-atom.x
            dy = self.ymid-atom.y
            dz = self.zmid-atom.z
            distToAtom = np.sqrt( (dx)**2 + (dy)**2 + (dz)**2 )
            if distToAtom < 30:
                for i,j,k in self.PxByPyByPz:
                    gp = self.gridpoints[i,j,k]
                    dx = gp.x-atom.x
                    dy = gp.y-atom.y
                    dz = gp.z-atom.z
                    r[i,j,k] = np.sqrt( (dx)**2 + (dy)**2 + (dz)**2 )
#                     Vext[i,j,k] += atom.V(gp.x,gp.y,gp.z)
    #                 try:
    #                     rho[i,j,k] += atom.interpolators['density'](r)
    #                 except ValueError:
    #                     rho[i,j,k] += 0.0   # if outside the interpolation range, assume 0.
#                 try:
#                     rho += atom.interpolators['density'](r)    # increment rho for each atom  
#                 except ValueError:
#                     rho += 0 
#                 print("About to use atom interpolator for atom ", atom)
                if atom.coreRepresentation=="AllElectron":
                    rho += atom.interpolators['density'](r)    # increment rho for each atom 
                    Vext += -atom.atomicNumber / r
                elif atom.coreRepresentation=="Pseudopotential":
#                     rprint(rank, "USING ALLELECTRON DENSITY AND POTENTIAL FOR INITIALIZATION OF PSEUDOPOTENTIAL CALCULATION")
#                     rho += atom.interpolators['density'](r)    # increment rho for each atom 
#                     Vext += -atom.atomicNumber / r
                    rho += np.reshape(atom.PSP.evaluateDensityInterpolator(r.flatten()),(self.px+1,self.py+1,self.pz+1))
                    Vext += np.reshape(atom.PSP.evaluateLocalPotentialInterpolator(r.flatten()),(self.px+1,self.py+1,self.pz+1))
                    Vext -= 10 # some pseudopotentials go positive at the origin.  Don't want those going in to the sqrt.  Constant shift is okay, it doesnt affect integration accuracy
                            
        
        densityIntegral = 1 #np.sum(rho*weights)
        sqrtDensityIntegral = 1 #np.sum(np.sqrt(rho)*weights) 
        sqrtDensityVextIntegral = np.sum(np.sqrt(rho)*Vext*weights)

        return densityIntegral, sqrtDensityIntegral, sqrtDensityVextIntegral
    
    
    def intializeAndIntegrateNonlocal(self):
        psi = np.zeros((self.px+1,self.py+1,self.pz+1))
        r = np.zeros((self.px+1,self.py+1,self.pz+1))
        X = np.zeros((self.px+1,self.py+1,self.pz+1))
        Y = np.zeros((self.px+1,self.py+1,self.pz+1))
        Z = np.zeros((self.px+1,self.py+1,self.pz+1))
        Vext = np.zeros((self.px+1,self.py+1,self.pz+1))
        weights = np.zeros((self.px+1,self.py+1,self.pz+1))
        
        aufbauList = ['10',                                     # n+ell = 1
                      '20',                                     # n+ell = 2
                      '21', '30',                               # n+ell = 3
                      '31', '40', 
                      '32', '41', '50'
                      '42', '51', '60'
                      '43', '52', '61', '70']

        
        
        for i,j,k in self.PxByPyByPz:
            weights[i,j,k] = self.w[i,j,k]
            X[i,j,k] = self.gridpoints[i,j,k].x
            Y[i,j,k] = self.gridpoints[i,j,k].y
            Z[i,j,k] = self.gridpoints[i,j,k].z
            
        integralVlocPsiPlusVnlocPsi = 0
                
        for atom in self.tree.atoms:
            
            dx = self.xmid-atom.x
            dy = self.ymid-atom.y
            dz = self.zmid-atom.z
            distToAtom = np.sqrt( (dx)**2 + (dy)**2 + (dz)**2 )
            if distToAtom < 8:
                
                if atom.coreRepresentation=="AllElectron":
#                     rho += atom.interpolators['density'](r)    # increment rho for each atom 
                    Vext += -atom.atomicNumber / r
                elif atom.coreRepresentation=="Pseudopotential":
#                     rho += np.reshape(atom.PSP.evaluateDensityInterpolator(r.flatten()),(self.px+1,self.py+1,self.pz+1))
                    Vext += np.reshape(atom.PSP.evaluateLocalPotentialInterpolator(r.flatten()),(self.px+1,self.py+1,self.pz+1))
                    Vext -= 10 # some pseudopotentials go positive at the origin.  Don't want those going in to the sqrt.  Constant shift is okay, it doesnt affect integration accuracy
                    atom.generateChi(X.flatten(),Y.flatten(),Z.flatten()
                                     
                                             )
                
                orbitalIndex=0
                nAtomicOrbitals = atom.nAtomicOrbitals
                singleAtomOrbitalCount=0

                
                for nell in aufbauList:
                
                    if singleAtomOrbitalCount< nAtomicOrbitals:  
                        n = int(nell[0])
                        ell = int(nell[1])
                        psiID = 'psi'+str(n)+str(ell)
    #                     rprint(rank, 'Using ', psiID)
                        for m in range(-ell,ell+1):
                            
                            if psiID in atom.interpolators:  # pseudopotentials don't start from 10, 20, 21,... they start from the valence, such as 30, 31, ...
                                dx = X-atom.x
                                dy = Y-atom.y
                                dz = Z-atom.z
                                psi = np.zeros(len(dx))
                                r = np.sqrt( dx**2 + dy**2 + dz**2 )
                                inclination = np.arccos(dz/r)
                                azimuthal = np.arctan2(dy,dx)
                                
                                if m<0:
                                    Ysp = (sph_harm(m,ell,azimuthal,inclination) + (-1)**m * sph_harm(-m,ell,azimuthal,inclination))/np.sqrt(2) 
                                if m>0:
                                    Ysp = 1j*(sph_harm(m,ell,azimuthal,inclination) - (-1)**m * sph_harm(-m,ell,azimuthal,inclination))/np.sqrt(2)
        #                                     if ( (m==0) and (ell>1) ):
                                if ( m==0 ):
                                    Ysp = sph_harm(m,ell,azimuthal,inclination)
        #                                     if ( (m==0) and (ell<=1) ):
        #                                         Y = 1
                                if np.max( abs(np.imag(Ysp)) ) > 1e-14:
                                    rprint(rank, 'imag(Y) ', np.imag(Ysp))
                                    return
        #                                     Y = np.real(sph_harm(m,ell,azimuthal,inclination))
        #                         phi = atom.interpolators[psiID](r)*np.real(Y)
                                try:
                                    psi = atom.interpolators[psiID](r)*np.real(Ysp)
                                except ValueError:
                                    psi = 0.0   # if outside the interpolation range, assume 0.
                                except KeyError:
                                    for key, value in atom.interpolators.items() :
                                        print (key, value)
                                    exit(-1)
                                
                                
#                                 rprint(rank, "Computing integral for psi"+str(n)+str(ell))
                                # integrate local piece
                                integralVlocPsiPlusVnlocPsi += np.sum(psi*Vext*weights)
                                
                                # compute nonlocal piece
#                                 V_nl_psi = atom.V_nonlocal_pseudopotential_times_psi(X.flatten(),Y.flatten(),Z.flatten(),psi.flatten(),weights.flatten(),comm=None)
                                V_nl_psi = atom.V_nonlocal_pseudopotential_times_psi(psi.flatten(),weights.flatten(),comm=None)
                                
                                # integrate nonlocal piece
                                integralVlocPsiPlusVnlocPsi += np.sum(V_nl_psi*weights.flatten())
#                                 integralVlocPsiPlusVnlocPsi = np.sum(V_nl_psi*weights.flatten())
                                
#                                 rprint(rank, "Integral now = ", integralVlocPsiPlusVnlocPsi)
                                
                                orbitalIndex += 1
                                singleAtomOrbitalCount += 1
                                
            atom.removeTempChi()

        return integralVlocPsiPlusVnlocPsi
    
    def intializeAndIntegrateProjectors(self):
#         psi = np.zeros((self.px+1,self.py+1,self.pz+1))
        r = np.zeros((self.px+1,self.py+1,self.pz+1))
        X = np.zeros((self.px+1,self.py+1,self.pz+1))
        Y = np.zeros((self.px+1,self.py+1,self.pz+1))
        Z = np.zeros((self.px+1,self.py+1,self.pz+1))
#         Vext = np.zeros((self.px+1,self.py+1,self.pz+1))
        weights = np.zeros((self.px+1,self.py+1,self.pz+1))
        
        aufbauList = ['10',                                     # n+ell = 1
                      '20',                                     # n+ell = 2
                      '21', '30',                               # n+ell = 3
                      '31', '40', 
                      '32', '41', '50'
                      '42', '51', '60'
                      '43', '52', '61', '70']

        
        
        for i,j,k in self.PxByPyByPz:
            weights[i,j,k] = self.w[i,j,k]
            X[i,j,k] = self.gridpoints[i,j,k].x
            Y[i,j,k] = self.gridpoints[i,j,k].y
            Z[i,j,k] = self.gridpoints[i,j,k].z
            
        integralProjectors = 0
                
        for atom in self.tree.atoms:
            
            dx = self.xmid-atom.x
            dy = self.ymid-atom.y
            dz = self.zmid-atom.z
            distToAtom = np.sqrt( (dx)**2 + (dy)**2 + (dz)**2 )
            if distToAtom < 32:
                
                if atom.coreRepresentation=="AllElectron":
                   exit(-1)
                elif atom.coreRepresentation=="Pseudopotential":
                    atom.generateChi(X.flatten(),Y.flatten(),Z.flatten())
                    integralProjectors += atom.integrateProjectors(weights.flatten(),comm=None)
                
            atom.removeTempChi()

        return integralProjectors
        
     
    def refineByCheckingParentChildrenIntegrals(self, divideParameter1):
        if self.level>=3:
            rprint(rank, 'Cell:                                      ', self.uniqueID)
        self.divideFlag = False
        
        
#         parentIntegral = self.intializeAndIntegrateDensity()
#         if not hasattr(self, "DensityIntegral"):
        parentDensityIntegral, parentSqrtDensityIntegral, parentSqrtDensityVextIntegral = self.intializeAndIntegrateDensity()
#         else:
#             rprint(rank, 'Not recomputing parent integrals...')
#             parentDensityIntegral = self.DensityIntegral
#             parentSqrtDensityIntegral = self.SqrtDensityIntegral
#             parentSqrtDensityVextIntegral = self.SqrtDensityVextIntegral
        sumChildrenIntegrals = 0.0 
        
        sumChildDensityIntegral=0.0
        sumChildSqrtDensityIntegral=0.0
        sumChildSqrtDensityVextVextIntegral=0.0
        
        xdiv = (self.xmax + self.xmin)/2   
        ydiv = (self.ymax + self.ymin)/2   
        zdiv = (self.zmax + self.zmin)/2   
        if self.kind=='second':
            self.divide_secondKind(xdiv, ydiv, zdiv, temporaryCell=True)
        elif self.kind=='first':
            self.divide_firstKind(xdiv, ydiv, zdiv, temporaryCell=True)
        else: 
            rprint(rank, "What kind???")
            return
        (ii,jj,kk) = np.shape(self.children)

        for i in range(ii):
            for j in range(jj):
                for k in range(kk):
                    childDensityIntegral, childSqrtDensityIntegral, childSqrtDensityVextIntegral = self.children[i,j,k].intializeAndIntegrateDensity()
#                     self.children[i,j,k].DensityIntegral = childDensityIntegral
#                     self.children[i,j,k].SqrtDensityIntegral = childSqrtDensityIntegral
#                     self.children[i,j,k].SqrtDensityVextIntegral = childSqrtDensityVextIntegral
                    sumChildDensityIntegral += childDensityIntegral
                    sumChildSqrtDensityIntegral += childSqrtDensityIntegral
                    sumChildSqrtDensityVextVextIntegral += childSqrtDensityVextIntegral
        
#         rprint(rank, " ")
#         rprint(rank, 'Cell:                  ', self.uniqueID)
#         rprint(rank, 'Parent Integral:       ', parentIntegral)
#         rprint(rank, 'Children Integral:     ', sumChildrenIntegrals)
#         rprint(rank, " ")
        
        if np.abs(parentSqrtDensityVextIntegral-sumChildSqrtDensityVextVextIntegral) > divideParameter1:
            self.childrenRefineCause=3
#             rprint(rank, " ")
#             rprint(rank, 'Cell:                                      ', self.uniqueID)
            rprint(rank, 'Parent sqrt(Density)Vext Integral:         ', parentSqrtDensityVextIntegral)
            rprint(rank, 'Children sqrt(Density)Vext Integral:       ', sumChildSqrtDensityVextVextIntegral)
            rprint(rank, " ")
            self.divideFlag=True
        
        
        
        
#         elif np.abs(parentSqrtDensityIntegral-sumChildSqrtDensityIntegral) > divideParameter2:
#             self.childrenRefineCause=2
#             rprint(rank, " ")
#             rprint(rank, 'Cell:                                      ', self.uniqueID)
#             rprint(rank, 'Parent sqrt(Density) Integral:             ', parentSqrtDensityIntegral)
#             rprint(rank, 'Children sqrt(Density) Integral:           ', sumChildSqrtDensityIntegral)
#             rprint(rank, " ")
#             self.divideFlag=True
#             
#         elif np.abs(parentDensityIntegral-sumChildDensityIntegral) > divideParameter1:
#             self.childrenRefineCause=1
#             rprint(rank, " ")
#             rprint(rank, 'Cell:                                      ', self.uniqueID)
#             rprint(rank, 'Parent Density Integral:                   ', parentDensityIntegral)
#             rprint(rank, 'Children Density Integral:                 ', sumChildDensityIntegral)
#             rprint(rank, " ")
#             self.divideFlag=True
            
            
#             for i,j,k in np.shape(self.children):
#                 self.children[i,j,k].checkParentChildrenIntegrals(divideParameter)

        # clean up by deleting children
        for i in range(ii):
            for j in range(jj):
                for k in range(kk):
                    child = self.children[i,j,k]
                    for i2,j2,k2 in child.PxByPyByPz:
                        gp = child.gridpoints[i2,j2,k2]
                        del gp
                        child.gridpoints[i2,j2,k2]=None
                    del child
#         self.children=None
        delattr(self,"children")
        self.leaf=True
        
    def refineByCheckingParentChildrenIntegrals_nonlocal(self, divideParameter1):
        if self.level>=3:
            rprint(rank, 'Cell:                                      ', self.uniqueID)
        self.divideFlag = False
        
        parentIntegral = self.intializeAndIntegrateNonlocal()
        
        if abs(parentIntegral)>0.0:
            sumChildrenIntegrals = 0.0 
            
            
            xdiv = (self.xmax + self.xmin)/2   
            ydiv = (self.ymax + self.ymin)/2   
            zdiv = (self.zmax + self.zmin)/2   
            self.divide_firstKind(xdiv, ydiv, zdiv, temporaryCell=True)
            (ii,jj,kk) = np.shape(self.children)
    
            for i in range(ii):
                for j in range(jj):
                    for k in range(kk):
                        childIntegral = self.children[i,j,k].intializeAndIntegrateNonlocal()
                        sumChildrenIntegrals += childIntegral
                       
            
            if np.abs(parentIntegral-sumChildrenIntegrals) > divideParameter1:
                self.childrenRefineCause=3
    #             rprint(rank, " ")
    #             rprint(rank, 'Cell:                                      ', self.uniqueID)
                rprint(rank, 'Parent Integral:         ', parentIntegral)
                rprint(rank, 'Children Integral:       ', sumChildrenIntegrals)
                rprint(rank, " ")
                self.divideFlag=True
            
            
            # clean up by deleting children
            for i in range(ii):
                for j in range(jj):
                    for k in range(kk):
                        child = self.children[i,j,k]
                        for i2,j2,k2 in child.PxByPyByPz:
                            gp = child.gridpoints[i2,j2,k2]
                            del gp
                            child.gridpoints[i2,j2,k2]=None
                        del child
    #         self.children=None
            delattr(self,"children")
            self.leaf=True
        else:
            self.divideFlag=False
        
    def refineByCheckingParentChildrenIntegrals_Chi(self, divideParameter1):
        if self.level>=3:
            rprint(rank, 'Cell:                                      ', self.uniqueID)
        self.divideFlag = False
        
        parentIntegral = self.intializeAndIntegrateProjectors()
        sumChildrenIntegrals = 0.0 
        
        
        xdiv = (self.xmax + self.xmin)/2   
        ydiv = (self.ymax + self.ymin)/2   
        zdiv = (self.zmax + self.zmin)/2   
        self.divide_firstKind(xdiv, ydiv, zdiv, temporaryCell=True)
        (ii,jj,kk) = np.shape(self.children)

        for i in range(ii):
            for j in range(jj):
                for k in range(kk):
                    childIntegral = self.children[i,j,k].intializeAndIntegrateProjectors()
                    sumChildrenIntegrals += childIntegral
                   
        
        if np.abs(parentIntegral-sumChildrenIntegrals) > divideParameter1:
            self.childrenRefineCause=3
#             rprint(rank, " ")
#             rprint(rank, 'Cell:                                      ', self.uniqueID)
            rprint(rank, 'Parent Integral:         ', parentIntegral)
            rprint(rank, 'Children Integral:       ', sumChildrenIntegrals)
            rprint(rank, " ")
            self.divideFlag=True
        
        
        # clean up by deleting children
        for i in range(ii):
            for j in range(jj):
                for k in range(kk):
                    child = self.children[i,j,k]
                    for i2,j2,k2 in child.PxByPyByPz:
                        gp = child.gridpoints[i2,j2,k2]
                        del gp
                        child.gridpoints[i2,j2,k2]=None
                    del child
#         self.children=None
        delattr(self,"children")
        self.leaf=True
        
    
    def refinePiecewiseUniform(self, nearFieldSpacing, nearFieldCutoff, midFieldSpacing, midFieldCutoff):
        self.divideFlag = False
        
        xdiv = (self.xmax + self.xmin)/2   
        ydiv = (self.ymax + self.ymin)/2   
        zdiv = (self.zmax + self.zmin)/2 
        
        # determine distance from cell corner to nearest atom
        distanceToNearestAtom = 10+midFieldCutoff
        for atom in self.tree.atoms:
            if ( (atom.x<self.xmax) and (atom.y<self.ymax) and (atom.z<self.zmax) and
                 (atom.x>self.xmin) and (atom.y>self.xmin) and (atom.z>self.zmin) ): # atom is in cell
                distanceToNearestAtom=0
            else:
                for x in [self.xmax,self.xmin]:
                    for y in [self.ymin,self.ymax]:
                        for z in [self.zmin,self.zmax]:
                            d = np.sqrt( (x-atom.x)**2 + (y-atom.y)**2 + (z-atom.z)**2)
                            distanceToNearestAtom = min(distanceToNearestAtom,d)
        
        # compute cell size
#         cellRadius = np.sqrt( (self.xmax-xdiv)**2 + (self.ymax-ydiv)**2 + (self.zmax-zdiv)**2 )
        cellSideLength = np.max( [self.zmax-self.zmin, self.xmax-self.xmin, self.ymax-self.ymin ] )
        
        
        
        # detemine if cell should be refined
        if distanceToNearestAtom<nearFieldCutoff:  # if in the inner ring
            if cellSideLength > nearFieldSpacing:
                rprint(rank, "REFINING INNER RING")
                rprint(rank, "New inner cell size will be ", cellSideLength/2)
                self.divideFlag=True
                
        elif distanceToNearestAtom<midFieldCutoff:  # if in the middle annulus
            if cellSideLength > midFieldSpacing:
                rprint(rank, "REFINING MIDDLE ANNULUS")
                rprint(rank, "New annulus cell size will be ", cellSideLength/2)
                self.divideFlag=True
        else:                                       # cell is in the far field
            pass
        
       
        if False:
            rprint(rank, '\nCell:   ', self.uniqueID)
            rprint(rank, "distance to nearest atom: ",distanceToNearestAtom)
            rprint(rank, "Cell center: ", xdiv,ydiv,zdiv)
            rprint(rank, "Cell side length: ",cellSideLength)
            rprint(rank, "Cell dividing: ", self.divideFlag)
            rprint(rank, " ")
          
          
          
            
    def refineCoarseningUniform(self, h, H, r, level):
        self.divideFlag = False
        
        xdiv = (self.xmax + self.xmin)/2   
        ydiv = (self.ymax + self.ymin)/2   
        zdiv = (self.zmax + self.zmin)/2 
        
        # how many coarsening steps are required:
        steps = int( np.log2(H/h))
        
        # determine distance from cell corner to nearest atom
        distanceToNearestAtom = 10+H
        for atom in self.tree.atoms:
            if ( (atom.x<self.xmax) and (atom.y<self.ymax) and (atom.z<self.zmax) and
                 (atom.x>self.xmin) and (atom.y>self.ymin) and (atom.z>self.zmin) ): # atom is in cell
                distanceToNearestAtom=0
            else:
                for x in [self.xmax,self.xmin]:
                    for y in [self.ymin,self.ymax]:
                        for z in [self.zmin,self.zmax]:
                            d = np.sqrt( (x-atom.x)**2 + (y-atom.y)**2 + (z-atom.z)**2)
#                             d = np.max(  [abs(x-atom.x), abs(y-atom.y), abs(z-atom.z)])
                            distanceToNearestAtom = min(distanceToNearestAtom,d)
        
        # compute cell size
#         cellRadius = np.sqrt( (self.xmax-xdiv)**2 + (self.ymax-ydiv)**2 + (self.zmax-zdiv)**2 )
        cellSideLength = np.max( [self.zmax-self.zmin, self.xmax-self.xmin, self.ymax-self.ymin ] )
        
        radiusFactor=1
        if level>=3: radiusFactor=2
        if level>=4: radiusFactor=3
        
        
        # detemine if cell should be refined
        if cellSideLength>H:
            self.divideFlag=True
            
#         elif distanceToNearestAtom<(r/8):    ## REFINING TO h/8 within r/8 DOES NOT IMPROVE ACCURACY
# #             rprint(rank, "Second check: distance < r/4")
#             if cellSideLength > (h/8):
#                 rprint(rank, "First check: New inner cell size will be ", cellSideLength/2)
#                 self.divideFlag=True
        elif ( (level>=2) and (distanceToNearestAtom<(r/4)) ):  
#             rprint(rank, "Second check: distance < r/4")
            if cellSideLength > (h/4):
                rprint(rank, "First check: New inner cell size will be ", cellSideLength/2)
                self.divideFlag=True
        elif ( (level>=1) and (distanceToNearestAtom<(r/2)) ): 
            if cellSideLength > (h/2):
                self.divideFlag=True
                rprint(rank, "Second check: New inner cell size will be ", cellSideLength/2)
        elif distanceToNearestAtom<r:  #if in the inner ring
#         elif distanceToNearestAtom<2:  #if in the inner ring
            if cellSideLength > h:
#                 rprint(rank, "New inner cell size will be ", cellSideLength/2)
                self.divideFlag=True
                
        elif distanceToNearestAtom<r+2*h*np.sqrt(radiusFactor):  # if in the inner ring
#         elif distanceToNearestAtom<3:  # if in the inner ring
            if cellSideLength > 2*h:
#                 rprint(rank, "First annulus cell size will be ", cellSideLength/2)
                self.divideFlag=True
        
        elif distanceToNearestAtom<r+(2*h+4*h)*np.sqrt(radiusFactor):  # if in the inner ring
#         elif distanceToNearestAtom<12:  # if in the inner ring
            if cellSideLength > 4*h:
#                 rprint(rank, "Second annulus cell size will be ", cellSideLength/2)
                self.divideFlag=True
        
        elif distanceToNearestAtom<r+(2*h+4*h+8*h)*np.sqrt(radiusFactor):
            if cellSideLength > 8*h:
#                 rprint(rank, "Third annulus cell size will be ", cellSideLength/2)
                self.divideFlag=True

        else:                                       # cell is in the far field
            pass
        
       
        if False:
            rprint(rank, '\nCell:   ', self.uniqueID)
            rprint(rank, "distance to nearest atom: ",distanceToNearestAtom)
            rprint(rank, "Cell center: ", xdiv,ydiv,zdiv)
            rprint(rank, "Cell side length: ",cellSideLength)
            rprint(rank, "Cell dividing: ", self.divideFlag)
            rprint(rank, " ")
            
            
    def refineCoarseningUniform_TwoLevel(self, h, H, r, level):
        self.divideFlag = False
        self.fineMesh = False
        
        xdiv = (self.xmax + self.xmin)/2   
        ydiv = (self.ymax + self.ymin)/2   
        zdiv = (self.zmax + self.zmin)/2 
        
        # how many coarsening steps are required:
        steps = int( np.log2(H/h))
        
        # determine distance from cell corner to nearest atom
        distanceToNearestAtom = 10+H
        for atom in self.tree.atoms:
            if ( (atom.x<self.xmax) and (atom.y<self.ymax) and (atom.z<self.zmax) and
                 (atom.x>self.xmin) and (atom.y>self.ymin) and (atom.z>self.zmin) ): # atom is in cell
                distanceToNearestAtom=0
            else:
                for x in [self.xmax,self.xmin]:
                    for y in [self.ymin,self.ymax]:
                        for z in [self.zmin,self.zmax]:
                            d = np.sqrt( (x-atom.x)**2 + (y-atom.y)**2 + (z-atom.z)**2)
#                             d = np.max(  [abs(x-atom.x), abs(y-atom.y), abs(z-atom.z)])
                            distanceToNearestAtom = min(distanceToNearestAtom,d)
        
        # compute cell size
#         cellRadius = np.sqrt( (self.xmax-xdiv)**2 + (self.ymax-ydiv)**2 + (self.zmax-zdiv)**2 )
        cellSideLength = np.max( [self.zmax-self.zmin, self.xmax-self.xmin, self.ymax-self.ymin ] )
        
        radiusFactor=1
#         if level>=3: radiusFactor=2
#         if level>=4: radiusFactor=3
        
        
        # DETERMINE IF THIS CELL SHOULD REFINE
        if cellSideLength>H:
            self.divideFlag=True
            
        elif ( (level>=2) and (distanceToNearestAtom<(r/4)) ):  
#             rprint(rank, "Second check: distance < r/4")
            if cellSideLength > (h/4):
                rprint(rank, "First check: New inner cell size will be ", cellSideLength/2)
                self.divideFlag=True
        elif ( (level>=1) and (distanceToNearestAtom<(r/2)) ): 
            if cellSideLength > (h/2):
                self.divideFlag=True
                rprint(rank, "Second check: New inner cell size will be ", cellSideLength/2)
        elif distanceToNearestAtom<r:  #if in the inner ring
#         elif distanceToNearestAtom<2:  #if in the inner ring
            if cellSideLength > h:
#                 rprint(rank, "New inner cell size will be ", cellSideLength/2)
                self.divideFlag=True
                
        elif distanceToNearestAtom<r+2*h*np.sqrt(radiusFactor):  # if in the inner ring
#         elif distanceToNearestAtom<3:  # if in the inner ring
            if cellSideLength > 2*h:
#                 rprint(rank, "First annulus cell size will be ", cellSideLength/2)
                self.divideFlag=True
        
        elif distanceToNearestAtom<r+(2*h+4*h)*np.sqrt(radiusFactor):  # if in the inner ring
#         elif distanceToNearestAtom<12:  # if in the inner ring
            if cellSideLength > 4*h:
#                 rprint(rank, "Second annulus cell size will be ", cellSideLength/2)
                self.divideFlag=True
        
        elif distanceToNearestAtom<r+(2*h+4*h+8*h)*np.sqrt(radiusFactor):
            if cellSideLength > 8*h:
#                 rprint(rank, "Third annulus cell size will be ", cellSideLength/2)
                self.divideFlag=True

        else:                                       # cell is in the far field
            pass
        
        # NOW, IF self.divideFlag STILL = FALSE, THEN THIS CELL IS GOING TO BE PART OF THE MESH.  
        # NEXT, CHECK IF WE WANT TO REFINE IT AGAIN FOR THE FINE MESH.
        if self.divideFlag==False:  # this cell is not refining.  It will be part of the coarse mesh.
            
            
            
#             if level==-1: # refine whole inner ball to 1/2 mesh spacing
            if level<-1: # refine depending on level
#                 rprint(rank, "TRIGGERED LEVEL==-1, exiting")
#                 exit(-1)


                if distanceToNearestAtom<r*1.5:
#                 if distanceToNearestAtom<r:
                    
#                     rprint(rank, "Creating fine mesh for cell a distance %f from atom." %distanceToNearestAtom)
                    
                    self.fineMesh = True
                    numX=-int(level)
                    numY=-int(level)
                    numZ=-int(level) 
                    
                    self.numFinePoints = numX*numY*numZ*(self.px+1)*(self.py+1)*(self.pz+1)
                    self.PxfByPyfByPzf = [element for element in itertools.product(range(numX*(self.px+1)),range(numY*(self.py+1)),range(numZ*(self.pz+1)))]
                    
                    
                    fine_gridpoints = np.empty(( (self.px+1)*numX,(self.py+1)*numY,(self.pz+1)*numZ),dtype=object)
                    fine_weights = np.empty(( (self.px+1)*numX,(self.py+1)*numY,(self.pz+1)*numZ),dtype=object)
                    for i in range(numX):
                        
                        dx = (self.xmax - self.xmin)/numX
                        
                        xlow  = self.xmin + (i+0)*dx
                        xhigh = self.xmin + (i+1)*dx
                        
                        for j in range(numY):
                            
                            dy = (self.ymax - self.ymin)/numY
                            
                            ylow  = self.ymin + (j+0)*dy
                            yhigh = self.ymin + (j+1)*dy
                            
                            for k in range(numZ):
                                
                                dz = (self.zmax-self.zmin)/numZ
                                
                                zlow  = self.zmin + (k+0)*dz
                                zhigh = self.zmin + (k+1)*dz
                            
                    
                                # generate points in [xlow,xhigh]x[ylow,yhigh]x[zlow,zhigh]
                                
#                                 fine_weights.append( weights3DFirstKind(xlow, xhigh, self.px, ylow, yhigh, self.py, zlow, zhigh, self.pz, self.unscaledW) )
                                
                                fine_weights[i*(self.px+1):(i+1)*(self.px+1), j*(self.py+1):(j+1)*(self.py+1), k*(self.pz+1):(k+1)*(self.pz+1)] = weights3DFirstKind(xlow, xhigh, self.px, ylow, yhigh, self.py, zlow, zhigh, self.pz, self.unscaledW) 
                                xvec = ChebyshevPointsFirstKind(xlow, xhigh, self.px)
                                yvec = ChebyshevPointsFirstKind(ylow, yhigh, self.py)
                                zvec = ChebyshevPointsFirstKind(zlow, zhigh, self.pz)
                                
                                for ii in range(self.px+1):
                                    for jj in range(self.py+1):
                                        for kk in range(self.pz+1):
                                            fine_gridpoints[i*(self.px+1) + ii, j*(self.py+1) + jj, k*(self.pz+1) + kk] = GridPoint(xvec[ii],yvec[jj],zvec[kk], 0.0, None, None, 0, initPotential=False)
                                
                    self.fine_gridpoints = fine_gridpoints
                    self.wf = fine_weights
                    
#                     rprint(rank, "Shape of self.wf: ", np.shape(self.wf))
#                     rprint(rank, "self.wf = ", self.wf)
                    
#                     rprint(rank,  "np.sum(self.wf) = ", np.sum(self.wf))
#                     rprint(rank,  "np.sum(self.w) = ", np.sum(self.w))
#                     exit(-1)

                    assert abs(np.sum(self.wf)-np.sum(self.w) )/np.sum(self.wf) < 1e-12, "Fine and Coarse mesh weights not summing to same value."
                        
                    
                                
                
                
        
    
    """
    DIVISION FUNCTIONS
    """
    def cleanupAfterDivide(self):
        self.w = None
        self.gridpoints = None
        self.PxByPyByPz = None
    
    def interpolator(self, x,y,z,f): # generate the interpolating polynomial for function func.  Must be of shape [px,py,pz]
        
        wx = np.ones(self.px)
        for i in range(self.px):
            wx[i] = (-1)**i * np.sin(  (2*i+1)*np.pi / (2*(self.px-1)+2)  )
        
        wy = np.ones(self.py)
        for j in range(self.py):
            wy[j] = (-1)**j * np.sin(  (2*j+1)*np.pi / (2*(self.py-1)+2)  )
            
        wz = np.ones(self.pz)
        for k in range(self.pz):
            wz[k] = (-1)**k * np.sin(  (2*k+1)*np.pi / (2*(self.pz-1)+2)  )
        
        def P(xt,yt,zt):  # 3D interpolator.  
            
            num = 0
            for i in range(self.px):
                numY = 0
                for j in range(self.py):
                    numZ = 0
                    for k in range(self.pz):
                        numZ += ( wz[k]/(zt-z[k])*f[i,j,k] )
                    numY += ( wy[j]/(yt-y[j])*numZ )
                num +=  ( wx[i]/(xt-x[i]) )*numY
            
            denX=0
            for i in range(self.px):
                denX += wx[i]/(xt-x[i])
            
            denY=0
            for j in range(self.py):
                denY += wy[j]/(yt-y[j])
                
            denZ=0
            for k in range(self.pz):
                denZ += wz[k]/(zt-z[k])
            
            den = denX*denY*denZ
                
            return num/den
    
        return np.vectorize(P)
        
        
#         self.interpolator = RegularGridInterpolator((xvec, yvec, zvec), phiCoarse,method='nearest') 
    
    def divide_secondKind(self, xdiv, ydiv, zdiv, printNumberOfCells=False, interpolate=False, temporaryCell=False):
           
           
        def divideInto8_secondKind_atomAtCorner(cell, xdiv, ydiv, zdiv, printNumberOfCells=False, interpolate=False):
            '''setup pxXpyXpz array of gridpoint objects.  These will be used to construct the 8 children cells'''
            
            children = np.empty((2,2,2), dtype=object)
            self.leaf = False
            
#             rprint(rank, self.atomAtCorner)
#             rprint(rank, self.atomAtCorner[0])
            if self.atomAtCorner[0]=='1':
                x_first = ChebyshevPointsFirstKind(float(xdiv),cell.xmax,cell.px)
            elif self.atomAtCorner[0]=='0':
                x_first = ChebyshevPointsFirstKind(cell.xmin,float(xdiv),cell.px)
            else:
                rprint(rank, 'What corner is atom at in x direction? ', self.atomAtCorner[0])
             
            if self.atomAtCorner[1]=='1':
                y_first = ChebyshevPointsFirstKind(float(ydiv),cell.ymax,cell.py)
            elif self.atomAtCorner[1]=='0':
                y_first = ChebyshevPointsFirstKind(cell.ymin,float(ydiv),cell.py)
            else:
                rprint(rank, 'What corner is atom at in y direction? ', self.atomAtCorner[1])
             
            if self.atomAtCorner[2]=='1':
                z_first = ChebyshevPointsFirstKind(float(zdiv),cell.zmax,cell.pz)
            elif self.atomAtCorner[2]=='0':
                z_first = ChebyshevPointsFirstKind(cell.zmin,float(zdiv),cell.pz)
            else: 
                rprint(rank, 'What corner is atom at in z direction? ', self.atomAtCorner[2])

            x = [ChebyshevPointsSecondKind(cell.xmin,float(xdiv),cell.px), ChebyshevPointsSecondKind(float(xdiv),cell.xmax,cell.px)]
            y = [ChebyshevPointsSecondKind(cell.ymin,float(ydiv),cell.py), ChebyshevPointsSecondKind(float(ydiv),cell.ymax,cell.py)]
            z = [ChebyshevPointsSecondKind(cell.zmin,float(zdiv),cell.pz), ChebyshevPointsSecondKind(float(zdiv),cell.zmax,cell.pz)]
            xbounds = np.array([cell.xmin, float(xdiv), cell.xmax])
            ybounds = np.array([cell.ymin, float(ydiv), cell.ymax])
            zbounds = np.array([cell.zmin, float(zdiv), cell.zmax])
    
            '''call the cell constructor for the children.  Set up parent, uniqueID, neighbor list.  Append to masterList'''
            
            for i, j, k in TwoByTwoByTwo:
                if hasattr(cell, "tree"):
                    childKind = 'second'
                    if str(i)+str(j)+str(k)==self.atomAtCorner:
                        childKind='first'
                        rprint(rank, 'Atom at corner %s of parent %s, making %i %i %i child of first kind' %(self.atomAtCorner, self.uniqueID,i,j,k ))
                    children[i,j,k] = Cell(childKind, xbounds[i], xbounds[i+1], cell.px, 
                                           ybounds[j], ybounds[j+1], cell.py,
                                           zbounds[k], zbounds[k+1], cell.pz, tree = cell.tree)
                    if childKind=='first':
                        children[i,j,k].atomAtCorner = self.atomAtCorner
                
                children[i,j,k].parent = cell # children should point to their parent
                if hasattr(cell, "childrenRefineCause"):
                    children[i,j,k].refineCause = cell.childrenRefineCause
                if hasattr(cell, "uniqueID"):
                    children[i,j,k].setUniqueID(i,j,k)
                    children[i,j,k].setNeighborList()
                if temporaryCell==False:   
                    if hasattr(cell, "tree"):
                        if hasattr(cell.tree, 'masterList'):
                            cell.tree.masterList.insert(bisect.bisect_left(cell.tree.masterList, [children[i,j,k].uniqueID,]), [children[i,j,k].uniqueID,children[i,j,k]])
    
            '''create new gridpoints wherever necessary.  Also create density points. '''
            newGridpointCount=0
            for ii,jj,kk in TwoByTwoByTwo:
                xOct = x[ii]
                yOct = y[jj]
                zOct = z[kk]
                if str(ii)+str(jj)+str(kk)==self.atomAtCorner:
                     xOct=x_first
                     yOct=y_first
                     zOct=z_first
                gridpoints = np.empty((cell.px+1,cell.py+1,cell.pz+1),dtype=object)
                for i, j, k in cell.PxByPyByPz:
                    newGridpointCount += 1
#                     gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k],self.nOrbitals, self.tree.gaugeShift, self.tree.atoms, self.tree.coreRepresentation, self.tree.nOrbitals)
                    gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k], self.tree.gaugeShift, self.tree.atoms, self.tree.coreRepresentation, self.tree.nOrbitals)
                    if interpolate == True:
                        for m in range(self.nOrbitals):
                            gridpoints[i,j,k].setPhi(interpolators[m](xOct[i],yOct[j],zOct[k]),m)
                children[ii,jj,kk].setGridpoints(gridpoints)
                if hasattr(cell,'level'):
                    children[ii,jj,kk].level = cell.level+1
                else:
                    rprint(rank, 'Warning: cell ',cell.uniqueID, ' does not have attribute level.')
                    
                    
#             for ii,jj,kk in TwoByTwoByTwo:
#                 xOct = x_density[ii]
#                 yOct = y_density[jj]
#                 zOct = z_density[kk]   
#                 densityPoints = np.empty((cell.pxd,cell.pyd,cell.pzd),dtype=object)
#                 for i, j, k in cell.PxByPyByPz_density:
#                     densityPoints[i,j,k] = DensityPoint(xOct[i],yOct[j],zOct[k])
#                 children[ii,jj,kk].setDensityPoints(densityPoints)
                
            
            if printNumberOfCells == True: rprint(rank, 'generated %i new gridpoints for parent cell %s' %(newGridpointCount, cell.uniqueID))
    
            '''set the parent cell's 'children' attribute to the array of children'''
            cell.children = children
            
                   
        def divideInto8_secondKind(cell, xdiv, ydiv, zdiv, printNumberOfCells=False, interpolate=False):
            '''setup pxXpyXpz array of gridpoint objects.  These will be used to construct the 8 children cells'''
            
            children = np.empty((2,2,2), dtype=object)
            self.leaf = False
#             self.nOrbitals = self.gridpoints[0,0,0].nOrbitals
            
#             if interpolate==True:
#                 # generate the x, y, and z arrays
#                 x = np.empty(self.px)
#                 y = np.empty(self.py)
#                 z = np.empty(self.pz)
#                 for i in range(self.px):
#                     x[i] = self.gridpoints[i,0,0].x
#                 for j in range(self.py):
#                     y[j] = self.gridpoints[0,j,0].y
#                 for k in range(self.pz):
#                     z[k] = self.gridpoints[0,0,k].z
#                     
#                 
#                 # Generate interpolators for each orbital
#                 self.nOrbitals = len(self.gridpoints[0,0,0].phi)
#                 interpolators = np.empty(self.nOrbitals,dtype=object)
#                 phi = np.zeros((self.px,self.py,self.pz,self.nOrbitals))
#                 for i,j,k in self.PxByPyByPz:
#                     for m in range(self.nOrbitals):
#                         phi[i,j,k,m] = self.gridpoints[i,j,k].phi[m]
#                 
#                 for m in range(self.nOrbitals):
#                     interpolators[m] = self.interpolator(x, y, z, phi[:,:,:,m])
                    
                    
        
            x = [ChebyshevPointsSecondKind(cell.xmin,float(xdiv),cell.px), ChebyshevPointsSecondKind(float(xdiv),cell.xmax,cell.px)]
            y = [ChebyshevPointsSecondKind(cell.ymin,float(ydiv),cell.py), ChebyshevPointsSecondKind(float(ydiv),cell.ymax,cell.py)]
            z = [ChebyshevPointsSecondKind(cell.zmin,float(zdiv),cell.pz), ChebyshevPointsSecondKind(float(zdiv),cell.zmax,cell.pz)]

            
            xbounds = np.array([cell.xmin, float(xdiv), cell.xmax])
            ybounds = np.array([cell.ymin, float(ydiv), cell.ymax])
            zbounds = np.array([cell.zmin, float(zdiv), cell.zmax])
    
            '''call the cell constructor for the children.  Set up parent, uniqueID, neighbor list.  Append to masterList'''
            
            for i, j, k in TwoByTwoByTwo:
                if hasattr(cell, "tree"):
                    children[i,j,k] = Cell('second', xbounds[i], xbounds[i+1], cell.px, 
                                           ybounds[j], ybounds[j+1], cell.py,
                                           zbounds[k], zbounds[k+1], cell.pz, tree = cell.tree)
                else:
                    children[i,j,k] = Cell('second', xbounds[i], xbounds[i+1], cell.px, 
                                           ybounds[j], ybounds[j+1], cell.py,
                                           zbounds[k], zbounds[k+1], cell.pz)
                children[i,j,k].parent = cell # children should point to their parent
                if hasattr(cell, "childrenRefineCause"):
                    children[i,j,k].refineCause = cell.childrenRefineCause
                if hasattr(cell, "uniqueID"):
                    children[i,j,k].setUniqueID(i,j,k)
                    children[i,j,k].setNeighborList()
                if temporaryCell==False:   
                    if hasattr(cell, "tree"):
                        if hasattr(cell.tree, 'masterList'):
                            cell.tree.masterList.insert(bisect.bisect_left(cell.tree.masterList, [children[i,j,k].uniqueID,]), [children[i,j,k].uniqueID,children[i,j,k]])
    
            '''create new gridpoints wherever necessary.  Also create density points. '''
            newGridpointCount=0
            for ii,jj,kk in TwoByTwoByTwo:
                xOct = x[ii]
                yOct = y[jj]
                zOct = z[kk]
                gridpoints = np.empty((cell.px+1,cell.py+1,cell.pz+1),dtype=object)
                for i, j, k in cell.PxByPyByPz:
                    newGridpointCount += 1
#                     gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k],self.nOrbitals, self.tree.gaugeShift, self.tree.atoms, self.tree.coreRepresentation, self.tree.nOrbitals, self.tree.nOrbitals)
                    gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k], self.tree.gaugeShift, self.tree.atoms, self.tree.coreRepresentation, self.tree.nOrbitals, self.tree.nOrbitals)
#                     if interpolate == True:
#                         for m in range(self.nOrbitals):
#                             gridpoints[i,j,k].setPhi(interpolators[m](xOct[i],yOct[j],zOct[k]),m)
                children[ii,jj,kk].setGridpoints(gridpoints)
                if hasattr(cell,'level'):
                    children[ii,jj,kk].level = cell.level+1
                else:
                    rprint(rank, 'Warning: cell ',cell.uniqueID, ' does not have attribute level.')
                    
                    
#             for ii,jj,kk in TwoByTwoByTwo:
#                 xOct = x_density[ii]
#                 yOct = y_density[jj]
#                 zOct = z_density[kk]   
#                 densityPoints = np.empty((cell.pxd,cell.pyd,cell.pzd),dtype=object)
#                 for i, j, k in cell.PxByPyByPz_density:
#                     densityPoints[i,j,k] = DensityPoint(xOct[i],yOct[j],zOct[k])
#                 children[ii,jj,kk].setDensityPoints(densityPoints)
                
            
            if printNumberOfCells == True: rprint(rank, 'generated %i new gridpoints for parent cell %s' %(newGridpointCount, cell.uniqueID))
    
            '''set the parent cell's 'children' attribute to the array of children'''
            cell.children = children
        
        def divideInto4_secondKind(cell, xdiv, ydiv, zdiv, printNumberOfCells=False):
            '''setup pxXpyXpz array of gridpoint objects.  These will be used to construct the 8 children cells'''
            if zdiv == None:
                children = np.empty((2,2,1), dtype=object)
                cell.leaf = False
        
                x = [ChebyshevPointsSecondKind(cell.xmin,float(xdiv),cell.px), ChebyshevPointsSecondKind(float(xdiv),cell.xmax,cell.px)]
                y = [ChebyshevPointsSecondKind(cell.ymin,float(ydiv),cell.py), ChebyshevPointsSecondKind(float(ydiv),cell.ymax,cell.py)]
                z = [ChebyshevPointsSecondKind(cell.zmin,cell.zmax,cell.pz)]
                
                xbounds = np.array([cell.xmin, float(xdiv), cell.xmax])
                ybounds = np.array([cell.ymin, float(ydiv), cell.ymax])        

                '''call the cell constructor for the children.  Set up parent, uniqueID, neighbor list.  Append to masterList'''
                for i, j in TwoByTwo:
                    
                    children[i,j,0] = Cell('second',xbounds[i], xbounds[i+1], cell.px, 
                                           ybounds[j], ybounds[j+1], cell.py,
                                           cell.zmin, cell.zmax, cell.pz, tree = cell.tree)
                    children[i,j,0].parent = cell # children should point to their parent
                    if hasattr(cell, "childrenRefineCause"):
                        children[i,j,0].refineCause = cell.childrenRefineCause
                    children[i,j,0].setUniqueID(i,j,0)  # neighbor lists going to be ruined once no longer dividing into 8
                    children[i,j,0].setNeighborList()
        
                    if temporaryCell==False:
                        if hasattr(cell.tree, 'masterList'):
                            cell.tree.masterList.insert(bisect.bisect_left(cell.tree.masterList, [children[i,j,0].uniqueID,]), [children[i,j,0].uniqueID,children[i,j,0]])
        
                '''create new gridpoints wherever necessary'''
                newGridpointCount=0
                for ii,jj in TwoByTwo:
                    xOct = x[ii]
                    yOct = y[jj]
                    zOct = z[0]
                    gridpoints = np.empty((cell.px+1,cell.py+1,cell.pz+1),dtype=object)
                    for i, j, k in cell.PxByPyByPz:
                        newGridpointCount += 1
                        gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k], self.tree.gaugeShift, self.tree.atoms, self.tree.coreRepresentation, self.tree.nOrbitals)
#                         gridpoints[i,j,k].setExternalPotential(cell.tree.atoms, cell.tree.gaugeShift, self.tree.nOrbitals)
                    children[ii,jj,0].setGridpoints(gridpoints)
                    if hasattr(cell,'level'):
                        children[ii,jj,0].level = cell.level+1
                        
            elif ydiv == None:  # divideInto8 along the x and z axes, but not the y
                children = np.empty((2,1,2), dtype=object)
                cell.leaf = False
            
                x = [ChebyshevPointsSecondKind(cell.xmin,float(xdiv),cell.px), ChebyshevPointsSecondKind(float(xdiv),cell.xmax,cell.px)]
                y = [ChebyshevPointsSecondKind(cell.ymin,cell.ymax,cell.py)]
                z = [ChebyshevPointsSecondKind(cell.zmin,float(zdiv),cell.pz), ChebyshevPointsSecondKind(float(zdiv),cell.zmax,cell.pz)]
                
                xbounds = np.array([cell.xmin, float(xdiv), cell.xmax])
                zbounds = np.array([cell.zmin, float(zdiv), cell.zmax])
        
                '''call the cell constructor for the children.  Set up parent, uniqueID, neighbor list.  Append to masterList'''
                for i, k in TwoByTwo:
                    
                    children[i,0,k] = Cell('second',xbounds[i], xbounds[i+1], cell.px, 
                                           cell.ymin, cell.ymax, cell.py,
                                           zbounds[k], zbounds[k+1], cell.pz, tree = cell.tree)
                    children[i,0,k].parent = cell # children should point to their parent
                    if hasattr(cell, "childrenRefineCause"):
                        children[i,0,k].refineCause = cell.childrenRefineCause
                    children[i,0,k].setUniqueID(i,0,k)  # neighbor lists going to be ruined once no longer dividing into 8
                    children[i,0,k].setNeighborList()
        
                    if temporaryCell==False:
                        if hasattr(cell.tree, 'masterList'):
                            cell.tree.masterList.insert(bisect.bisect_left(cell.tree.masterList, [children[i,0,k].uniqueID,]), [children[i,0,k].uniqueID,children[i,0,k]])
        
        
                '''create new gridpoints wherever necessary'''
                newGridpointCount=0
                for ii,kk in TwoByTwo:
                    xOct = x[ii]
                    yOct = y[0]
                    zOct = z[kk]
                    gridpoints = np.empty((cell.px+1,cell.py+1,cell.pz+1),dtype=object)
                    for i, j, k in cell.PxByPyByPz:
                        newGridpointCount += 1
                        gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k], self.tree.gaugeShift, self.tree.atoms, self.tree.coreRepresentation, self.tree.nOrbitals)
#                         gridpoints[i,j,k].setExternalPotential(cell.tree.atoms)
                    children[ii,0,kk].setGridpoints(gridpoints)
                    if hasattr(cell,'level'):
                        children[ii,0,kk].level = cell.level+1
                        
            elif xdiv == None:  # divideInto8 along the x and z axes, but not the y
                children = np.empty((1,2,2), dtype=object)
                cell.leaf = False
            
                x = [ChebyshevPointsSecondKind(cell.xmin,cell.xmax,cell.px)]
                y = [ChebyshevPointsSecondKind(cell.ymin,float(ydiv),cell.py), ChebyshevPointsSecondKind(float(ydiv),cell.ymax,cell.py)]
                z = [ChebyshevPointsSecondKind(cell.zmin,float(zdiv),cell.pz), ChebyshevPointsSecondKind(float(zdiv),cell.zmax,cell.pz)]
                
                ybounds = np.array([cell.ymin, float(ydiv), cell.ymax])
                zbounds = np.array([cell.zmin, float(zdiv), cell.zmax])

                '''call the cell constructor for the children.  Set up parent, uniqueID, neighbor list.  Append to masterList'''
                for j, k in TwoByTwo:
                    
                    children[0,j,k] = Cell('second',cell.xmin, cell.xmax, cell.px, 
                                           ybounds[j], ybounds[j+1], cell.py,
                                           zbounds[k], zbounds[k+1], cell.pz, tree = cell.tree)
                    children[0,j,k].parent = cell # children should point to their parent
                    if hasattr(cell, "childrenRefineCause"):
                        children[0,j,k].refineCause = cell.childrenRefineCause
                    children[0,j,k].setUniqueID(0,j,k)  # neighbor lists going to be ruined once no longer dividing into 8
                    children[0,j,k].setNeighborList()
                    
                    if temporaryCell==False:
                        if hasattr(cell.tree, 'masterList'):
                            cell.tree.masterList.insert(bisect.bisect_left(cell.tree.masterList, [children[0,j,k].uniqueID,]), [children[0,j,k].uniqueID,children[0,j,k]])
        
                '''create new gridpoints wherever necessary'''
                newGridpointCount=0
                for jj,kk in TwoByTwo:
                    xOct = x[0]
                    yOct = y[jj]
                    zOct = z[kk]
                    gridpoints = np.empty((cell.px+1,cell.py+1,cell.pz+1),dtype=object)
                    for i, j, k in cell.PxByPyByPz:
                        newGridpointCount += 1
                        gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k], self.tree.gaugeShift, self.tree.atoms, self.tree.coreRepresentation, self.tree.nOrbitals)
                    children[0,jj,kk].setGridpoints(gridpoints)
                    if hasattr(cell,'level'):
                        children[0,jj,kk].level = cell.level+1
                
            if printNumberOfCells == True: rprint(rank, 'generated %i new gridpoints for parent cell %s' %(newGridpointCount, cell.uniqueID))
        
            '''set the parent cell's 'children' attribute to the array of children'''
            cell.children = children
        
        def divideInto2_secondKind(cell, xdiv, ydiv, zdiv, printNumberOfCells=False):
            
            '''setup pxXpyXpz array of gridpoint objects.  These will be used to construct the 8 children cells'''
            cell.leaf = False
            if ( (zdiv == None) and (ydiv==None) ):
                
                # First check bounds:
                if ( (xdiv < cell.xmin) or (xdiv > cell.xmax) ):
                    rprint(rank, 'WARNING: XDIV NOT IN CELL BOUNDS')
                children = np.empty((2,1,1), dtype=object)
        
                x = [ChebyshevPointsSecondKind(cell.xmin,float(xdiv),cell.px), ChebyshevPointsSecondKind(float(xdiv),cell.xmax,cell.px)]
                y = [ChebyshevPointsSecondKind(cell.ymin,cell.ymax,cell.py)]
                z = [ChebyshevPointsSecondKind(cell.zmin,cell.zmax,cell.pz)]
                
                xbounds = np.array([cell.xmin, float(xdiv), cell.xmax])
        
                '''call the cell constructor for the children.  Set up parent, uniqueID, neighbor list.  Append to masterList'''
                for i in range(2):
                    
                    children[i,0,0] = Cell('second',xbounds[i], xbounds[i+1], cell.px, 
                                           cell.ymin, cell.ymax, cell.py,
                                           cell.zmin, cell.zmax, cell.pz, tree = cell.tree)
                    children[i,0,0].parent = cell # children should point to their parent
                    if hasattr(cell, "childrenRefineCause"):
                        children[i,0,0].refineCause = cell.childrenRefineCause
                    children[i,0,0].setUniqueID(i,0,0)  # neighbor lists going to be ruined once no longer dividing into 8
                    children[i,0,0].setNeighborList()
        
                    if hasattr(cell.tree, 'masterList'):
                        cell.tree.masterList.insert(bisect.bisect_left(cell.tree.masterList, [children[i,0,0].uniqueID,]), [children[i,0,0].uniqueID,children[i,0,0]])
        
                '''create new gridpoints wherever necessary'''
                newGridpointCount=0
                for ii in range(2):
                    xOct = x[ii]
                    yOct = y[0]
                    zOct = z[0]
                    gridpoints = np.empty((cell.px+1,cell.py+1,cell.pz+1),dtype=object)
                    for i, j, k in cell.PxByPyByPz:
                        newGridpointCount += 1
                        gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k], self.tree.gaugeShift, self.tree.atoms, self.tree.coreRepresentation, self.tree.nOrbitals)
#                         gridpoints[i,j,k].setExternalPotential(cell.tree.atoms, cell.tree.gaugeShift)
                    children[ii,0,0].setGridpoints(gridpoints)
                    if hasattr(cell,'level'):
                        children[ii,0,0].level = cell.level+1
#                 rprint(rank, 'Not increasing the cell level because only dividing along x axis.')
                        
            elif ( (zdiv == None) and (xdiv==None) ):  # divide along y axis only
                # First check bounds:
                if ( (ydiv < cell.ymin) or (ydiv > cell.ymax) ):
                    rprint(rank, 'WARNING: YDIV NOT IN CELL BOUNDS')
                    
                children = np.empty((1,2,1), dtype=object)
                cell.leaf = False
        
                x = [ChebyshevPointsSecondKind(cell.xmin,cell.xmax,cell.px)]
                y = [ChebyshevPointsSecondKind(cell.ymin,float(ydiv),cell.py), ChebyshevPointsSecondKind(float(ydiv),cell.ymax,cell.py)]
                z = [ChebyshevPointsSecondKind(cell.zmin,cell.zmax,cell.pz)]
                
                ybounds = np.array([cell.ymin, float(ydiv), cell.ymax])
        
                '''call the cell constructor for the children.  Set up parent, uniqueID, neighbor list.  Append to masterList'''
                for j in range(2):
                    
                    children[0,j,0] = Cell('second',cell.xmin, cell.xmax, cell.px, 
                                           ybounds[j], ybounds[j+1], cell.py,
                                           cell.zmin, cell.zmax, cell.pz, tree = cell.tree)
                    children[0,j,0].parent = cell # children should point to their parent
                    if hasattr(cell, "childrenRefineCause"):
                        children[0,j,0].refineCause = cell.childrenRefineCause
                    children[0,j,0].setUniqueID(0,j,0)  # neighbor lists going to be ruined once no longer dividing into 8
                    children[0,j,0].setNeighborList()
        
                    if hasattr(cell.tree, 'masterList'):
                        cell.tree.masterList.insert(bisect.bisect_left(cell.tree.masterList, [children[0,j,0].uniqueID,]), [children[0,j,0].uniqueID,children[0,j,0]])
        
                '''create new gridpoints wherever necessary'''
                newGridpointCount=0
                for jj in range(2):
                    xOct = x[0]
                    yOct = y[jj]
                    zOct = z[0]
                    gridpoints = np.empty((cell.px+1,cell.py+1,cell.pz+1),dtype=object)
                    for i, j, k in cell.PxByPyByPz:
                        newGridpointCount += 1
                        gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k], self.tree.gaugeShift, self.tree.atoms, self.tree.coreRepresentation, self.tree.nOrbitals)
#                         gridpoints[i,j,k].setExternalPotential(cell.tree.atoms, cell.tree.gaugeShift)
                    children[0,jj,0].setGridpoints(gridpoints)
                    if hasattr(cell,'level'):
                        children[0,jj,0].level = cell.level+1
#                 rprint(rank, 'Not increasing the cell level because only dividing along y axis.')

                        
            elif ( (xdiv == None) and (ydiv==None) ):  # divide along z axis only
                # First check bounds:
                if ( (zdiv < cell.zmin) or (zdiv > cell.zmax) ):
                    rprint(rank, 'WARNING: ZDIV NOT IN CELL BOUNDS')
                    
                children = np.empty((1,1,2), dtype=object)
                cell.leaf = False
        
                x = [ChebyshevPointsSecondKind(cell.xmin,cell.xmax,cell.px)]
                y = [ChebyshevPointsSecondKind(cell.ymin,cell.ymax,cell.py)]
                z = [ChebyshevPointsSecondKind(cell.zmin,float(zdiv),cell.pz), ChebyshevPointsSecondKind(float(zdiv),cell.zmax,cell.pz)]
                
                zbounds = np.array([cell.zmin, float(zdiv), cell.zmax])

                '''call the cell constructor for the children.  Set up parent, uniqueID, neighbor list.  Append to masterList'''
                for k in range(2):
                    children[0,0,k] = Cell('second',cell.xmin, cell.xmax, cell.px, 
                                           cell.ymin, cell.ymax, cell.py,
                                           zbounds[k], zbounds[k+1], cell.pz, tree = cell.tree)
                    children[0,0,k].parent = cell # children should point to their parent
                    if hasattr(cell, "childrenRefineCause"):
                        children[0,0,k].refineCause = cell.childrenRefineCause
                    children[0,0,k].setUniqueID(0,0,k)  # neighbor lists going to be ruined once no longer dividing into 8
                    children[0,0,k].setNeighborList()
        
                    if hasattr(cell.tree, 'masterList'):
                        cell.tree.masterList.insert(bisect.bisect_left(cell.tree.masterList, [children[0,0,k].uniqueID,]), [children[0,0,k].uniqueID,children[0,0,k]])
        
                '''create new gridpoints wherever necessary'''
                newGridpointCount=0
                for kk in range(2):
                    xOct = x[0]
                    yOct = y[0]
                    zOct = z[kk]
                    gridpoints = np.empty((cell.px+1,cell.py+1,cell.pz+1),dtype=object)
                    for i, j, k in cell.PxByPyByPz:
                        newGridpointCount += 1
                        gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k], self.tree.gaugeShift, self.tree.atoms, self.tree.coreRepresentation, self.tree.nOrbitals)
#                         gridpoints[i,j,k].setExternalPotential(cell.tree.atoms, cell.tree.gaugeShift)
                    children[0,0,kk].setGridpoints(gridpoints)
                    if hasattr(cell,'level'):
                        children[0,0,kk].level = cell.level+1
#                 rprint(rank, 'Not increasing the cell level because only dividing along z axis.')

                        
            if printNumberOfCells == True: rprint(rank, 'generated %i new gridpoints for parent cell %s' %(newGridpointCount, cell.uniqueID))
        
            '''set the parent cell's 'children' attribute to the array of children'''
            cell.children = children
            
        if self.leaf == False:
            rprint(rank, 'Why are you dividing a non-leaf cell?')
        
        noneCount = 0
        if xdiv == None: noneCount += 1
        if ydiv == None: noneCount += 1
        if zdiv == None: noneCount += 1
        
        
#         rprint(rank, self.atomAtCorner)
        if self.atomAtCorner!=False:
            rprint(rank, 'Using divideInto8_secondKind_atomAtCorner')
            divideInto8_secondKind_atomAtCorner(self, xdiv, ydiv, zdiv, printNumberOfCells)
        elif noneCount == 0:
            divideInto8_secondKind(self, xdiv, ydiv, zdiv, printNumberOfCells, interpolate)
        elif noneCount == 1:
#             rprint(rank, 'Using divideInto4... are you sure?')
            divideInto4_secondKind(self, xdiv, ydiv, zdiv, printNumberOfCells) 
        elif noneCount == 2:
#             rprint(rank, 'Using divideInto2... are you sure?')
            divideInto2_secondKind(self, xdiv, ydiv, zdiv, printNumberOfCells)
        elif noneCount == 3:
            rprint(rank, 'Not acutally dividing because xdiv=ydiv=zdiv=None.  Happens when trying to divide at a nucleus that is already at a vertex.')
            
            
    def divide_firstKind(self, xdiv, ydiv, zdiv, printNumberOfCells=False, interpolate=False, temporaryCell=False):
                  
        def divideInto8_firstKind(cell, xdiv, ydiv, zdiv, printNumberOfCells=False, interpolate=False):
            '''setup pxXpyXpz array of gridpoint objects.  These will be used to construct the 8 children cells'''
            
            children = np.empty((2,2,2), dtype=object)
            self.leaf = False
            
                    
        
            x = [ChebyshevPointsFirstKind(cell.xmin,float(xdiv),cell.px), ChebyshevPointsFirstKind(float(xdiv),cell.xmax,cell.px)]
            y = [ChebyshevPointsFirstKind(cell.ymin,float(ydiv),cell.py), ChebyshevPointsFirstKind(float(ydiv),cell.ymax,cell.py)]
            z = [ChebyshevPointsFirstKind(cell.zmin,float(zdiv),cell.pz), ChebyshevPointsFirstKind(float(zdiv),cell.zmax,cell.pz)]
            
            xf = [ChebyshevPointsFirstKind(cell.xmin,float(xdiv),cell.pxf), ChebyshevPointsFirstKind(float(xdiv),cell.xmax,cell.pxf)]
            yf = [ChebyshevPointsFirstKind(cell.ymin,float(ydiv),cell.pyf), ChebyshevPointsFirstKind(float(ydiv),cell.ymax,cell.pyf)]
            zf = [ChebyshevPointsFirstKind(cell.zmin,float(zdiv),cell.pzf), ChebyshevPointsFirstKind(float(zdiv),cell.zmax,cell.pzf)]
            
         
            xbounds = np.array([cell.xmin, float(xdiv), cell.xmax])
            ybounds = np.array([cell.ymin, float(ydiv), cell.ymax])
            zbounds = np.array([cell.zmin, float(zdiv), cell.zmax])
    
            '''call the cell constructor for the children.  Set up parent, uniqueID, neighbor list.  Append to masterList'''
            
            for i, j, k in TwoByTwoByTwo:
                if hasattr(cell, "tree"):
                    children[i,j,k] = Cell('first', xbounds[i], xbounds[i+1], cell.px, 
                                           ybounds[j], ybounds[j+1], cell.py,
                                           zbounds[k], zbounds[k+1], cell.pz, fine_p=self.pxf, tree = cell.tree)
                else:
                    children[i,j,k] = Cell('first', xbounds[i], xbounds[i+1], cell.px, 
                                           ybounds[j], ybounds[j+1], cell.py,
                                           zbounds[k], zbounds[k+1], cell.pz, fine_p=self.pxf,)
                children[i,j,k].parent = cell # children should point to their parent
                if hasattr(cell, "childrenRefineCause"):
                    children[i,j,k].refineCause = cell.childrenRefineCause
                if hasattr(cell, "uniqueID"):
                    children[i,j,k].setUniqueID(i,j,k)
                    children[i,j,k].setNeighborList()
                if temporaryCell==False:   
                    if hasattr(cell, "tree"):
                        if hasattr(cell.tree, 'masterList'):
                            cell.tree.masterList.insert(bisect.bisect_left(cell.tree.masterList, [children[i,j,k].uniqueID,]), [children[i,j,k].uniqueID,children[i,j,k]])
    
            '''create new gridpoints wherever necessary.  Also create density points. '''
            newGridpointCount=0
            for ii,jj,kk in TwoByTwoByTwo:
                xOct = x[ii]
                yOct = y[jj]
                zOct = z[kk]
                xOctf = xf[ii]
                yOctf = yf[jj]
                zOctf = zf[kk]
                
                gridpoints = np.empty((cell.px+1,cell.py+1,cell.pz+1),dtype=object)
                fine_gridpoints = np.empty((cell.pxf+1,cell.pyf+1,cell.pzf+1),dtype=object)
                for i, j, k in cell.PxByPyByPz:
                    newGridpointCount += 1
                    gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k], self.tree.gaugeShift, self.tree.atoms, self.tree.coreRepresentation, self.tree.nOrbitals)
                
                for i, j, k in cell.PxfByPyfByPzf:
                    fine_gridpoints[i,j,k] = GridPoint(xOctf[i],yOctf[j],zOctf[k], self.tree.gaugeShift, self.tree.atoms, self.tree.coreRepresentation, self.tree.nOrbitals)

                children[ii,jj,kk].setGridpoints(gridpoints)
                children[ii,jj,kk].setFineGridpoints(fine_gridpoints)
                
                if hasattr(cell,'level'):
                    children[ii,jj,kk].level = cell.level+1
                else:
                    rprint(rank, 'Warning: cell ',cell.uniqueID, ' does not have attribute level.')
                    
                    
#             for ii,jj,kk in TwoByTwoByTwo:
#                 xOct = x_density[ii]
#                 yOct = y_density[jj]
#                 zOct = z_density[kk]   
#                 densityPoints = np.empty((cell.pxd,cell.pyd,cell.pzd),dtype=object)
#                 for i, j, k in cell.PxByPyByPz_density:
#                     densityPoints[i,j,k] = DensityPoint(xOct[i],yOct[j],zOct[k])
#                 children[ii,jj,kk].setDensityPoints(densityPoints)
                
            
            if printNumberOfCells == True: rprint(rank, 'generated %i new gridpoints for parent cell %s' %(newGridpointCount, cell.uniqueID))
    
            '''set the parent cell's 'children' attribute to the array of children'''
            cell.children = children
        
        def divideInto4_firstKind(cell, xdiv, ydiv, zdiv, printNumberOfCells=False):
            '''setup pxXpyXpz array of gridpoint objects.  These will be used to construct the 8 children cells'''
            if zdiv == None:
                children = np.empty((2,2,1), dtype=object)
                cell.leaf = False
        
                x = [ChebyshevPointsFirstKind(cell.xmin,float(xdiv),cell.px), ChebyshevPointsFirstKind(float(xdiv),cell.xmax,cell.px)]
                y = [ChebyshevPointsFirstKind(cell.ymin,float(ydiv),cell.py), ChebyshevPointsFirstKind(float(ydiv),cell.ymax,cell.py)]
                z = [ChebyshevPointsFirstKind(cell.zmin,cell.zmax,cell.pz)]
                
                xf = [ChebyshevPointsFirstKind(cell.xmin,float(xdiv),cell.pxf), ChebyshevPointsFirstKind(float(xdiv),cell.xmax,cell.pxf)]
                yf = [ChebyshevPointsFirstKind(cell.ymin,float(ydiv),cell.pyf), ChebyshevPointsFirstKind(float(ydiv),cell.ymax,cell.pyf)]
                zf = [ChebyshevPointsFirstKind(cell.zmin,cell.zmax,cell.pzf)]
                
                xbounds = np.array([cell.xmin, float(xdiv), cell.xmax])
                ybounds = np.array([cell.ymin, float(ydiv), cell.ymax])        

                '''call the cell constructor for the children.  Set up parent, uniqueID, neighbor list.  Append to masterList'''
                for i, j in TwoByTwo:
                    
                    children[i,j,0] = Cell('first', xbounds[i], xbounds[i+1], cell.px, 
                                           ybounds[j], ybounds[j+1], cell.py,
                                           cell.zmin, cell.zmax, cell.pz, fine_p=self.pxf, tree = cell.tree)
                    children[i,j,0].parent = cell # children should point to their parent
                    if hasattr(cell, "childrenRefineCause"):
                        children[i,j,0].refineCause = cell.childrenRefineCause
                    children[i,j,0].setUniqueID(i,j,0)  # neighbor lists going to be ruined once no longer dividing into 8
                    children[i,j,0].setNeighborList()
        
                    if temporaryCell==False:
                        if hasattr(cell.tree, 'masterList'):
                            cell.tree.masterList.insert(bisect.bisect_left(cell.tree.masterList, [children[i,j,0].uniqueID,]), [children[i,j,0].uniqueID,children[i,j,0]])
        
                '''create new gridpoints wherever necessary'''
                newGridpointCount=0
                for ii,jj in TwoByTwo:
                    xOct = x[ii]
                    yOct = y[jj]
                    zOct = z[0]
                    
                    xOctf = xf[ii]
                    yOctf = yf[jj]
                    zOctf = zf[0]
                    
                    gridpoints = np.empty((cell.px+1,cell.py+1,cell.pz+1),dtype=object)
                    fine_gridpoints = np.empty((cell.pxf+1,cell.pyf+1,cell.pzf+1),dtype=object)
                    for i, j, k in cell.PxByPyByPz:
                        newGridpointCount += 1
                        gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k], self.tree.gaugeShift, self.tree.atoms, self.tree.coreRepresentation, self.tree.nOrbitals)

                    for i, j, k in cell.PxfByPyfByPzf:
                        fine_gridpoints[i,j,k] = GridPoint(xOctf[i],yOctf[j],zOctf[k], self.tree.gaugeShift, self.tree.atoms, self.tree.coreRepresentation, self.tree.nOrbitals)


                    children[ii,jj,0].setGridpoints(gridpoints)
                    children[ii,jj,0].setFineGridpoints(fine_gridpoints)
                    if hasattr(cell,'level'):
                        children[ii,jj,0].level = cell.level+1
                        
            elif ydiv == None:  # divideInto8 along the x and z axes, but not the y
                children = np.empty((2,1,2), dtype=object)
                cell.leaf = False
            
                x = [ChebyshevPointsFirstKind(cell.xmin,float(xdiv),cell.px), ChebyshevPointsFirstKind(float(xdiv),cell.xmax,cell.px)]
                y = [ChebyshevPointsFirstKind(cell.ymin,cell.ymax,cell.py)]
                z = [ChebyshevPointsFirstKind(cell.zmin,float(zdiv),cell.pz), ChebyshevPointsFirstKind(float(zdiv),cell.zmax,cell.pz)]
                
                xf = [ChebyshevPointsFirstKind(cell.xmin,float(xdiv),cell.pxf), ChebyshevPointsFirstKind(float(xdiv),cell.xmax,cell.pxf)]
                yf = [ChebyshevPointsFirstKind(cell.ymin,cell.ymax,cell.pyf)]
                zf = [ChebyshevPointsFirstKind(cell.zmin,float(zdiv),cell.pzf), ChebyshevPointsFirstKind(float(zdiv),cell.zmax,cell.pzf)]
                
                
                xbounds = np.array([cell.xmin, float(xdiv), cell.xmax])
                zbounds = np.array([cell.zmin, float(zdiv), cell.zmax])
        
                '''call the cell constructor for the children.  Set up parent, uniqueID, neighbor list.  Append to masterList'''
                for i, k in TwoByTwo:
                    
                    children[i,0,k] = Cell('first', xbounds[i], xbounds[i+1], cell.px, 
                                           cell.ymin, cell.ymax, cell.py,
                                           zbounds[k], zbounds[k+1], cell.pz, fine_p=self.pxf, tree = cell.tree)
                    children[i,0,k].parent = cell # children should point to their parent
                    if hasattr(cell, "childrenRefineCause"):
                        children[i,0,k].refineCause = cell.childrenRefineCause
                    children[i,0,k].setUniqueID(i,0,k)  # neighbor lists going to be ruined once no longer dividing into 8
                    children[i,0,k].setNeighborList()
        
                    if temporaryCell==False:
                        if hasattr(cell.tree, 'masterList'):
                            cell.tree.masterList.insert(bisect.bisect_left(cell.tree.masterList, [children[i,0,k].uniqueID,]), [children[i,0,k].uniqueID,children[i,0,k]])
        
        
                '''create new gridpoints wherever necessary'''
                newGridpointCount=0
                for ii,kk in TwoByTwo:
                    xOct = x[ii]
                    yOct = y[0]
                    zOct = z[kk]
                    
                    xOctf = xf[ii]
                    yOctf = yf[0]
                    zOctf = zf[kk]
                    
                    gridpoints = np.empty((cell.px+1,cell.py+1,cell.pz+1),dtype=object)
                    fine_gridpoints = np.empty((cell.pxf+1,cell.pyf+1,cell.pzf+1),dtype=object)
                    for i, j, k in cell.PxByPyByPz:
                        newGridpointCount += 1
                        gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k], self.tree.gaugeShift, self.tree.atoms, self.tree.coreRepresentation, self.tree.nOrbitals)
                    
                    for i, j, k in cell.PxfByPyfByPzf:
                        fine_gridpoints[i,j,k] = GridPoint(xOctf[i],yOctf[j],zOctf[k], self.tree.gaugeShift, self.tree.atoms, self.tree.coreRepresentation, self.tree.nOrbitals)
                    
                    children[ii,0,kk].setGridpoints(gridpoints)
                    children[ii,0,kk].setFineGridpoints(fine_gridpoints)
                    if hasattr(cell,'level'):
                        children[ii,0,kk].level = cell.level+1
                        
            elif xdiv == None:  # divideInto8 along the x and z axes, but not the y
                children = np.empty((1,2,2), dtype=object)
                cell.leaf = False
            
                x = [ChebyshevPointsFirstKind(cell.xmin,cell.xmax,cell.px)]
                y = [ChebyshevPointsFirstKind(cell.ymin,float(ydiv),cell.py), ChebyshevPointsFirstKind(float(ydiv),cell.ymax,cell.py)]
                z = [ChebyshevPointsFirstKind(cell.zmin,float(zdiv),cell.pz), ChebyshevPointsFirstKind(float(zdiv),cell.zmax,cell.pz)]
                
                xf = [ChebyshevPointsFirstKind(cell.xmin,cell.xmax,cell.pxf)]
                yf = [ChebyshevPointsFirstKind(cell.ymin,float(ydiv),cell.pyf), ChebyshevPointsFirstKind(float(ydiv),cell.ymax,cell.pyf)]
                zf = [ChebyshevPointsFirstKind(cell.zmin,float(zdiv),cell.pzf), ChebyshevPointsFirstKind(float(zdiv),cell.zmax,cell.pzf)]
                
                ybounds = np.array([cell.ymin, float(ydiv), cell.ymax])
                zbounds = np.array([cell.zmin, float(zdiv), cell.zmax])

                '''call the cell constructor for the children.  Set up parent, uniqueID, neighbor list.  Append to masterList'''
                for j, k in TwoByTwo:
                    
                    children[0,j,k] = Cell('first', cell.xmin, cell.xmax, cell.px, 
                                           ybounds[j], ybounds[j+1], cell.py,
                                           zbounds[k], zbounds[k+1], cell.pz, fine_p=self.pxf, tree = cell.tree)
                    children[0,j,k].parent = cell # children should point to their parent
                    if hasattr(cell, "childrenRefineCause"):
                        children[0,j,k].refineCause = cell.childrenRefineCause
                    children[0,j,k].setUniqueID(0,j,k)  # neighbor lists going to be ruined once no longer dividing into 8
                    children[0,j,k].setNeighborList()
                    
                    if temporaryCell==False:
                        if hasattr(cell.tree, 'masterList'):
                            cell.tree.masterList.insert(bisect.bisect_left(cell.tree.masterList, [children[0,j,k].uniqueID,]), [children[0,j,k].uniqueID,children[0,j,k]])
        
                '''create new gridpoints wherever necessary'''
                newGridpointCount=0
                for jj,kk in TwoByTwo:
                    xOct = x[0]
                    yOct = y[jj]
                    zOct = z[kk]
                    
                    xOctf = xf[0]
                    yOctf = yf[jj]
                    zOctf = zf[kk]
                    
                    gridpoints = np.empty((cell.px+1,cell.py+1,cell.pz+1),dtype=object)
                    fine_gridpoints = np.empty((cell.pxf+1,cell.pyf+1,cell.pzf+1),dtype=object)
                    
                    for i, j, k in cell.PxByPyByPz:
                        newGridpointCount += 1
                        gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k], self.tree.gaugeShift, self.tree.atoms, self.tree.coreRepresentation, self.tree.nOrbitals)
                    
                    for i, j, k in cell.PxfByPyfByPzf:
                        fine_gridpoints[i,j,k] = GridPoint(xOctf[i],yOctf[j],zOctf[k], self.tree.gaugeShift, self.tree.atoms, self.tree.coreRepresentation, self.tree.nOrbitals)
                    
                    
                    children[0,jj,kk].setGridpoints(gridpoints)
                    children[0,jj,kk].setFineGridpoints(fine_gridpoints)

                    if hasattr(cell,'level'):
                        children[0,jj,kk].level = cell.level+1
                
            if printNumberOfCells == True: rprint(rank, 'generated %i new gridpoints for parent cell %s' %(newGridpointCount, cell.uniqueID))
        
            '''set the parent cell's 'children' attribute to the array of children'''
            cell.children = children
        
        def divideInto2_firstKind(cell, xdiv, ydiv, zdiv, printNumberOfCells=False):
            
            '''setup pxXpyXpz array of gridpoint objects.  These will be used to construct the 8 children cells'''
            cell.leaf = False
            if ( (zdiv == None) and (ydiv==None) ):
                
                # First check bounds:
                if ( (xdiv < cell.xmin) or (xdiv > cell.xmax) ):
                    rprint(rank, 'WARNING: XDIV NOT IN CELL BOUNDS')
                children = np.empty((2,1,1), dtype=object)
        
                x = [ChebyshevPointsFirstKind(cell.xmin,float(xdiv),cell.px), ChebyshevPointsFirstKind(float(xdiv),cell.xmax,cell.px)]
                y = [ChebyshevPointsFirstKind(cell.ymin,cell.ymax,cell.py)]
                z = [ChebyshevPointsFirstKind(cell.zmin,cell.zmax,cell.pz)]
                
                xf = [ChebyshevPointsFirstKind(cell.xmin,float(xdiv),cell.pxf), ChebyshevPointsFirstKind(float(xdiv),cell.xmax,cell.pxf)]
                yf = [ChebyshevPointsFirstKind(cell.ymin,cell.ymax,cell.pyf)]
                zf = [ChebyshevPointsFirstKind(cell.zmin,cell.zmax,cell.pzf)]
                
                xbounds = np.array([cell.xmin, float(xdiv), cell.xmax])
        
                '''call the cell constructor for the children.  Set up parent, uniqueID, neighbor list.  Append to masterList'''
                for i in range(2):
                    
                    children[i,0,0] = Cell('first', xbounds[i], xbounds[i+1], cell.px, 
                                           cell.ymin, cell.ymax, cell.py,
                                           cell.zmin, cell.zmax, cell.pz, fine_p=self.pxf, tree = cell.tree)
                    children[i,0,0].parent = cell # children should point to their parent
                    if hasattr(cell, "childrenRefineCause"):
                        children[i,0,0].refineCause = cell.childrenRefineCause
                    children[i,0,0].setUniqueID(i,0,0)  # neighbor lists going to be ruined once no longer dividing into 8
                    children[i,0,0].setNeighborList()
        
                    if hasattr(cell.tree, 'masterList'):
                        cell.tree.masterList.insert(bisect.bisect_left(cell.tree.masterList, [children[i,0,0].uniqueID,]), [children[i,0,0].uniqueID,children[i,0,0]])
        
                '''create new gridpoints wherever necessary'''
                newGridpointCount=0
                for ii in range(2):
                    xOct = x[ii]
                    yOct = y[0]
                    zOct = z[0]
                    
                    xOctf = xf[ii]
                    yOctf = yf[0]
                    zOctf = zf[0]
                    
                    gridpoints = np.empty((cell.px+1,cell.py+1,cell.pz+1),dtype=object)
                    fine_gridpoints = np.empty((cell.px+1,cell.py+1,cell.pz+1),dtype=object)
                    
                    for i, j, k in cell.PxByPyByPz:
                        newGridpointCount += 1
                        gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k], self.tree.gaugeShift, self.tree.atoms, self.tree.coreRepresentation, self.tree.nOrbitals)
#                         gridpoints[i,j,k].setExternalPotential(cell.tree.atoms, cell.tree.gaugeShift)
                    children[ii,0,0].setGridpoints(gridpoints)
                    
                    
                    for i, j, k in cell.PxfByPyfByPzf:
                        fine_gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k], self.tree.gaugeShift, self.tree.atoms, self.tree.coreRepresentation, self.tree.nOrbitals)
#                         gridpoints[i,j,k].setExternalPotential(cell.tree.atoms, cell.tree.gaugeShift)
                    children[ii,0,0].setFineGridpoints(fine_gridpoints)
                    
                    
                    if hasattr(cell,'level'):
                        children[ii,0,0].level = cell.level+1
#                 rprint(rank, 'Not increasing the cell level because only dividing along x axis.')
                        
            elif ( (zdiv == None) and (xdiv==None) ):  # divide along y axis only
                # First check bounds:
                if ( (ydiv < cell.ymin) or (ydiv > cell.ymax) ):
                    rprint(rank, 'WARNING: YDIV NOT IN CELL BOUNDS')
                    
                children = np.empty((1,2,1), dtype=object)
                cell.leaf = False
        
                x = [ChebyshevPointsFirstKind(cell.xmin,cell.xmax,cell.px)]
                y = [ChebyshevPointsFirstKind(cell.ymin,float(ydiv),cell.py), ChebyshevPointsFirstKind(float(ydiv),cell.ymax,cell.py)]
                z = [ChebyshevPointsFirstKind(cell.zmin,cell.zmax,cell.pz)]
                
                xf = [ChebyshevPointsFirstKind(cell.xmin,cell.xmax,cell.pxf)]
                yf = [ChebyshevPointsFirstKind(cell.ymin,float(ydiv),cell.pyf), ChebyshevPointsFirstKind(float(ydiv),cell.ymax,cell.pyf)]
                zf = [ChebyshevPointsFirstKind(cell.zmin,cell.zmax,cell.pzf)]
                
                ybounds = np.array([cell.ymin, float(ydiv), cell.ymax])
        
                '''call the cell constructor for the children.  Set up parent, uniqueID, neighbor list.  Append to masterList'''
                for j in range(2):
                    
                    children[0,j,0] = Cell('first', cell.xmin, cell.xmax, cell.px, 
                                           ybounds[j], ybounds[j+1], cell.py,
                                           cell.zmin, cell.zmax, cell.pz, fine_p=self.pxf, tree = cell.tree)
                    children[0,j,0].parent = cell # children should point to their parent
                    if hasattr(cell, "childrenRefineCause"):
                        children[0,j,0].refineCause = cell.childrenRefineCause
                    children[0,j,0].setUniqueID(0,j,0)  # neighbor lists going to be ruined once no longer dividing into 8
                    children[0,j,0].setNeighborList()
        
                    if hasattr(cell.tree, 'masterList'):
                        cell.tree.masterList.insert(bisect.bisect_left(cell.tree.masterList, [children[0,j,0].uniqueID,]), [children[0,j,0].uniqueID,children[0,j,0]])
        
                '''create new gridpoints wherever necessary'''
                newGridpointCount=0
                for jj in range(2):
                    xOct = x[0]
                    yOct = y[jj]
                    zOct = z[0]
                    
                    xOctf = xf[0]
                    yOctf = yf[jj]
                    zOctf = zf[0]
                    
                    gridpoints = np.empty((cell.px+1,cell.py+1,cell.pz+1),dtype=object)
                    fine_gridpoints = np.empty((cell.pxf+1,cell.pyf+1,cell.pzf+1),dtype=object)
                    for i, j, k in cell.PxByPyByPz:
                        newGridpointCount += 1
                        gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k], self.tree.gaugeShift, self.tree.atoms, self.tree.coreRepresentation, self.tree.nOrbitals)
                    children[0,jj,0].setGridpoints(gridpoints)
                    
                    for i, j, k in cell.PxfByPyfByPzf:
                        fine_gridpoints[i,j,k] = GridPoint(xOctf[i],yOctf[j],zOctf[k], self.tree.gaugeShift, self.tree.atoms, self.tree.coreRepresentation, self.tree.nOrbitals)
                    children[0,jj,0].setFineGridpoints(fine_gridpoints)
                    
                    if hasattr(cell,'level'):
                        children[0,jj,0].level = cell.level+1
#                 rprint(rank, 'Not increasing the cell level because only dividing along y axis.')

                        
            elif ( (xdiv == None) and (ydiv==None) ):  # divide along z axis only
                # First check bounds:
                if ( (zdiv < cell.zmin) or (zdiv > cell.zmax) ):
                    rprint(rank, 'WARNING: ZDIV NOT IN CELL BOUNDS')
                    
                children = np.empty((1,1,2), dtype=object)
                cell.leaf = False
        
                x = [ChebyshevPointsFirstKind(cell.xmin,cell.xmax,cell.px)]
                y = [ChebyshevPointsFirstKind(cell.ymin,cell.ymax,cell.py)]
                z = [ChebyshevPointsFirstKind(cell.zmin,float(zdiv),cell.pz), ChebyshevPointsFirstKind(float(zdiv),cell.zmax,cell.pz)]
                
                xf = [ChebyshevPointsFirstKind(cell.xmin,cell.xmax,cell.pxf)]
                yf = [ChebyshevPointsFirstKind(cell.ymin,cell.ymax,cell.pyf)]
                zf = [ChebyshevPointsFirstKind(cell.zmin,float(zdiv),cell.pzf), ChebyshevPointsFirstKind(float(zdiv),cell.zmax,cell.pzf)]
                
                zbounds = np.array([cell.zmin, float(zdiv), cell.zmax])

                '''call the cell constructor for the children.  Set up parent, uniqueID, neighbor list.  Append to masterList'''
                for k in range(2):
                    children[0,0,k] = Cell('first', cell.xmin, cell.xmax, cell.px, 
                                           cell.ymin, cell.ymax, cell.py,
                                           zbounds[k], zbounds[k+1], cell.pz, fine_p=self.pxf, tree = cell.tree)
                    children[0,0,k].parent = cell # children should point to their parent
                    if hasattr(cell, "childrenRefineCause"):
                        children[0,0,k].refineCause = cell.childrenRefineCause
                    children[0,0,k].setUniqueID(0,0,k)  # neighbor lists going to be ruined once no longer dividing into 8
                    children[0,0,k].setNeighborList()
        
                    if hasattr(cell.tree, 'masterList'):
                        cell.tree.masterList.insert(bisect.bisect_left(cell.tree.masterList, [children[0,0,k].uniqueID,]), [children[0,0,k].uniqueID,children[0,0,k]])
        
                '''create new gridpoints wherever necessary'''
                newGridpointCount=0
                for kk in range(2):
                    xOct = x[0]
                    yOct = y[0]
                    zOct = z[kk]
                    
                    xOctf = xf[0]
                    yOctf = yf[0]
                    zOctf = zf[kk]
                    
                    
                    gridpoints = np.empty((cell.px+1,cell.py+1,cell.pz+1),dtype=object)
                    for i, j, k in cell.PxByPyByPz:
                        newGridpointCount += 1
                        gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k], self.tree.gaugeShift, self.tree.atoms, self.tree.coreRepresentation, self.tree.nOrbitals)
#                         gridpoints[i,j,k].setExternalPotential(cell.tree.atoms, cell.tree.gaugeShift)
                    children[0,0,kk].setGridpoints(gridpoints)
                    
                    
                    fine_gridpoints = np.empty((cell.pxf+1,cell.pyf+1,cell.pzf+1),dtype=object)
                    for i, j, k in cell.PxfByPyfByPzf:
                        fine_gridpoints[i,j,k] = GridPoint(xOctf[i],yOctf[j],zOctf[k], self.tree.gaugeShift, self.tree.atoms, self.tree.coreRepresentation, self.tree.nOrbitals)
                    children[0,0,kk].setFineGridpoints(fine_gridpoints)
                    
                    
                    if hasattr(cell,'level'):
                        children[0,0,kk].level = cell.level+1
#                 rprint(rank, 'Not increasing the cell level because only dividing along z axis.')

                        
            if printNumberOfCells == True: rprint(rank, 'generated %i new gridpoints for parent cell %s' %(newGridpointCount, cell.uniqueID))
        
            '''set the parent cell's 'children' attribute to the array of children'''
            cell.children = children
            
        if self.leaf == False:
            rprint(rank, 'Why are you dividing a non-leaf cell?')
        
        noneCount = 0
        if xdiv == None: noneCount += 1
        if ydiv == None: noneCount += 1
        if zdiv == None: noneCount += 1
        
        if noneCount == 0:
            divideInto8_firstKind(self, xdiv, ydiv, zdiv, printNumberOfCells, interpolate)
        elif noneCount == 1:
#             rprint(rank, 'Using divideInto4... are you sure?')
            divideInto4_firstKind(self, xdiv, ydiv, zdiv, printNumberOfCells) 
        elif noneCount == 2:
#             rprint(rank, 'Using divideInto2... are you sure?')
            divideInto2_firstKind(self, xdiv, ydiv, zdiv, printNumberOfCells)
        elif noneCount == 3:
            rprint(rank, 'Not acutally dividing because xdiv=ydiv=zdiv=None.  Happens when trying to divide at a nucleus that is already at a vertex.')

    def divideIfAspectRatioExceeds(self, tolerance):
        
        def getAspectRatio(cell):
        
            dx = (cell.xmax-cell.xmin)
            dy = (cell.ymax-cell.ymin)
            dz = (cell.zmax-cell.zmin)
            L = max(dx, max(dy,dz))
            l = min(dx, min(dy,dz))
                    
            return L/l
    
        aspectRatio = getAspectRatio(self)
        
        if aspectRatio > tolerance:  # fix tolerance to 1.5 for now
#             for atom in self.tree.atoms
#             distToAtom = np.sqrt(self.xmid-atom.x)
#             rprint(rank, 'Cell ', self.uniqueID,' has apsect ratio of ', aspectRatio,'.  Dividing')
            # find longest edge:
            dx = self.xmax-self.xmin
            dy = self.ymax-self.ymin
            dz = self.zmax-self.zmin
#             rprint(rank, 'dx = ', dx)
#             rprint(rank, 'dy = ', dy)
#             rprint(rank, 'dz = ', dz)
            
            # locate shortest dimension.  Divide, then check aspect ratio of children.  
            if (dx <= min(dy,dz)): # x is shortest dimension.
                self.divide_firstKind(xdiv = None, ydiv=(self.ymax+self.ymin)/2, zdiv=(self.zmax+self.zmin)/2)
            elif (dy <= min(dx,dz)): # y is shortest dimension
                self.divide_firstKind(xdiv=(self.xmax+self.xmin)/2, ydiv = None, zdiv=(self.zmax+self.zmin)/2)
            elif (dz <= max(dx,dy)): # z is shortest dimension
                self.divide_firstKind(xdiv=(self.xmax+self.xmin)/2, ydiv=(self.ymax+self.ymin)/2, zdiv = None)
#                 
#               Should I divide children?  Maybe it's okay if a child still has a bad aspect ratio because
#               at least no one side  
  
#             if hasattr(self, "children"):
#                 (ii,jj,kk) = np.shape(self.children)
#                 for i in range(ii):
#                     for j in range(jj):
#                         for k in range(kk):
#                             self.children[i,j,k].divideIfAspectRatioExceeds(tolerance)
                
                
# #             locate longest dimension.  Divide, then check aspect ratio of children.  
#             if (dx >= max(dy,dz)): # x is longest dimension.
#                 self.divide(xdiv = (self.xmax+self.xmin)/2, ydiv=None, zdiv=None)
#                 self.children[0,0,0].divideIfAspectRatioExceeds(tolerance)
#                 self.children[1,0,0].divideIfAspectRatioExceeds(tolerance)
#             elif (dy >= max(dx,dz)): # y is longest dimension
#                 self.divide(xdiv=None, ydiv = (self.ymax+self.ymin)/2, zdiv=None)
#                 self.children[0,0,0].divideIfAspectRatioExceeds(tolerance)
#                 self.children[0,1,0].divideIfAspectRatioExceeds(tolerance)
#             elif (dz >= max(dx,dy)): # z is longest dimension
#                 self.divide(xdiv=None, ydiv=None, zdiv = (self.zmax+self.zmin)/2)
#                 self.children[0,0,0].divideIfAspectRatioExceeds(tolerance)
#                 self.children[0,0,1].divideIfAspectRatioExceeds(tolerance)
             
    def divideButJustReturnChildren(self):
        '''setup pxXpyXpz array of gridpoint objects.  These will be used to construct the 8 children cells'''
        children = np.empty((2,2,2), dtype=object)
        
        x = [ChebyshevPointsFirstKind(self.xmin,self.xmid,self.px), ChebyshevPointsFirstKind(self.xmid,self.xmax,self.px)]
        y = [ChebyshevPointsFirstKind(self.ymin,self.ymid,self.py), ChebyshevPointsFirstKind(self.ymid,self.ymax,self.py)]
        z = [ChebyshevPointsFirstKind(self.zmin,self.zmid,self.pz), ChebyshevPointsFirstKind(self.zmid,self.zmax,self.pz)]
                
        xbounds = [self.xmin, self.xmid, self.xmax]
        ybounds = [self.ymin, self.ymid, self.ymax]
        zbounds = [self.zmin, self.zmid, self.zmax]
        
        
        '''call the cell constructor for the children.  Set up parent, uniqueID, neighbor list.  Append to masterList'''
        for i, j, k in TwoByTwoByTwo:
            children[i,j,k] = Cell(xbounds[i], xbounds[i+1], self.px, 
                                   ybounds[j], ybounds[j+1], self.py,
                                   zbounds[k], zbounds[k+1], self.pz, tree = self.tree)


            
        '''create new gridpoints wherever necessary'''
        for ii,jj,kk in TwoByTwoByTwo:
            xOct = x[ii]
            yOct = y[jj]
            zOct = z[kk]
            gridpoints = np.empty((self.px+1,self.py+1,self.pz+1),dtype=object)
            for i, j, k in self.PxByPyByPz:
                gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k], self.tree.gaugeShift, self.tree.atoms, self.tree.coreRepresentation, self.tree.nOrbitals)
            children[ii,jj,kk].setGridpoints(gridpoints)

        return children

          
    """
    HAMILTONIAN FUNCTIONS
    """
    def computeOrbitalPotentials(self,targetEnergy=None):
        
        phi = np.empty((self.px+1,self.py+1,self.pz+1))
        pot = np.empty((self.px+1,self.py+1,self.pz+1))
        if targetEnergy!=None:
            for i,j,k in self.PxByPyByPz:
                gp = self.gridpoints[i,j,k]
                phi[i,j,k] = gp.phi[targetEnergy]
                pot[i,j,k] = gp.v_eff
                
            self.orbitalPE[targetEnergy] = np.sum( self.w * phi**2 * pot)
        else:   
            for m in range(self.tree.nOrbitals):
                if self.tree.occupations[m] > -1e-10: #otherwise dont update energy
                    for i,j,k in self.PxByPyByPz:
                        gp = self.gridpoints[i,j,k]
                        phi[i,j,k] = gp.phi[m]
                        pot[i,j,k] = gp.v_eff
                        
                    self.orbitalPE[m] = np.sum( self.w * phi**2 * pot)

    def computeOrbitalKinetics(self,targetEnergy=None):
        
        phi = np.empty((self.px+1,self.py+1,self.pz+1))
        
        if targetEnergy!=None:
            for i,j,k in self.PxByPyByPz:
                gp = self.gridpoints[i,j,k]
                phi[i,j,k] = gp.phi[targetEnergy]
        
            gradPhi = ChebGradient3D(self.DopenX, self.DopenY, self.DopenZ, self.px, phi)
#                gradPhi = ChebGradient3D(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax,self.px,phi) 
            gradPhiSq = gradPhi[0]**2 + gradPhi[1]**2 + gradPhi[2]**2
            
            self.orbitalKE[targetEnergy] = 1/2*np.sum( self.w * gradPhiSq )
        else:
            for m in range(self.tree.nOrbitals):
                if self.tree.occupations[m] > -1e-10: #otherwise dont update energy
                    for i,j,k in self.PxByPyByPz:
                        gp = self.gridpoints[i,j,k]
                        phi[i,j,k] = gp.phi[m]
                
                    gradPhi = ChebGradient3D(self.DopenX, self.DopenY, self.DopenZ, self.px, phi)
    #                gradPhi = ChebGradient3D(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax,self.px,phi) 
                    gradPhiSq = gradPhi[0]**2 + gradPhi[1]**2 + gradPhi[2]**2
                    
                    self.orbitalKE[m] = 1/2*np.sum( self.w * gradPhiSq )
                else:
                    rprint(rank, 'Not updating orbital kinetics because occupation < -1e-10')
    
    def computeOrbitalKinetics_Laplacian(self,targetEnergy=None):
        
        phi = np.empty((self.px+1,self.py+1,self.pz+1))
        
        if targetEnergy!=None:
            for i,j,k in self.PxByPyByPz:
                gp = self.gridpoints[i,j,k]
                phi[i,j,k] = gp.phi[targetEnergy]
        
            laplacianPhi = ChebLaplacian3D(self.DopenX, self.DopenY, self.DopenZ, self.px, phi)
            
            self.orbitalKE[targetEnergy] = -1/2*np.sum( self.w * phi*laplacianPhi )
        else:
            for m in range(self.tree.nOrbitals):
                if self.tree.occupations[m] > -1e-10: #otherwise dont update energy
                    for i,j,k in self.PxByPyByPz:
                        gp = self.gridpoints[i,j,k]
                        phi[i,j,k] = gp.phi[m]
                
                    laplacianPhi = ChebLaplacian3D(self.DopenX, self.DopenY, self.DopenZ, self.px, phi)
                    
                    self.orbitalKE[m] = -1/2*np.sum( self.w * phi*laplacianPhi )
                else:
                    rprint(rank, 'Not updating orbital kinetics because occupation < -1e-10')
    
    def computeDerivativeMatrices(self):
        self.DopenX = computeDerivativeMatrix(self.xmin, self.xmax, self.px)
        self.DopenY = computeDerivativeMatrix(self.ymin, self.ymax, self.py)
        self.DopenZ = computeDerivativeMatrix(self.zmin, self.zmax, self.pz)
        
    def computeLaplacianAndInverse(self):
        self.laplacian = computeLaplacianMatrix(self.xmin, self.xmax, self.px,
                                                self.ymin, self.ymax, self.py,
                                                self.zmin, self.zmax, self.pz)
        
        self.inverseLaplacian = np.linalg.inv(self.laplacian)
        
    def computeLaplacian(self):
        self.laplacian = computeLaplacianMatrix(self.xmin, self.xmax, self.px,
                                                self.ymin, self.ymax, self.py,
                                                self.zmin, self.zmax, self.pz)
        
        
    def setDensityInterpolator(self):
        rho = np.zeros((self.px+1,self.py+1,self.pz+1))
        for i,j,k in self.PxByPyByPz:
            rho[i,j,k] = self.gridpoints[i,j,k].rho
        
        x = np.zeros(self.px)    
        y = np.zeros(self.py)    
        z = np.zeros(self.pz)    
        for i in range(self.px):  # Assumes px=py=pz
            x[i] = self.gridpoints[i,0,0].x
            y[i] = self.gridpoints[0,i,0].y
            z[i] = self.gridpoints[0,0,i].z
            
         
  
        self.densityInterpolator = self.interpolator(x, y, z, rho)
        

        