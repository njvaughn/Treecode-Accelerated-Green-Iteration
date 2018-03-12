'''
Created on Mar 5, 2018

@author: nathanvaughn
'''
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import itertools
import bisect

from hydrogenPotential import potential
from GridpointStruct import GridPoint

ThreeByThreeByThree = [element for element in itertools.product(range(3),range(3),range(3))]
TwoByTwoByTwo = [element for element in itertools.product(range(2),range(2),range(2))]
FiveByFiveByFive = [element for element in itertools.product(range(5),range(5),range(5))]
 
 
class Cell(object):
    '''
    Cell object.  Cells are composed of gridpoint objects.
    '''
    def __init__(self, gridpoints=None, tree=None):
        '''
        Cell Constructor.  Cell composed of gridpoint objects
        '''
        self.tree = tree
        self.leaf = True
        if np.shape(gridpoints) == (3,3,3):
            self.setGridpoints(gridpoints)

            
    
    def setGridpoints(self,gridpoints):
        self.gridpoints = gridpoints
        self.getCellBoundsAndVolume()
        
    def getCellBoundsAndVolume(self):
        self.xmin = self.gridpoints[0,0,0].x
        self.xmax = self.gridpoints[2,2,2].x
        self.ymin = self.gridpoints[0,0,0].y
        self.ymax = self.gridpoints[2,2,2].y
        self.zmin = self.gridpoints[0,0,0].z
        self.zmax = self.gridpoints[2,2,2].z
        self.dx = self.gridpoints[1,0,0].x - self.xmin
        self.dy = self.gridpoints[0,1,0].y - self.ymin
        self.dz = self.gridpoints[0,0,1].z - self.zmin
        self.volume = (self.xmax-self.xmin)*(self.ymax-self.ymin)*(self.zmax-self.zmin)
        midpoint = self.gridpoints[1,1,1]
        self.hydrogenV = potential(midpoint.x, midpoint.y, midpoint.z, epsilon=0)
     
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
        
        
    def interpolatForDivision(self):
        if hasattr(self, 'psi'):
            psiCoarse = np.empty((3,3,3))
            xvec = np.array(self.gridpoints[0,0,0].x,self.gridpoints[1,0,0].x,self.gridpoints[2,0,0].x)
            yvec = np.array(self.gridpoints[0,0,0].y,self.gridpoints[0,1,0].y,self.gridpoints[0,2,0].y)
            zvec = np.array(self.gridpoints[0,0,0].z,self.gridpoints[0,0,1].z,self.gridpoints[0,0,2].z)
            for i,j,k in ThreeByThreeByThree:
                psiCoarse[i,j,k] = self.gridpoints[i,j,k].psi
        
            self.interpolator = RegularGridInterpolator((xvec, yvec, zvec), psiCoarse) 
        else:
            print("Can't generate interpolator because psi hasn't been set yet.")
    

    def getPsiVariation(self):
        minPsi = self.gridpoints[0,0,0].psi
        maxPsi = self.gridpoints[0,0,0].psi
        for i,j,k in ThreeByThreeByThree:
            if self.gridpoints[i,j,k].psi < minPsi: minPsi = self.gridpoints[i,j,k].psi
            if self.gridpoints[i,j,k].psi > maxPsi: maxPsi = self.gridpoints[i,j,k].psi
        
        self.psiVariation = maxPsi - minPsi
        
#     def getPsiVPsiVariation(self):
#         minPsiVPsi = self.gridpoints[0,0,0].psi**2*
#         maxPsiVPsi = self.gridpoints[0,0,0].psi**2
#         for i,j,k in ThreeByThreeByThree:
#             if self.gridpoints[i,j,k].psi < minPsi: minPsi = self.gridpoints[i,j,k].psi
#             if self.gridpoints[i,j,k].psi > maxPsi: maxPsi = self.gridpoints[i,j,k].psi
#         
#         self.psiVariation = maxPsi - minPsi
            
    
    
    def checkIfCellShouldDivide(self, divideTolerance):
        '''
        Perform midpoint method for integral(testFunction) for parent cell
        If parent integral differs from the sum of children integrals over the test function
        then set divideFlag to true.
        :param divideTolerance:
        '''

#         self.gridpoints[1,1,1].setTestFunctionValue()
#         parentIntegral = self.gridpoints[1,1,1].testFunctionValue*self.volume
#         
#         childrenIntegral = 0.0
#         xmids = np.array([(3*self.xmin+self.xmax)/4, (self.xmin+3*self.xmax)/4])
#         ymids = np.array([(3*self.ymin+self.ymax)/4, (self.ymin+3*self.ymax)/4])
#         zmids = np.array([(3*self.zmin+self.zmax)/4, (self.zmin+3*self.zmax)/4])
#         for i,j,k in TwoByTwoByTwo:
#             tempChild = GridPoint(xmids[i],ymids[j],zmids[k])
#             tempChild.setTestFunctionValue()
#             childrenIntegral += tempChild.testFunctionValue*(self.volume/8)
#         
#         if abs(parentIntegral-childrenIntegral) > divideTolerance:
#             self.divideFlag = True
#         else:
#             self.divideFlag = False

        self.getPsiVariation()
        if self.psiVariation > divideTolerance:
            self.divideFlag = True
        else:
            self.divideFlag = False
            
#         self.divideFlag=False
#         for i,j,k in ThreeByThreeByThree:
#             if ( (self.gridpoints[i,j,k].x==0) and (self.gridpoints[i,j,k].y==0) and (self.gridpoints[i,j,k].z==0) and (self.dx > divideTolerance) ):
#                 self.divideFlag = True
#                 return
     
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
        if printNeighborResults == True: print('\nTarget Cell ID      ', self.uniqueID)
    
        '''fill in any gridpoints coming from X neighbors'''
        try: 
            xLowID =   [element[1] for element in self.neighbors if element[0] == 'xLow'][0]
            xLowCell = self.tree.masterList[ find(self.tree.masterList, xLowID) ][1]
#             xLowID =   [element[1] for element in self.neighbors if element[0] == 'xLow'][0]
#             xLowCell = [element[1] for element in self.tree.masterList if str(element[0]) == [element[1] for element in self.neighbors if element[0] == 'xLow'][0]][0]
            if hasattr(xLowCell, 'gridpoints'): gridpoints[0,:,:] = xLowCell.gridpoints[2,:,:] # this is failing
            if printNeighborResults == True: print('found xLowCell:   ', xLowCell, 'whose ID is ', xLowCell.uniqueID)
        except: pass
        try: 
            xHighID =  [element[1] for element in self.neighbors if element[0] == 'xHigh'][0]
            xHighCell = self.tree.masterList[ find(self.tree.masterList, xHighID) ][1]
#             xHighCell = [element[1] for element in self.tree.masterList if str(element[0]) == [element[1] for element in self.neighbors if element[0] == 'xHigh'][0]][0]
            if hasattr(xHighCell, 'gridpoints'): gridpoints[2,:,:] = xHighCell.gridpoints[0,:,:] # this is failing
            if printNeighborResults == True: print('found xHighCell:  ', xHighCell, 'whose ID is ', xHighCell.uniqueID)
        except: pass
        
        '''fill in any gridpoints coming from Y neighbors'''
        try: 
            yLowID =     [element[1] for element in self.neighbors if element[0] == 'yLow'][0]
            yLowCell = self.tree.masterList[ find(self.tree.masterList, yLowID) ][1]
#             yLowCell = [element[1] for element in self.tree.masterList if str(element[0]) == [element[1] for element in self.neighbors if element[0] == 'yLow'][0]][0]
            if hasattr(yLowCell, 'gridpoints'): gridpoints[:,0,:] = yLowCell.gridpoints[:,2,:] # this is failing
            if printNeighborResults == True: print('found yLowCell:     ', yLowCell, 'whose ID is ', yLowCell.uniqueID)
        except: pass
        try: 
            yHighID =    [element[1] for element in self.neighbors if element[0] == 'yHigh'][0]
            yHighCell = self.tree.masterList[ find(self.tree.masterList, yHighID) ][1]
#             yHighCell = [element[1] for element in self.tree.masterList if str(element[0]) == [element[1] for element in self.neighbors if element[0] == 'yHigh'][0]][0]
            if hasattr(yHighCell, 'gridpoints'): gridpoints[:,2,:] = yHighCell.gridpoints[:,0,:] # this is failing
            if printNeighborResults == True: print('found yHighCell:    ', yHighCell, 'whose ID is ', yHighCell.uniqueID)
        except: pass
        
        '''fill in any gridpoints coming from Z neighbors'''
        try: 
            zLowID = [element[1] for element in self.neighbors if element[0] == 'zLow'][0]
            zLowCell = self.tree.masterList[ find(self.tree.masterList, zLowID) ][1]
#             zLowCell = [element[1] for element in self.tree.masterList if str(element[0]) == [element[1] for element in self.neighbors if element[0] == 'zLow'][0]][0]
            if hasattr(zLowCell, 'gridpoints'): gridpoints[:,:,0] = zLowCell.gridpoints[:,:,2] # this is failing
            if printNeighborResults == True: print('found zLowCell: ', zLowCell, 'whose ID is ', zLowCell.uniqueID)
        except: pass 
        try: 
            zHighID = [element[1] for element in self.neighbors if element[0] == 'zHigh'][0]
            zHighCell = self.tree.masterList[ find(self.tree.masterList, zHighID) ][1]
#             zHighCell = [element[1] for element in self.tree.masterList if str(element[0]) == [element[1] for element in self.neighbors if element[0] == 'zHigh'][0]][0]
            if hasattr(zHighCell, 'gridpoints'): gridpoints[:,:,2] = zHighCell.gridpoints[:,:,0] # this is failing
            if printNeighborResults == True: print('found zHighCell:    ', zHighCell, 'whose ID is ', zHighCell.uniqueID)
        except: pass  
        
        ''' return the (potentially) modified sub-array of gridpoints'''
        return gridpoints


        
    def divide(self, printNumberOfCells=False):
        '''setup 5x5x5 array of gridpoint objects.  These will be used to construct the 8 children cells'''
        children = np.empty((2,2,2), dtype=object)
        self.leaf = False
        x = np.linspace(self.xmin,self.xmax,5)
        y = np.linspace(self.ymin,self.ymax,5)
        z = np.linspace(self.zmin,self.zmax,5)
        gridpoints = np.empty((5,5,5),dtype=object)
        gridpoints[::2,::2,::2] = self.gridpoints  # AVOIDS DUPLICATION OF GRIDPOINTS.  The 5x5x5 array of gridpoints should have the original 3x3x3 objects within
        
        '''call the cell constructor for the children.  Set up parent, uniqueID, neighbor list.  Append to masterList'''
        for i, j, k in TwoByTwoByTwo:
            children[i,j,k] = Cell(tree = self.tree)
            children[i,j,k].parent = self # children should point to their parent
            children[i,j,k].setUniqueID(i,j,k)
            children[i,j,k].setNeighborList()
#             self.tree.masterList.append([children[i,j,k].uniqueID,children[i,j,k]])  # add cell to the master list
            self.tree.masterList.insert(bisect.bisect_left(self.tree.masterList, [children[i,j,k].uniqueID,]), [children[i,j,k].uniqueID,children[i,j,k]])
#             bisect.insort(self.tree.masterList, [children[i,j,k].uniqueID,children[i,j,k]])
        '''fill in any already existing gridpoints from neighboring cells that have already divided'''
        for i, j, k in TwoByTwoByTwo:    
            gridpoints[2*i:2*i+3, 2*j:2*j+3, 2*k:2*k+3] = children[i,j,k].fillInNeighbors(gridpoints[2*i:2*i+3, 2*j:2*j+3, 2*k:2*k+3])
            
        '''create new gridpoints wherever necessary'''
        newGridpointCount=0
        for i, j, k in FiveByFiveByFive:
            if gridpoints[i,j,k] == None:
                newGridpointCount += 1
                gridpoints[i,j,k] = GridPoint(x[i],y[j],z[k])
        
        if printNumberOfCells == True: print('generated %i new gridpoints for parent cell %s' %(newGridpointCount, self.uniqueID))
        '''set up the children gridpoints from the 5x5x5 array of gridpoints'''
        for i, j, k in TwoByTwoByTwo:
            children[i,j,k].setGridpoints(gridpoints[2*i:2*i+3, 2*j:2*j+3, 2*k:2*k+3])
            '''if this cell is part of a tree, maintain its level'''
            if hasattr(self,'level'):
                children[i,j,k].level = self.level+1
        '''set the parent cell's 'children' attribute to the array of children'''
        self.children = children

    
    def computePotential(self, epsilon=0):
        midpoint = self.gridpoints[1,1,1]
        self.PE = self.volume*midpoint.psi*midpoint.psi*potential(midpoint.x,midpoint.y,midpoint.z, epsilon)

    def computeKinetic(self):
        midpoint = self.gridpoints[1,1,1]
        def computeLaplacian(Cell):
            # get the psi values on a grid
            psi = np.empty((3,3,3))
            for i,j,k in ThreeByThreeByThree:
                psi[i,j,k] = Cell.gridpoints[i,j,k].psi
            gradient = np.gradient(psi, Cell.dx, Cell.dy, Cell.dz, edge_order=2)
            Dxx = np.gradient(gradient[0],self.dx,edge_order=2,axis=0)
            Dyy = np.gradient(gradient[1],self.dy,edge_order=2,axis=1)
            Dzz = np.gradient(gradient[2],self.dz,edge_order=2,axis=2)
            Laplacian = (Dxx + Dyy + Dzz)  # only use the Laplacian at the midpoint, for now at least
            return Laplacian
        Laplacian = computeLaplacian(self)
        self.KE = -1/2*self.volume*midpoint.psi*Laplacian[1,1,1]
            
#         psi = np.empty((3,3,3))
#         for i,j,k in ThreeByThreeByThree:
#             psi[i,j,k] = self.gridpoints[i,j,k].psi
#         gradient = np.gradient(psi, self.dx, self.dy, self.dz, edge_order=2)
#         grad = gradient[0]+gradient[1]+gradient[2]
#         gradPsiSquared = -grad*grad
#         self.KE = -1/2*self.volume*gradPsiSquared[1,1,1]
         
        
   
        
        
        
        