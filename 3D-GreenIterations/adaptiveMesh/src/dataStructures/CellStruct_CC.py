'''
Created on Mar 5, 2018

@author: nathanvaughn
'''
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import itertools
import bisect

from hydrogenAtom import potential
from meshUtilities import meshDensity, weights3D, unscaledWeights, ChebGradient3D, ChebyshevPoints
from GridpointStruct import GridPoint

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
    def __init__(self, xmin, xmax, px, ymin, ymax, py, zmin, zmax, pz, gridpoints=None, tree=None):
        '''
        Cell Constructor.  Cell composed of gridpoint objects
        '''
        self.tree = tree
        self.px = px
        self.py = py
        self.pz = pz
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.gridpoints = gridpoints
        self.leaf = True
        W = unscaledWeights(px)  # assumed px=py=pz
        self.w = weights3D(xmin, xmax, px, ymin, ymax, py, zmin, zmax, pz, W)
        self.PxByPyByPz = [element for element in itertools.product(range(self.px),range(self.py),range(self.pz))]
        self.setCellMidpointAndVolume()

    def setGridpoints(self,gridpoints):
        self.gridpoints = gridpoints
             
    def setCellMidpointAndVolume(self):

        self.volume = (self.xmax-self.xmin)*(self.ymax-self.ymin)*(self.zmax-self.zmin)
        self.xmid = 1/2*(self.xmin + self.xmax)
        self.ymid = 1/2*(self.ymin + self.ymax)
        self.zmid = 1/2*(self.zmin + self.zmax)
        
        if self.volume==0.0:
            print('warning: cell has zero volume')
            print('Dx = ', self.xmax-self.xmin)
            print('xmin = ', self.xmin)
            print('xmax = ', self.xmax)
            print('Dy = ', self.ymax-self.ymin)
            print('ymin = ', self.ymin)
            print('ymax = ', self.ymax)
            print('Dz = ', self.zmax-self.zmin)
            print('zmin = ', self.zmin)
            print('zmax = ', self.zmax)
            print()
            
        
        if abs(np.sum(self.w) - self.volume) > 1e-10:
            print('warning, cell weights dont sum to cell volume.')

    def getAspectRatio(self):
        
        dx = (self.xmax-self.xmin)
        dy = (self.ymax-self.ymin)
        dz = (self.zmax-self.zmin)
        L = max(dx, max(dy,dz))
        l = min(dx, min(dy,dz))
                
        return L/l
    
    
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
            tempChild = GridPoint(xmids[i],ymids[j],zmids[k],self.tree.nOrbitals)
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
               
    
    """
    DIVISION FUNCTIONS
    """
    def cleanupAfterDivide(self):
        self.w = None
        self.gridpoints = None
        self.PxByPyByPz = None
    
    def interpolatForDivision(self):
        phiCoarse = np.empty((self.px,self.py,self.pz))
        xvec = np.empty((self.px))
        yvec = np.empty((self.py))
        zvec = np.empty((self.pz))
        for i,j,k in self.PxByPyByPz:
            xvec[i] = self.gridpoints[i,j,k].x
            yvec[j] = self.gridpoints[i,j,k].y
            zvec[k] = self.gridpoints[i,j,k].z
            phiCoarse[i,j,k] = self.gridpoints[i,j,k].phi
        print('Interpolator generated')
        self.interpolator = RegularGridInterpolator((xvec, yvec, zvec), phiCoarse,method='nearest') 
    
    def divide(self, xdiv, ydiv, zdiv, printNumberOfCells=False, interpolate=False):
                  
        def divideInto8(cell, xdiv, ydiv, zdiv, printNumberOfCells=False, interpolate=False):
            '''setup pxXpyXpz array of gridpoint objects.  These will be used to construct the 8 children cells'''
            
            children = np.empty((2,2,2), dtype=object)
            self.leaf = False
            
            if interpolate==True:
                self.interpolatForDivision()
        
            x = [ChebyshevPoints(cell.xmin,float(xdiv),cell.px), ChebyshevPoints(float(xdiv),cell.xmax,cell.px)]
            y = [ChebyshevPoints(cell.ymin,float(ydiv),cell.py), ChebyshevPoints(float(ydiv),cell.ymax,cell.py)]
            z = [ChebyshevPoints(cell.zmin,float(zdiv),cell.pz), ChebyshevPoints(float(zdiv),cell.zmax,cell.pz)]
            
            xbounds = np.array([cell.xmin, float(xdiv), cell.xmax])
            ybounds = np.array([cell.ymin, float(ydiv), cell.ymax])
            zbounds = np.array([cell.zmin, float(zdiv), cell.zmax])
    
            '''call the cell constructor for the children.  Set up parent, uniqueID, neighbor list.  Append to masterList'''
            for i, j, k in TwoByTwoByTwo:
                
                children[i,j,k] = Cell(xbounds[i], xbounds[i+1], cell.px, 
                                       ybounds[j], ybounds[j+1], cell.py,
                                       zbounds[k], zbounds[k+1], cell.pz, tree = cell.tree)
                children[i,j,k].parent = cell # children should point to their parent
                children[i,j,k].setUniqueID(i,j,k)
                children[i,j,k].setNeighborList()
    
                if hasattr(cell.tree, 'masterList'):
                    cell.tree.masterList.insert(bisect.bisect_left(cell.tree.masterList, [children[i,j,k].uniqueID,]), [children[i,j,k].uniqueID,children[i,j,k]])
    
            '''create new gridpoints wherever necessary'''
            newGridpointCount=0
            for ii,jj,kk in TwoByTwoByTwo:
                xOct = x[ii]
                yOct = y[jj]
                zOct = z[kk]
                gridpoints = np.empty((cell.px,cell.py,cell.pz),dtype=object)
                for i, j, k in cell.PxByPyByPz:
                    newGridpointCount += 1
                    gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k],self.tree.nOrbitals)
                    if interpolate == True:
                        phi = self.interpolator([xOct[i],yOct[j],zOct[k]])
                        gridpoints[i,j,k].setPhi(phi)
                    gridpoints[i,j,k].setExternalPotential(cell.tree.atoms)
                children[ii,jj,kk].setGridpoints(gridpoints)
                if hasattr(cell,'level'):
                    children[ii,jj,kk].level = cell.level+1
            
            if printNumberOfCells == True: print('generated %i new gridpoints for parent cell %s' %(newGridpointCount, cell.uniqueID))
    
            '''set the parent cell's 'children' attribute to the array of children'''
            cell.children = children
        
        def divideInto4(cell, xdiv, ydiv, zdiv, printNumberOfCells=False):
            '''setup pxXpyXpz array of gridpoint objects.  These will be used to construct the 8 children cells'''
            if zdiv == None:
                children = np.empty((2,2,1), dtype=object)
                cell.leaf = False
        
                x = [ChebyshevPoints(cell.xmin,float(xdiv),cell.px), ChebyshevPoints(float(xdiv),cell.xmax,cell.px)]
                y = [ChebyshevPoints(cell.ymin,float(ydiv),cell.py), ChebyshevPoints(float(ydiv),cell.ymax,cell.py)]
                z = [ChebyshevPoints(cell.zmin,cell.zmax,cell.pz)]
                
                xbounds = np.array([cell.xmin, float(xdiv), cell.xmax])
                ybounds = np.array([cell.ymin, float(ydiv), cell.ymax])        

                '''call the cell constructor for the children.  Set up parent, uniqueID, neighbor list.  Append to masterList'''
                for i, j in TwoByTwo:
                    
                    children[i,j,0] = Cell(xbounds[i], xbounds[i+1], cell.px, 
                                           ybounds[j], ybounds[j+1], cell.py,
                                           cell.zmin, cell.zmax, cell.pz, tree = cell.tree)
                    children[i,j,0].parent = cell # children should point to their parent
                    children[i,j,0].setUniqueID(i,j,0)  # neighbor lists going to be ruined once no longer dividing into 8
                    children[i,j,0].setNeighborList()
        
                    if hasattr(cell.tree, 'masterList'):
                        cell.tree.masterList.insert(bisect.bisect_left(cell.tree.masterList, [children[i,j,0].uniqueID,]), [children[i,j,0].uniqueID,children[i,j,0]])
        
                '''create new gridpoints wherever necessary'''
                newGridpointCount=0
                for ii,jj in TwoByTwo:
                    xOct = x[ii]
                    yOct = y[jj]
                    zOct = z[0]
                    gridpoints = np.empty((cell.px,cell.py,cell.pz),dtype=object)
                    for i, j, k in cell.PxByPyByPz:
                        newGridpointCount += 1
                        gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k],self.tree.nOrbitals)
                        gridpoints[i,j,k].setExternalPotential(cell.tree.atoms)
                    children[ii,jj,0].setGridpoints(gridpoints)
                    if hasattr(cell,'level'):
                        children[ii,jj,0].level = cell.level+1
                        
            elif ydiv == None:  # divideInto8 along the x and z axes, but not the y
                children = np.empty((2,1,2), dtype=object)
                cell.leaf = False
            
                x = [ChebyshevPoints(cell.xmin,float(xdiv),cell.px), ChebyshevPoints(float(xdiv),cell.xmax,cell.px)]
                y = [ChebyshevPoints(cell.ymin,cell.ymax,cell.py)]
                z = [ChebyshevPoints(cell.zmin,float(zdiv),cell.pz), ChebyshevPoints(float(zdiv),cell.zmax,cell.pz)]
                
                xbounds = np.array([cell.xmin, float(xdiv), cell.xmax])
                zbounds = np.array([cell.zmin, float(zdiv), cell.zmax])
        
                '''call the cell constructor for the children.  Set up parent, uniqueID, neighbor list.  Append to masterList'''
                for i, k in TwoByTwo:
                    
                    children[i,0,k] = Cell(xbounds[i], xbounds[i+1], cell.px, 
                                           cell.ymin, cell.ymax, cell.py,
                                           zbounds[k], zbounds[k+1], cell.pz, tree = cell.tree)
                    children[i,0,k].parent = cell # children should point to their parent
                    children[i,0,k].setUniqueID(i,0,k)  # neighbor lists going to be ruined once no longer dividing into 8
                    children[i,0,k].setNeighborList()
        
                    if hasattr(cell.tree, 'masterList'):
                        cell.tree.masterList.insert(bisect.bisect_left(cell.tree.masterList, [children[i,0,k].uniqueID,]), [children[i,0,k].uniqueID,children[i,0,k]])
        
        
                '''create new gridpoints wherever necessary'''
                newGridpointCount=0
                for ii,kk in TwoByTwo:
                    xOct = x[ii]
                    yOct = y[0]
                    zOct = z[kk]
                    gridpoints = np.empty((cell.px,cell.py,cell.pz),dtype=object)
                    for i, j, k in cell.PxByPyByPz:
                        newGridpointCount += 1
                        gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k],self.tree.nOrbitals)
                        gridpoints[i,j,k].setExternalPotential(cell.tree.atoms)
                    children[ii,0,kk].setGridpoints(gridpoints)
                    if hasattr(cell,'level'):
                        children[ii,0,kk].level = cell.level+1
                        
            elif xdiv == None:  # divideInto8 along the x and z axes, but not the y
                children = np.empty((1,2,2), dtype=object)
                cell.leaf = False
            
                x = [ChebyshevPoints(cell.xmin,cell.xmax,cell.px)]
                y = [ChebyshevPoints(cell.ymin,float(ydiv),cell.py), ChebyshevPoints(float(ydiv),cell.ymax,cell.py)]
                z = [ChebyshevPoints(cell.zmin,float(zdiv),cell.pz), ChebyshevPoints(float(zdiv),cell.zmax,cell.pz)]
                
                ybounds = np.array([cell.ymin, float(ydiv), cell.ymax])
                zbounds = np.array([cell.zmin, float(zdiv), cell.zmax])

                '''call the cell constructor for the children.  Set up parent, uniqueID, neighbor list.  Append to masterList'''
                for j, k in TwoByTwo:
                    
                    children[0,j,k] = Cell(cell.xmin, cell.xmax, cell.px, 
                                           ybounds[j], ybounds[j+1], cell.py,
                                           zbounds[k], zbounds[k+1], cell.pz, tree = cell.tree)
                    children[0,j,k].parent = cell # children should point to their parent
                    children[0,j,k].setUniqueID(0,j,k)  # neighbor lists going to be ruined once no longer dividing into 8
                    children[0,j,k].setNeighborList()
        
                    if hasattr(cell.tree, 'masterList'):
                        cell.tree.masterList.insert(bisect.bisect_left(cell.tree.masterList, [children[0,j,k].uniqueID,]), [children[0,j,k].uniqueID,children[0,j,k]])
        
                '''create new gridpoints wherever necessary'''
                newGridpointCount=0
                for jj,kk in TwoByTwo:
                    xOct = x[0]
                    yOct = y[jj]
                    zOct = z[kk]
                    gridpoints = np.empty((cell.px,cell.py,cell.pz),dtype=object)
                    for i, j, k in cell.PxByPyByPz:
                        newGridpointCount += 1
                        gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k],self.tree.nOrbitals)
                    children[0,jj,kk].setGridpoints(gridpoints)
                    if hasattr(cell,'level'):
                        children[0,jj,kk].level = cell.level+1
                
            if printNumberOfCells == True: print('generated %i new gridpoints for parent cell %s' %(newGridpointCount, cell.uniqueID))
        
            '''set the parent cell's 'children' attribute to the array of children'''
            cell.children = children
        
        def divideInto2(cell, xdiv, ydiv, zdiv, printNumberOfCells=False):
            '''setup pxXpyXpz array of gridpoint objects.  These will be used to construct the 8 children cells'''
            if ( (zdiv == None) and (ydiv==None) ):
                children = np.empty((2,1,1), dtype=object)
                cell.leaf = False
        
                x = [ChebyshevPoints(cell.xmin,float(xdiv),cell.px), ChebyshevPoints(float(xdiv),cell.xmax,cell.px)]
                y = [ChebyshevPoints(cell.ymin,cell.ymax,cell.py)]
                z = [ChebyshevPoints(cell.zmin,cell.zmax,cell.pz)]
                
                xbounds = np.array([cell.xmin, float(xdiv), cell.xmax])
        
                '''call the cell constructor for the children.  Set up parent, uniqueID, neighbor list.  Append to masterList'''
                for i in range(2):
                    
                    children[i,0,0] = Cell(xbounds[i], xbounds[i+1], cell.px, 
                                           cell.ymin, cell.ymax, cell.py,
                                           cell.zmin, cell.zmax, cell.pz, tree = cell.tree)
                    children[i,0,0].parent = cell # children should point to their parent
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
                    gridpoints = np.empty((cell.px,cell.py,cell.pz),dtype=object)
                    for i, j, k in cell.PxByPyByPz:
                        newGridpointCount += 1
                        gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k],self.tree.nOrbitals)
                        gridpoints[i,j,k].setExternalPotential(cell.tree.atoms)
                    children[ii,0,0].setGridpoints(gridpoints)
                    if hasattr(cell,'level'):
                        children[ii,0,0].level = cell.level+1
                        
            elif ( (zdiv == None) and (xdiv==None) ):  # divide along y axis only
                children = np.empty((1,2,1), dtype=object)
                cell.leaf = False
        
                x = [ChebyshevPoints(cell.xmin,cell.xmax,cell.px)]
                y = [ChebyshevPoints(cell.ymin,float(ydiv),cell.py), ChebyshevPoints(float(ydiv),cell.ymax,cell.py)]
                z = [ChebyshevPoints(cell.zmin,cell.zmax,cell.pz)]
                
                ybounds = np.array([cell.ymin, float(ydiv), cell.ymax])
        
                '''call the cell constructor for the children.  Set up parent, uniqueID, neighbor list.  Append to masterList'''
                for j in range(2):
                    
                    children[0,j,0] = Cell(cell.xmin, cell.xmax, cell.px, 
                                           ybounds[j], ybounds[j+1], cell.py,
                                           cell.zmin, cell.zmax, cell.pz, tree = cell.tree)
                    children[0,j,0].parent = cell # children should point to their parent
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
                    gridpoints = np.empty((cell.px,cell.py,cell.pz),dtype=object)
                    for i, j, k in cell.PxByPyByPz:
                        newGridpointCount += 1
                        gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k],self.tree.nOrbitals)
                        gridpoints[i,j,k].setExternalPotential(cell.tree.atoms)
                    children[0,jj,0].setGridpoints(gridpoints)
                    if hasattr(cell,'level'):
                        children[0,jj,0].level = cell.level+1
                        
            elif ( (xdiv == None) and (ydiv==None) ):  # divide along z axis only
                children = np.empty((1,1,2), dtype=object)
                cell.leaf = False
        
                x = [ChebyshevPoints(cell.xmin,cell.xmax,cell.px)]
                y = [ChebyshevPoints(cell.ymin,cell.ymax,cell.py)]
                z = [ChebyshevPoints(cell.zmin,float(zdiv),cell.pz), ChebyshevPoints(float(zdiv),cell.zmax,cell.pz)]
                
                zbounds = np.array([cell.zmin, float(zdiv), cell.zmax])

                '''call the cell constructor for the children.  Set up parent, uniqueID, neighbor list.  Append to masterList'''
                for k in range(2):
                    children[0,0,k] = Cell(cell.xmin, cell.xmax, cell.px, 
                                           cell.ymin, cell.ymax, cell.py,
                                           zbounds[k], zbounds[k+1], cell.pz, tree = cell.tree)
                    children[0,0,k].parent = cell # children should point to their parent
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
                    gridpoints = np.empty((cell.px,cell.py,cell.pz),dtype=object)
                    for i, j, k in cell.PxByPyByPz:
                        newGridpointCount += 1
                        gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k],self.tree.nOrbitals)
                        gridpoints[i,j,k].setExternalPotential(cell.tree.atoms)
                    children[0,0,kk].setGridpoints(gridpoints)
                    if hasattr(cell,'level'):
                        children[0,0,kk].level = cell.level+1
                        
            if printNumberOfCells == True: print('generated %i new gridpoints for parent cell %s' %(newGridpointCount, cell.uniqueID))
        
            '''set the parent cell's 'children' attribute to the array of children'''
            cell.children = children
            
        if self.leaf == False:
            print('Why are you dividing a non-leaf cell?')
        
        noneCount = 0
        if xdiv == None: noneCount += 1
        if ydiv == None: noneCount += 1
        if zdiv == None: noneCount += 1
        
        if noneCount == 0:
            divideInto8(self, xdiv, ydiv, zdiv, printNumberOfCells, interpolate)
        elif noneCount == 1:
            divideInto4(self, xdiv, ydiv, zdiv, printNumberOfCells) 
        elif noneCount == 2:
            divideInto2(self, xdiv, ydiv, zdiv, printNumberOfCells)

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
#             print('Cell ', self.uniqueID,' has apsect ratio of ', aspectRatio,'.  Dividing')
            # find longest edge:
            dx = self.xmax-self.xmin
            dy = self.ymax-self.ymin
            dz = self.zmax-self.zmin
#             print('dx = ', dx)
#             print('dy = ', dy)
#             print('dz = ', dz)
            
            # locate shortest dimension.  Divide, then check aspect ratio of children.  
            if (dx <= min(dy,dz)): # x is shortest dimension.
                self.divide(xdiv = None, ydiv=(self.ymax+self.ymin)/2, zdiv=(self.zmax+self.zmin)/2)
            elif (dy <= min(dx,dz)): # y is shortest dimension
                self.divide(xdiv=(self.xmax+self.xmin)/2, ydiv = None, zdiv=(self.zmax+self.zmin)/2)
            elif (dz <= max(dx,dy)): # z is shortest dimension
                self.divide(xdiv=(self.xmax+self.xmin)/2, ydiv=(self.ymax+self.ymin)/2, zdiv = None)
             
            if hasattr(self, "children"):
                (ii,jj,kk) = np.shape(self.children)
                for i in range(ii):
                    for j in range(jj):
                        for k in range(kk):
                            self.children[i,j,k].divideIfAspectRatioExceeds(tolerance)
                
            # locate longest dimension.  Divide, then check aspect ratio of children.  
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
        
        x = [ChebyshevPoints(self.xmin,self.xmid,self.px), ChebyshevPoints(self.xmid,self.xmax,self.px)]
        y = [ChebyshevPoints(self.ymin,self.ymid,self.py), ChebyshevPoints(self.ymid,self.ymax,self.py)]
        z = [ChebyshevPoints(self.zmin,self.zmid,self.pz), ChebyshevPoints(self.zmid,self.zmax,self.pz)]
                
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
            gridpoints = np.empty((self.px,self.py,self.pz),dtype=object)
            for i, j, k in self.PxByPyByPz:
                gridpoints[i,j,k] = GridPoint(xOct[i],yOct[j],zOct[k],self.tree.nOrbitals)
            children[ii,jj,kk].setGridpoints(gridpoints)

        return children

          
    """
    HAMILTONIAN FUNCTIONS
    """
    def computeOrbitalPotential(self,epsilon=0):
        
        phi = np.empty((self.px,self.py,self.pz))
        pot = np.empty((self.px,self.py,self.pz))
        
        for i,j,k in self.PxByPyByPz:
            gp = self.gridpoints[i,j,k]
            phi[i,j,k] = gp.phi
            pot[i,j,k] = gp.v_eff
            
        self.orbitalPE = np.sum( self.w * phi**2 * pot)

    def computeOrbitalKinetic(self):
        
        phi = np.empty((self.px,self.py,self.pz))
        
        for i,j,k in self.PxByPyByPz:
            gp = self.gridpoints[i,j,k]
            phi[i,j,k] = gp.phi
        
        gradPhi = ChebGradient3D(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax,self.px,phi) 
        gradPhiSq = gradPhi[0]**2 + gradPhi[1]**2 + gradPhi[2]**2
        
        self.orbitalKE = 1/2*np.sum( self.w * gradPhiSq )



        
        
        