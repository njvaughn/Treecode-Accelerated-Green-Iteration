'''
The main Tree data structure.  The root of the tree is a Cell object that is comprised of the 
entire domain.  The tree gets built by dividing the root cell, recursively, based on the set 
divideInto8 condition.  The current implementation uses the variation of psi within a cell to 
dictate whether or not it divides.  

Cells can perform recursive functions on the tree.  The tree can also extract all gridpoints or
all midpoints as arrays which can be fed in to the GPU kernels, or other tree-external functions.
-- 03/20/2018 NV

@author: nathanvaughn
'''

import numpy as np
import itertools
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')  # to not display
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from GridpointStruct import GridPoint
from CellStruct import Cell
from hydrogenPotential import potential, trueWavefunction, trueEnergy
from timer import Timer

ThreeByThreeByThree = [element for element in itertools.product(range(3),range(3),range(3))]
ThreeByThree = [element for element in itertools.product(range(3),range(3))]
TwoByTwoByTwo = [element for element in itertools.product(range(2),range(2),range(2))]
FiveByFiveByFive = [element for element in itertools.product(range(5),range(5),range(5))]

class Tree(object):
    '''
    Tree object. Constructed of cells, which are composed of gridpoint objects.  
    Trees contain their root, as well as their masterList.
    '''
    def __init__(self, xmin,xmax,ymin,ymax,zmin,zmax):
        '''
        Tree constructor:  
        First construct the gridpoints for cell consisting of entire domain.  
        Then construct the cell that are composed of gridpoints. 
        Then construct the root of the tree.
        '''
        # generate gridpoint objects.  
        xvec = np.linspace(xmin,xmax,3)
        yvec = np.linspace(ymin,ymax,3)
        zvec = np.linspace(zmin,zmax,3)
        gridpoints = np.empty((3,3,3),dtype=object)

        for i, j, k in ThreeByThreeByThree:
            gridpoints[i,j,k] = GridPoint(xvec[i],yvec[j],zvec[k])
        
        # generate root cell from the gridpoint objects  
        self.root = Cell( gridpoints, self )
        self.root.level = 0
        self.root.uniqueID = ''
        self.masterList = [[self.root.uniqueID, self.root]]
        self.xmin = xmin
        self.xmax = xmax
        
    def buildTree(self,minLevels,maxLevels, divideCriterion, divideParameter, printNumberOfCells=False, printTreeProperties = True): # call the recursive divison on the root of the tree
        # max depth returns the maximum depth of the tree.  maxLevels is the limit on how large the tree is allowed to be,
        # regardless of division criteria
        # N is roughly the number of grid points.  It is used to generate the density function.
        timer = Timer()
        def recursiveDivide(self, Cell, minLevels, maxLevels, divideCriterion, divideParameter, levelCounter, printNumberOfCells, maxDepthAchieved=0, minDepthAchieved=100, currentLevel=0):
            levelCounter += 1
            if currentLevel < maxLevels:
                
                if currentLevel < minLevels:
                    Cell.divideFlag = True 
                else:  
                    if (divideCriterion == 'LW1') or (divideCriterion == 'LW2') or (divideCriterion == 'LW3'):
                        Cell.checkIfAboveMeshDensity(divideParameter,divideCriterion)  
                    else:                        
                        Cell.checkIfCellShouldDivide(divideParameter)
                    
                if Cell.divideFlag == True:   
                    Cell.divideInto8(printNumberOfCells)
#                     for i,j,k in TwoByTwoByTwo: # update the list of cells
#                         self.masterList.append([CellStruct.children[i,j,k].uniqueID, CellStruct.children[i,j,k]])
                    for i,j,k in TwoByTwoByTwo:
                        maxDepthAchieved, minDepthAchieved, levelCounter = recursiveDivide(self,Cell.children[i,j,k], minLevels, maxLevels, divideCriterion, divideParameter, levelCounter, printNumberOfCells, maxDepthAchieved, minDepthAchieved, currentLevel+1)
                else:
                    minDepthAchieved = min(minDepthAchieved, currentLevel)
                    
                    
            maxDepthAchieved = max(maxDepthAchieved, currentLevel)                                                                                                                                                       
            return maxDepthAchieved, minDepthAchieved, levelCounter
        
        timer.start()
        levelCounter=0
        self.maxDepthAchieved, self.minDepthAchieved, self.treeSize = recursiveDivide(self, self.root, minLevels, maxLevels, divideCriterion, divideParameter, levelCounter, printNumberOfCells, maxDepthAchieved=0, minDepthAchieved=maxLevels, currentLevel=0 )
        timer.stop()
        
        """ Count the number of unique leaf cells and gridpoints """
        self.numberOfGridpoints = 0
        self.numberOfCells = 0
        for element in self.masterList:
            if element[1].leaf==True:
                self.numberOfCells += 1
                for i,j,k in ThreeByThreeByThree:
                    if not hasattr(element[1].gridpoints[i,j,k], "counted"):
                        self.numberOfGridpoints += 1
                        element[1].gridpoints[i,j,k].counted = True
        
                        
        for element in self.masterList:
            for i,j,k in ThreeByThreeByThree:
                if hasattr(element[1].gridpoints[i,j,k], "counted"):
                    element[1].gridpoints[i,j,k].counted = None
                    
        if printTreeProperties == True: 
            print("Tree build completed. \n"
                  "Domain Size:                 [%.1f, %.1f] \n"
                  "Divide Ciretion:             %s \n"
                  "Divide Parameter:            %1.2e \n"
                  "Total Number of Cells:       %i \n"
                  "Total Number of Leaf Cells:  %i \n"
                  "Total Number of Gridpoints:  %i \n"
                  "Minimum Depth                %i levels \n"
                  "Maximum Depth:               %i levels \n"
                  "Construction time:           %.3g seconds." 
                  %(self.xmin, self.xmax, divideCriterion,divideParameter, self.treeSize, self.numberOfCells, self.numberOfGridpoints, self.minDepthAchieved,self.maxDepthAchieved,timer.elapsedTime))
        
            
    def buildTreeOneCondition(self,minLevels,maxLevels,divideTolerance,printNumberOfCells=False, printTreeProperties = True): # call the recursive divison on the root of the tree
        # max depth returns the maximum depth of the tree.  maxLevels is the limit on how large the tree is allowed to be,
        # regardless of division criteria
        timer = Timer()
        def recursiveDivide(self, Cell, minLevels, maxLevels, divideTolerance, levelCounter, printNumberOfCells, maxDepthAchieved=0, minDepthAchieved=100, currentLevel=0):
            levelCounter += 1
            if currentLevel < maxLevels:
                
                if currentLevel < minLevels:
                    Cell.divideFlag = True 
                else:                             
                    Cell.checkIfCellShouldDivide(divideTolerance)
                    
                if Cell.divideFlag == True:   
                    Cell.divideInto8(printNumberOfCells)
#                     for i,j,k in TwoByTwoByTwo: # update the list of cells
#                         self.masterList.append([CellStruct.children[i,j,k].uniqueID, CellStruct.children[i,j,k]])
                    for i,j,k in TwoByTwoByTwo:
                        maxDepthAchieved, minDepthAchieved, levelCounter = recursiveDivide(self,Cell.children[i,j,k], minLevels, maxLevels, divideTolerance, levelCounter, printNumberOfCells, maxDepthAchieved, minDepthAchieved, currentLevel+1)
                else:
                    minDepthAchieved = min(minDepthAchieved, currentLevel)
                    
                    
            maxDepthAchieved = max(maxDepthAchieved, currentLevel)                                                                                                                                                       
            return maxDepthAchieved, minDepthAchieved, levelCounter
        
        timer.start()
        levelCounter=0
        self.maxDepthAchieved, self.minDepthAchieved, self.treeSize = recursiveDivide(self, self.root, minLevels, maxLevels, divideTolerance, levelCounter, printNumberOfCells, maxDepthAchieved=0, minDepthAchieved=maxLevels, currentLevel=0 )
        timer.stop()
        
        """ Count the number of unique gridpoints """
        self.numberOfGridpoints = 0
        for element in self.masterList:
            if element[1].leaf==True:
                for i,j,k in ThreeByThreeByThree:
                    if not hasattr(element[1].gridpoints[i,j,k], "counted"):
                        self.numberOfGridpoints += 1
                        element[1].gridpoints[i,j,k].counted = True
        
                        
        for element in self.masterList:
            for i,j,k in ThreeByThreeByThree:
                if hasattr(element[1].gridpoints[i,j,k], "counted"):
                    element[1].gridpoints[i,j,k].counted = None
                    
        if printTreeProperties == True: 
            print("Tree build completed. \n"
                  "Domain Size:                 [%.1f, %.1f] \n"
                  "Tolerance:                   %1.2e \n"
                  "Total Number of Cells:       %i \n"
                  "Total Number of Gridpoints:  %i \n"
                  "Minimum Depth                %i levels \n"
                  "Maximum Depth:               %i levels \n"
                  "Construction time:           %.3g seconds." 
                  %(self.xmin, self.xmax, divideTolerance, self.treeSize, self.numberOfGridpoints, self.minDepthAchieved,self.maxDepthAchieved,timer.elapsedTime))
        
     
    def buildTreeTwoConditions(self,minLevels,maxLevels, maxDx, divideTolerance1, divideTolerance2,printNumberOfCells=False, printTreeProperties = True): # call the recursive divison on the root of the tree
        # max depth returns the maximum depth of the tree.  maxLevels is the limit on how large the tree is allowed to be,
        # regardless of division criteria
        timer = Timer()
        def recursiveDivide(self, Cell, minLevels, maxLevels, maxDx, divideTolerance1, divideTolerance2, levelCounter, printNumberOfCells, maxDepthAchieved=0, minDepthAchieved=100, currentLevel=0):
            levelCounter += 1
            if currentLevel < maxLevels:
                
                if ( (currentLevel < minLevels) or (Cell.dx > maxDx)):
                    Cell.divideFlag = True 
                    
                else:                             
                    Cell.checkIfCellShouldDivideTwoConditions(divideTolerance1, divideTolerance2)
                    
                if Cell.divideFlag == True:   
                    Cell.divideInto8(printNumberOfCells)
#                     for i,j,k in TwoByTwoByTwo: # update the list of cells
#                         self.masterList.append([CellStruct.children[i,j,k].uniqueID, CellStruct.children[i,j,k]])
                    for i,j,k in TwoByTwoByTwo:
                        maxDepthAchieved, minDepthAchieved, levelCounter = recursiveDivide(self,Cell.children[i,j,k], minLevels, maxLevels, maxDx, divideTolerance1, divideTolerance2, levelCounter, printNumberOfCells, maxDepthAchieved, minDepthAchieved, currentLevel+1)
                else:
                    minDepthAchieved = min(minDepthAchieved, currentLevel)
                    
                    
            maxDepthAchieved = max(maxDepthAchieved, currentLevel)                                                                                                                                                       
            return maxDepthAchieved, minDepthAchieved, levelCounter
        
        timer.start()
        levelCounter=0
        self.maxDepthAchieved, self.minDepthAchieved, self.treeSize = recursiveDivide(self, self.root, minLevels, maxLevels, maxDx, divideTolerance1, divideTolerance2, levelCounter, printNumberOfCells, maxDepthAchieved=0, minDepthAchieved=maxLevels, currentLevel=0 )
        timer.stop()
        
        """ Count the number of unique gridpoints """
        self.numberOfGridpoints = 0
        for element in self.masterList:
            if element[1].leaf==True:
                for i,j,k in ThreeByThreeByThree:
                    if not hasattr(element[1].gridpoints[i,j,k], "counted"):
                        self.numberOfGridpoints += 1
                        element[1].gridpoints[i,j,k].counted = True
        
                        
        for element in self.masterList:
            for i,j,k in ThreeByThreeByThree:
                if hasattr(element[1].gridpoints[i,j,k], "counted"):
                    element[1].gridpoints[i,j,k].counted = None
                    
        if printTreeProperties == True: 
            print("Tree build completed. \n"
                  "Domain Size:                 [%.1f, %.1f] \n"
                  "Tolerance1:                  %1.2e \n"
                  "Tolerance2:                  %1.2e \n"
                  "Total Number of Cells:       %i \n"
                  "Total Number of Gridpoints:  %i \n"
                  "Minimum Depth                %i levels \n"
                  "Maximum Depth:               %i levels \n"
                  "Construction time:           %.3g seconds." 
                  %(self.xmin, self.xmax, divideTolerance1, divideTolerance2, self.treeSize, self.numberOfGridpoints, self.minDepthAchieved,self.maxDepthAchieved,timer.elapsedTime))
        
            
                    
        
    def walkTree(self, attribute='', storeOutput = False, leavesOnly=False):  # walk through the tree, printing out specified data (in this case, the midpoints)
        '''
        Walk through the tree and either print or store the midpoint coordinates as well as the desired cell or midpoint attribute.
        Eventually, this should be modified to output either all grid points or just midpoints.
        Midpoints have the additional simplicity that they do not repeat.  When going to gridpoints in general,
        could add an adition attribute to the gridpoint class that indicates whether or not it has been walked over yet. 
        
        :param attribute:  the attribute you want printed or stored (in addition to coordinates)
        :param storeOutput: boolean, indicates if output should be stored or printed to screen
        '''
        
        def recursiveWalkMidpoint(Cell, attribute, storeOutput, outputArray, leavesOnly):  
            midpoint = Cell.gridpoints[1,1,1]
            truePsi = trueWavefunction(1, midpoint.x, midpoint.y, midpoint.z)
            
            # if not storing output, then print to screen
            if storeOutput == False:
                if (leavesOnly==False or not hasattr(Cell,'children')):
                    if hasattr(Cell,attribute):
                        print('Level: ', Cell.level,', midpoint: (', midpoint.x,', ',midpoint.y,', ',midpoint.z,'), ', attribute,': ', getattr(Cell,attribute))
                    elif hasattr(midpoint,attribute):
                        print('Level: ', Cell.level,', midpoint: (', midpoint.x,', ',midpoint.y,', ',midpoint.z,'), ', attribute,': ', getattr(midpoint,attribute))
                    else:
                        print('Level: ', Cell.level,', midpoint: (', midpoint.x,', ',midpoint.y,', ',midpoint.z,')')
            
            # if storing output in outputArray.  This will be a Nx4 dimension array containing the midpoint (x,y,z,attribute)
            elif storeOutput == True:
                if (leavesOnly==False or not hasattr(Cell,'children')):
                    if hasattr(Cell,attribute):
                        outputArray.append([midpoint.x,midpoint.y,midpoint.z,getattr(Cell,attribute)])
                    if hasattr(midpoint,attribute):
#                         outputArray.append([midpoint.x,midpoint.y,midpoint.z,np.log(abs(getattr(midpoint,attribute)-truePsi))])
                        outputArray.append([midpoint.x,midpoint.y,midpoint.z, getattr(midpoint,attribute)])
            
            # if cell has children, recursively walk through them
            if hasattr(Cell,'children'):
                if storeOutput == False: print()
                for i,j,k in TwoByTwoByTwo:
                    recursiveWalkMidpoint(Cell.children[i,j,k],attribute, storeOutput, outputArray,leavesOnly)
                if storeOutput == False: print()
            
            
        # initialize empty array for output.  Shape not known apriori
        outputArray = [] 
        # call the recursive walk on the root            
        recursiveWalkMidpoint(self.root, attribute, storeOutput, outputArray, leavesOnly)
        
        if storeOutput == True: # else the output waas printed to screen
            return np.array(outputArray)
        
    def visualizeMesh(self,attributeForColoring):
        '''
        Tasks-- modify the walk so I only get the final mesh, not the entire tree.
                figure out how to title and add colorbar.
        :param attributeForColoring:
        '''
    
                
#         outputData = self.walkTree(attribute='psi', storeOutput = True)        
        outputData = self.walkTree(attributeForColoring, storeOutput = True, leavesOnly=True)        
        x = outputData[:,0]
        y = outputData[:,1]
        z = outputData[:,2]
        psi = outputData[:,3]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(x, y, zs=z, c=psi, s=2, cmap=plt.get_cmap('brg'),depthshade=True)
        """ consider using s=cell.volume of volume^(2/3), and squares instead of circles """
#         plt.colorbar()
#         ax.set_title('Adaptive Mesh Colored by ', attributeForColoring)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        fig.colorbar(scatter, ax=ax)
        plt.show()
        
    def wavefunctionSlice(self,zSlizeLocation,n=-1,scalingFactor=1,saveID=False):
        '''
        Tasks-- modify the walk so I only get the final mesh, not the entire tree.
                figure out how to title and add colorbar.
        :param attributeForColoring:
        '''
        Etrue = trueEnergy(n)
        
        x=[]
        y=[]
        psi=[]
        psiTrue = []
        for element in self.masterList:
            for i,j in ThreeByThree:
                if element[1].gridpoints[i,j,2].z == zSlizeLocation:
                    x.append(element[1].gridpoints[i,j,2].x)
                    y.append(element[1].gridpoints[i,j,2].y)
                    psi.append(element[1].gridpoints[i,j,2].psi)
                    if n>= 0:
                        psiTrue.append(scalingFactor*trueWavefunction(n, element[1].gridpoints[i,j,2].x, element[1].gridpoints[i,j,2].y, element[1].gridpoints[i,j,2].z))

        print('Extracted %i gridpoints for the scatter plot.' %len(x))
        psi = np.array(psi)
        psiTrue = np.array(psiTrue)
        
#         if np.sum(psi-psiTrue) > np.sum(psi+psiTrue):
#             psi = -psi

        if np.sum(psi)*np.sum(psiTrue) < 0:
            psi = -psi
            
                
        if n>=0:
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(figsize=(8, 6), ncols=2, nrows=2)
            if saveID!=False: plt.suptitle('Iteration %s, Energy Error %.2e' %(saveID[-2:],(Etrue-self.E)) )
#             logAbsErr = np.log(abs(psiTrue-psi))
#             logRelErr = np.log(abs((psiTrue-psi)/psiTrue))
            analyticWave =  ax1.scatter(x, y, s=3, c=psiTrue, cmap=plt.get_cmap('Blues'))
            computedWave =  ax2.scatter(x, y, s=3, c=psi, cmap=plt.get_cmap('Blues'))
            absErr =        ax3.scatter(x, y, s=3, c=abs(psiTrue-psi), cmap=plt.get_cmap('Reds'),norm=matplotlib.colors.LogNorm())
#             absErr =        ax3.scatter(x, y, s=3, c=abs(psiTrue-psi), cmap=plt.get_cmap('Reds'))
#             absErr =        ax3.scatter(x, y, s=3, c=logAbsErr)
            relErr =        ax4.scatter(x, y, s=3, c=abs((psiTrue-psi)/psiTrue), cmap=plt.get_cmap('Reds'),norm=matplotlib.colors.LogNorm())
#             relErr =        ax4.scatter(x, y, s=3, c=abs((psiTrue-psi)/psiTrue), cmap=plt.get_cmap('Reds'))
#             relErr =        ax4.scatter(x, y, s=3, c=logRelErr)

            fig.colorbar(analyticWave, ax=ax1)
            fig.colorbar(computedWave, ax=ax2)
            fig.colorbar(absErr, ax=ax3)
            fig.colorbar(relErr, ax=ax4)
            
            ax1.set_title('Analytic Wavefunction')
            ax2.set_title('Computed Wavefunction')
            ax3.set_title('Log Absolute Error')
            ax4.set_title('Log Relative Error')

            
            if saveID!=False:
#                 plt.savefig(saveID, bbox_inches='tight',format='pdf')
                plt.savefig(saveID, bbox_inches='tight')
                plt.close(fig) 
            else: plt.show()
#             fig1 = plt.figure()
#             ax1 = fig1.add_subplot(111)
#             scatter = ax1.scatter(x, y, s=20, c=abs(np.array(psiTrue)-np.array(psi)), cmap=plt.get_cmap('brg'))
#             plt.title('Absolute Errors')
#             ax1.set_xlabel('X')
#             ax1.set_ylabel('Y')
#             fig1.colorbar(scatter, ax=ax1)
#             
#             fig2 = plt.figure()
#             ax2 = fig2.add_subplot(111)
#             scatter = ax2.scatter(x, y, s=20, c=abs(np.array(psiTrue)-np.array(psi))/np.array(psiTrue), cmap=plt.get_cmap('brg'))
#             plt.title('Relative Errors')
#             ax2.set_xlabel('X')
#             ax2.set_ylabel('Y')
#             fig1.colorbar(scatter, ax=ax2)
#             
#             fig3 = plt.figure()
#             ax3 = fig3.add_subplot(111)
#             scatter = ax3.scatter(x, y, s=20, c=psi, cmap=plt.get_cmap('brg'))
#             plt.title('Wavefunction')
#             ax1.set_xlabel('X')
#             ax1.set_ylabel('Y')
#             fig3.colorbar(scatter, ax=ax3)
            
            
        else:
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            scatter = ax1.scatter(x, y, s=20, c=psi, cmap=plt.get_cmap('brg'))
            plt.title('Wavefunction')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            fig1.colorbar(scatter, ax=ax1)
        plt.show()
     
    def computePotentialOnTree(self, epsilon=0, timePotential = False): 
    
        timer = Timer() 
 
        self.totalPotential = 0
        def recursiveComputePotential(Cell, epsilon=0):
            if hasattr(Cell,'children'):
                for i,j,k in TwoByTwoByTwo:
                    recursiveComputePotential(Cell.children[i,j,k])
            
            else: # this cell has no children
                Cell.computePotential(epsilon)
                self.totalPotential += Cell.PE
        timer.start()        
        recursiveComputePotential(self.root, epsilon)
        timer.stop() 
        if timePotential == True:
            self.PotentialTime = timer.elapsedTime
            
    def computePotentialOnList(self, epsilon=0, timePotential = False): 
       
        timer = Timer() 
 
        self.totalPotential = 0
        
        timer.start() 
        for element in self.masterList:
            Cell = element[1]
            if Cell.leaf == True:
                Cell.computePotential(epsilon)
                self.totalPotential += Cell.PE
                       
        timer.stop() 
        if timePotential == True:
            self.PotentialTime = timer.elapsedTime
    
    def computeKineticOnTree(self, timeKinetic = False):
        self.totalKinetic = 0
        def recursiveComputeKinetic(Cell):
            if hasattr(Cell,'children'):
                for i,j,k in TwoByTwoByTwo:
                    recursiveComputeKinetic(Cell.children[i,j,k])
            
            else: # this cell has no children
                Cell.computeKinetic()
                self.totalKinetic += Cell.KE
        timer = Timer()
        timer.start()
        recursiveComputeKinetic(self.root)
        timer.stop()
        if timeKinetic == True:
            self.KineticTime = timer.elapsedTime
            
    def computeKineticOnList(self, timeKinetic = False):

        
        self.totalKinetic = 0
        timer = Timer()
        timer.start()
        for element in self.masterList:
            Cell = element[1]
            if Cell.leaf == True:
                Cell.computeKinetic()
                self.totalKinetic += Cell.KE
        timer.stop()
        if timeKinetic == True:
            self.KineticTime = timer.elapsedTime
            
    def updateEnergy(self,epsilon=0.0):
        self.computeKineticOnList()
        self.computePotentialOnList(epsilon)
        self.E = self.totalKinetic + self.totalPotential
            
            
    def GreenFunctionConvolutionRecursive(self, timeConvolution = True):
        
        def GreenFunction(r,E):
            return np.exp(-np.sqrt(-2*E)*r)/(4*np.pi*r)
        
        def singleTargetConvolve(self,GridPoint):
            
            def recursivelyInteractWithSources(SourceCell,TargetGridpoint):
                psiNew = 0.0
                xt = TargetGridpoint.x
                yt = TargetGridpoint.y
                zt = TargetGridpoint.z
                if hasattr(SourceCell,'children'):
                    for i,j,k in TwoByTwoByTwo:
                        recursivelyInteractWithSources(SourceCell.children[i,j,k],TargetGridpoint)
                else:
                    xs = SourceCell.gridpoints[1,1,1].x
                    ys = SourceCell.gridpoints[1,1,1].y
                    zs = SourceCell.gridpoints[1,1,1].z
                    
                    
                    
                    dist = np.sqrt( (xs-xt)**2 + (ys-yt)**2 + (zs-zt)**2 )
                    if dist > 0:
                        psiNew += -2*SourceCell.volume * potential(xs,ys,zs) * GreenFunction(dist, self.E) * SourceCell.gridpoints[1,1,1].psi
                    
                return psiNew
                    
            
            GridPoint.psiNew = recursivelyInteractWithSources(self.root, GridPoint)
            
        
        def recursiveConvolutionForEachTarget(Cell):
            '''
            Performs the convolution for all the gridpoints.  This works by recursively walking the tree to 
            the leaves, then performing the convolution for all their gridpoints.  The gridpoints have an 
            attribute signaling whether or not they have been updated
            :param Cell:
            '''
            if hasattr(Cell,'children'):
                for i,j,k in TwoByTwoByTwo:
                    recursiveConvolutionForEachTarget(Cell.children[i,j,k])
            
            else: # this cell has no children
                for l,m,n in ThreeByThreeByThree:
                    if not hasattr(Cell.gridpoints[l,m,n],'psiNew'):
                        target = Cell.gridpoints[l,m,n]
                        singleTargetConvolve(self,target)
                
        def applyUpdate(Cell):
            '''
            All gridpoints store their new value in attribute psiNew, but don't update until the entire convolution 
            is complete.  Upon updating, psiNew attribute is set to None.  
            :param Cell:
            '''
            if hasattr(Cell,'children'):
                for i,j,k in TwoByTwoByTwo:
                    applyUpdate(Cell.children[i,j,k])
            else:
                for i,j,k in ThreeByThreeByThree:
                    Cell.gridpoints[i,j,k].psi = Cell.gridpoints[i,j,k].psiNew
                    Cell.gridpoints[i,j,k].psiNew = None
                    
        timer = Timer()
        timer.start()
        recursiveConvolutionForEachTarget(self.root)
        applyUpdate(self.root)
        timer.stop()
        if timeConvolution == True:
            self.ConvolutionTime = timer.elapsedTime
             
    
    def GreenFunctionConvolutionList(self,timeConvolution=True):
            
        def GreenFunction(r,E):
            return np.exp(-np.sqrt(-2*E)*r)/(4*np.pi*r)
        
        timer = Timer()
        timer.start()
        
        # initialize convolution flag to false.  This gets flipped to true when the convolution is performed for the gridpoint
        for element in self.masterList:
            if element[1].leaf == True:
                for i,j,k in ThreeByThreeByThree:
                    element[1].gridpoints[i,j,k].convolutionComplete = False
        
        for targetElement in self.masterList:
            if targetElement[1].leaf == True:
                targetCell = targetElement[1]
                
                for i,j,k in ThreeByThreeByThree:
                    targetPoint = targetCell.gridpoints[i,j,k] 
                    if targetPoint.convolutionComplete == False:
                        targetPoint.convolutionComplete = True
                        targetPoint.psiNew = 0.0
                        xt = targetPoint.x
                        yt = targetPoint.y
                        zt = targetPoint.z
                        
                        for sourceElement in self.masterList:
                            if sourceElement[1].leaf==True:
                                sourceCell = sourceElement[1]
                                sourceMidpoint = sourceCell.gridpoints[1,1,1]
                                xs = sourceMidpoint.x
                                ys = sourceMidpoint.y
                                zs = sourceMidpoint.z
                            
                                r = np.sqrt( (xt-xs)**2 + (yt-ys)**2 + (zt-zs)**2 )
                                if r > 0:
                                    targetPoint.psiNew += -2*sourceCell.volume * sourceCell.hydrogenV * np.exp(-np.sqrt(-2*self.E)*r)/(4*np.pi*r) * sourceMidpoint.psi
        
        
        for element in self.masterList:
            if element[1].leaf == True:
                element[1].gridpoints[1,1,1].psi = element[1].gridpoints[1,1,1].psiNew
                element[1].gridpoints[1,1,1].psiNew = None
                
        self.normalizeWavefunction()
#                 for i,j,k in ThreeByThreeByThree:
#                     element[1].gridpoints[i,j,k].psi = element[1].gridpoints[i,j,k].psiNew
#                     element[1].gridpoints[i,j,k].psiNew = None
            
        timer.stop()
        if timeConvolution == True:
            self.ConvolutionTime = timer.elapsedTime
            
            
    def computeWaveErrors(self,energyLevel,normalizationFactor):
        
        # need normalizationFactor because the analytic wavefunctions aren't normalized for this finite domain.
        maxErr = 0.0
        errorsIfSameSign = []
        errorsIfDifferentSign = []
        for element in self.masterList:
            if element[1].leaf == True:
                midpoint = element[1].gridpoints[1,1,1]
                errorsIfSameSign.append( (midpoint.psi - normalizationFactor*trueWavefunction(energyLevel,midpoint.x,midpoint.y,midpoint.z))**2 * element[1].volume)
                errorsIfDifferentSign.append( (midpoint.psi + normalizationFactor*trueWavefunction(energyLevel,midpoint.x,midpoint.y,midpoint.z))**2 * element[1].volume)
        
                absErr = abs(midpoint.psi - normalizationFactor*trueWavefunction(energyLevel,midpoint.x,midpoint.y,midpoint.z))
                if absErr > maxErr:
                    maxErr = absErr
        if np.sum(errorsIfSameSign) < np.sum(errorsIfDifferentSign):
            errors = errorsIfSameSign
        else:
            errors = errorsIfDifferentSign
                   
        self.L2NormError = np.sum(errors)
        self.maxCellError = np.max(errors)
        self.maxPointwiseError = maxErr
        
    def normalizeWavefunction(self):
        """ Compute integral psi*2 dxdydz """
        A = 0.0
        for element in self.masterList:
            if element[1].leaf == True:
                A += element[1].gridpoints[1,1,1].psi**2*element[1].volume
        
        print('A = ', A)        
        """ Initialize the normalization flag for each gridpoint """        
        for element in self.masterList:
            if element[1].leaf==True:
                for i,j,k in ThreeByThreeByThree:
                    element[1].gridpoints[i,j,k].normalized = False
        
        """ Rescale wavefunction values, flip the flag """
        for element in self.masterList:
            if element[1].leaf==True:
                for i,j,k in ThreeByThreeByThree:
                    if element[1].gridpoints[i,j,k].normalized == False:
                        element[1].gridpoints[i,j,k].psi /= np.sqrt(A)
                        element[1].gridpoints[i,j,k].normalized = True
        
        """  Delete the flag, if desired               
        for element in self.masterList:
            if element[1].leaf==True:
                for i,j,k in ThreeByThreeByThree:
                    element[1].gridpoints[i,j,k].normalized = None
            
        """    
        
    def orthogonalizeWavefunction(self,n):
        """ Orthgononalizes psi against wavefunction n """
        B = 0.0
        for element in self.masterList:
            if element[1].leaf == True:
                midpoint = element[1].gridpoints[1,1,1]
                B += midpoint.psi*midpoint.finalWavefunction[n]*element[1].volume
                
        """ Initialize the orthogonalization flag for each gridpoint """        
        for element in self.masterList:
            if element[1].leaf==True:
                for i,j,k in ThreeByThreeByThree:
                    element[1].gridpoints[i,j,k].orthogonalized = False
        
        """ Subtract the projection, flip the flag """
        for element in self.masterList:
            if element[1].leaf==True:
                for i,j,k in ThreeByThreeByThree:
                    gridpoint = element[1].gridpoints[i,j,k]
                    if gridpoint.orthogonalized == False:
                        gridpoint.psi -= B*gridpoint.finalWavefunction[n]
                        gridpoint.orthogonalized = True
                        
                        
    def extractLeavesMidpointsOnly(self):
        '''
        Extract the leaves as a Nx4 array [ [x1,y1,z1,psi1], [x2,y2,z2,psi2], ... ]
        '''
#         leaves = np.empty((self.numberOfGridpoints,4))
        leaves = []
        counter=0
        for element in self.masterList:
            if element[1].leaf == True:
                midpoint =  element[1].gridpoints[1,1,1]
                leaves.append( [midpoint.x, midpoint.y, midpoint.z, midpoint.psi, potential(midpoint.x, midpoint.y, midpoint.z), element[1].volume ] )
                counter+=1 
                
        return np.array(leaves)
    
    def extractLeavesAllGridpoints(self):
        '''
        Extract the leaves as a Nx4 array [ [x1,y1,z1,psi1], [x2,y2,z2,psi2], ... ]
        '''
        leaves = []
        for element in self.masterList:
            
            for i,j,k in ThreeByThreeByThree:
                element[1].gridpoints[i,j,k].extracted = False
                
        for element in self.masterList:
            if element[1].leaf == True:
                for i,j,k in ThreeByThreeByThree:
                    gridpt = element[1].gridpoints[i,j,k]
                    if gridpt.extracted == False:
                        leaves.append( [gridpt.x, gridpt.y, gridpt.z, gridpt.psi, potential(gridpt.x, gridpt.y, gridpt.z), element[1].volume ] )
                        gridpt.extracted = True
                    

        for element in self.masterList:
            for i,j,k in ThreeByThreeByThree:
                element[1].gridpoints[i,j,k].extracted = None
                
        return np.array(leaves)
                
    
    def importPsiOnLeaves(self,psiNew):
        '''
        Import psi values, apply to leaves
        '''
        for element in self.masterList:
            for i,j,k in ThreeByThreeByThree:
                element[1].gridpoints[i,j,k].psiImported = False
        importIndex = 0        
        for element in self.masterList:
            if element[1].leaf == True:
                for i,j,k in ThreeByThreeByThree:
                    gridpt = element[1].gridpoints[i,j,k]
                    if gridpt.psiImported == False:
                        gridpt.psi = psiNew[importIndex]
                        gridpt.psiImported = True
                        importIndex += 1
                    
        for element in self.masterList:
            for i,j,k in ThreeByThreeByThree:
                element[1].gridpoints[i,j,k].psiImported = None
                
    def copyPsiToFinalWavefunction(self, n):
        for element in self.masterList:
            for i,j,k in ThreeByThreeByThree:
                gridpt = element[1].gridpoints[i,j,k]
                if len(gridpt.finalWavefunction) == n:
                    gridpt.finalWavefunction.append(gridpt.psi)
                    
    def populatePsiWithAnalytic(self,n):
        for element in self.masterList:
            for i,j,k in ThreeByThreeByThree:
                element[1].gridpoints[i,j,k].setAnalyticPsi(n)
        self.normalizeWavefunction()

                            
            
def TestTreeForProfiling():
    xmin = ymin = zmin = -12
    xmax = ymax = zmax = -xmin
    tree = Tree(xmin,xmax,ymin,ymax,zmin,zmax)
    tree.buildTree( minLevels=4, maxLevels=4, divideTolerance=0.07,printTreeProperties=True)

def TestConvolutionForProfiling():
    xmin = ymin = zmin = -12
    xmax = ymax = zmax = -xmin
    tree = Tree(xmin,xmax,ymin,ymax,zmin,zmax)
    tree.buildTree( minLevels=3, maxLevels=3, divideTolerance=0.07,printTreeProperties=True) 
    tree.E = -0.5
#     print('\nUsing Tree')
#     tree.GreenFunctionConvolutionRecursive(timeConvolution=True)
#     print('Convolution took         %.4f seconds. ' %tree.ConvolutionTime)    
    
    print('\nUsing List')
    tree.GreenFunctionConvolutionList(timeConvolution=True)
    print('Convolution took         %.4f seconds. ' %tree.ConvolutionTime)
    
def visualizeWavefunction():
    xmin = ymin = zmin = -8
    xmax = ymax = zmax = -xmin
    tree = Tree(xmin,xmax,ymin,ymax,zmin,zmax)
    tree.buildTree( minLevels=5, maxLevels=6, divideTolerance=0.012,printTreeProperties=True)
    
#     tree.visualizeMesh('psi')
    tree.wavefunctionSlice(0.25)
           
        
if __name__ == "__main__":
#     TestTreeForProfiling()
    visualizeWavefunction()
    

    
    
       
    