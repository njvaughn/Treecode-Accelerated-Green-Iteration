'''
Created on Mar 5, 2018

@author: nathanvaughn
'''
import numpy as np
import itertools
from GridpointStruct import GridPoint
from CellStruct import Cell
from hydrogenPotential import potential, trueWavefunction
from timer import Timer
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ThreeByThreeByThree = [element for element in itertools.product(range(3),range(3),range(3))]
TwoByTwoByTwo = [element for element in itertools.product(range(2),range(2),range(2))]
FiveByFiveByFive = [element for element in itertools.product(range(5),range(5),range(5))]

class Tree(object):
    '''
    Tree object. Constructed of cells, which are composed of gridpoint objects.
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
            
    def buildTree(self,minLevels,maxLevels,divideTolerance,printNumberOfCells=False, printTreeProperties = True): # call the recursive divison on the root of the tree
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
                    Cell.divide(printNumberOfCells)
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
        if printTreeProperties == True: 
            print("Tree build completed. \n"
                  "Domain Size:             [%.1f, %.1f] \n"
                  "Tolerance:               %1.2e \n"
                  "Total Number of Cells:   %i \n"
                  "Minimum Depth            %i levels \n"
                  "Maximum Depth:           %i levels \n"
                  "Construction time:       %.3g seconds." 
                  %(self.xmin, self.xmax, divideTolerance, self.treeSize, self.minDepthAchieved,self.maxDepthAchieved,timer.elapsedTime))
        
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
                        outputArray.append([midpoint.x,midpoint.y,midpoint.z,np.log(abs(getattr(midpoint,attribute)-truePsi))])
            
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
#         plt.colorbar()
#         ax.set_title('Adaptive Mesh Colored by ', attributeForColoring)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        fig.colorbar(scatter, ax=ax)
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
        for targetElement in self.masterList:
            if targetElement[1].leaf == True:
                targetCell = targetElement[1]
                
                targetPoint = targetCell.gridpoints[1,1,1]
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
                        
                        
#                         for i,j,k in ThreeByThreeByThree:
#                             targetPoint = targetCell.gridpoints[i,j,k]
#                             targetPoint.psiNew = 0.0
#                             xt = targetPoint.x
#                             yt = targetPoint.y
#                             zt = targetPoint.z
#                             
#                             r = np.sqrt( (xt-xs)**2 + (yt-ys)**2 + (zt-zs)**2 )
#                             if r > 0:
#                                 targetPoint.psiNew += -2*sourceCell.volume * sourceCell.hydrogenV * np.exp(-np.sqrt(-2*self.E)*r)/(4*np.pi*r) * sourceMidpoint.psi
                                
                        
                        
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
            
            
    def computeWaveErrors(self):
        errors = []
        for element in self.masterList:
            if element[1].leaf == True:
                midpoint = element[1].gridpoints[1,1,1]
                errors.append( (midpoint.psi - trueWavefunction(1,midpoint.x,midpoint.y,midpoint.z))**2 * element[1].volume)
                
        self.L2NormError = np.sum(errors)
        self.maxCellError = np.max(errors)
        
    def normalizeWavefunction(self):
        A = 0.0
        for element in self.masterList:
            if element[1].leaf == True:
                A += element[1].gridpoints[1,1,1].psi**2*element[1].volume
        
        for element in self.masterList:
            if element[1].leaf==True:
                element[1].gridpoints[1,1,1].psi /= np.sqrt(A)
            
            
            
            
            
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
        
if __name__ == "__main__":
    TestTreeForProfiling()

    
    
       
    