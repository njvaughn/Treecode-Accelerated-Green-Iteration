import numpy as np
from scipy.interpolate import RegularGridInterpolator
import itertools
from hydrogenPotential import potential, trueWavefunction
from timer import Timer
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# plt.ion()
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
        self.root = Cell( gridpoints )
        self.root.level = 0
            
    def buildTree(self,minLevels,maxLevels,divideTolerance): # call the recursive divison on the root of the tree
        # max depth returns the maximum depth of the tree.  maxLevels is the limit on how large the tree is allowed to be,
        # regardless of division criteria
        timer = Timer()
        def recursiveDivide(Cell, minLevels, maxLevels, divideTolerance, counter, maxDepthAchieved=0, minDepthAchieved=100, currentLevel=0):
            counter += 1
            if currentLevel < maxLevels:
                
                if currentLevel < minLevels:
                    Cell.divideFlag = True 
                else:                             
                    Cell.checkIfCellShouldDivide(divideTolerance)
                    
                if Cell.divideFlag == True:   
                    Cell.divide()
                    for i,j,k in TwoByTwoByTwo:
                        maxDepthAchieved, minDepthAchieved, counter = recursiveDivide(Cell.children[i,j,k], minLevels, maxLevels, divideTolerance, counter, maxDepthAchieved, minDepthAchieved, currentLevel+1)
                else:
                    minDepthAchieved = min(minDepthAchieved, currentLevel)
                    
                    
            maxDepthAchieved = max(maxDepthAchieved, currentLevel)                                                                                                                                                       
            return maxDepthAchieved, minDepthAchieved, counter
        
        timer.start()
        self.maxDepthAchieved, self.minDepthAchieved, self.treeSize = recursiveDivide(self.root, minLevels, maxLevels, divideTolerance, counter=0, maxDepthAchieved=0, minDepthAchieved=maxLevels, currentLevel=0 )
        timer.stop()
        print('Tree build completed. \nTolerance:               %1.2e \nTotal Number of Cells:   %i \nMinimum Depth            %i levels \nMaximum Depth:           %i levels \nConstruction time:       %.3g seconds.' %(divideTolerance,self.treeSize, self.minDepthAchieved,self.maxDepthAchieved,timer.elapsedTime()))
        
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
                        outputArray.append([midpoint.x,midpoint.y,midpoint.z,getattr(midpoint,attribute)])
            
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
        scatter = ax.scatter(x, y, zs=z, c=psi, s=2, cmap=plt.get_cmap('brg'),depthshade=False)
#         plt.colorbar()
#         ax.set_title('Adaptive Mesh Colored by ', attributeForColoring)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        fig.colorbar(scatter, ax=ax)
        plt.show()
     
    def computePotentialOnTree(self, epsilon=0):    
        self.totalPotential = 0
        def recursiveComputePotential(Cell, epsilon=0):
            if hasattr(Cell,'children'):
                for i,j,k in TwoByTwoByTwo:
                    recursiveComputePotential(Cell.children[i,j,k])
            
            else: # this cell has no children
                Cell.computePotential(epsilon)
                self.totalPotential += Cell.PE
                
        recursiveComputePotential(self.root, epsilon=0)
    
    def computeKineticOnTree(self):
        self.totalKinetic = 0
        def recursiveComputeKinetic(Cell):
            if hasattr(Cell,'children'):
                for i,j,k in TwoByTwoByTwo:
                    recursiveComputeKinetic(Cell.children[i,j,k])
            
            else: # this cell has no children
                Cell.computeKinetic()
                self.totalKinetic += Cell.KE
                
        recursiveComputeKinetic(self.root)
    
            
class Cell(object):
    '''
    Cell object.  Cells are composed of gridpoint objects.
    '''
    def __init__(self, gridpoints):
        '''
        Cell Constructor.  Cell composed of gridpoint objects
        '''
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
    
    def checkIfCellShouldDivide(self, divideTolerance):
        # perform midpoint method for integral(testFunction) for parent cell
        # If parent integral differs from the sum of children integrals over the test function
        # then set divideFlag to true.
        self.gridpoints[1,1,1].setTestFunctionValue()
        parentIntegral = self.gridpoints[1,1,1].testFunctionValue*self.volume
        
        childrenIntegral = 0.0
        xmids = np.array([(3*self.xmin+self.xmax)/4, (self.xmin+3*self.xmax)/4])
        ymids = np.array([(3*self.ymin+self.ymax)/4, (self.ymin+3*self.ymax)/4])
        zmids = np.array([(3*self.zmin+self.zmax)/4, (self.zmin+3*self.zmax)/4])
        for i,j,k in TwoByTwoByTwo:
            tempChild = GridPoint(xmids[i],ymids[j],zmids[k])
            tempChild.setTestFunctionValue()
            childrenIntegral += tempChild.testFunctionValue*(self.volume/8)
        
        if abs(parentIntegral-childrenIntegral) > divideTolerance:
            self.divideFlag = True
        else:
            self.divideFlag = False
        
    def divide(self):
        children = np.empty((2,2,2), dtype=object)
        x = np.linspace(self.xmin,self.xmax,5)
        y = np.linspace(self.ymin,self.ymax,5)
        z = np.linspace(self.zmin,self.zmax,5)
        gridpoints = np.empty((5,5,5),dtype=object)
        gridpoints[::2,::2,::2] = self.gridpoints  # AVOIDS DUPLICATION OF GRIDPOINTS.  The 5x5x5 array of gridpoints should have the original 3x3x3 objects within
        
        for i, j, k in FiveByFiveByFive:
            if gridpoints[i,j,k] == None:
                gridpoints[i,j,k] = GridPoint(x[i],y[j],z[k])
        
        # generate the 8 children cells        
        for i, j, k in TwoByTwoByTwo:
            children[i,j,k] = Cell(gridpoints[2*i:2*i+3, 2*j:2*j+3, 2*k:2*k+3])
            children[i,j,k].parent = self # children should point to their parent
            if hasattr(self,'level'):
                children[i,j,k].level = self.level+1
        
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
        
          
class GridPoint(object):
    '''
    The gridpoint object.  Will contain the coordinates, wavefunction value, and any other metadata such as
    whether or not the wavefunction has been updated, which cells the gridpoint belongs to, etc.
    '''
    def __init__(self, x,y,z):
        '''
        Gridpoint Constructor.  For minimal example, a gridpoint simply has x and y values.
        '''
        self.x = x
        self.y = y
        self.z = z
        self.setAnalyticPsi()  # for now, set the analytic psi value.  Eventually, need to use interpolator
    
    def setPsi(self, psi):
        self.psi = psi
    
    def setAnalyticPsi(self):
        self.psi = trueWavefunction(1, self.x,self.y,self.z)
        
    def setTestFunctionValue(self):
        '''
        Set the test function value.  For now, this can be the single atom single electron wavefunction.
        Generally, this should be some representative function that we can use apriori to set up the 
        refined mesh.  Bikash uses single atom densities, or a sum of single atom densities to give an 
        indication of where he should refine before actually computing the many-atom electron density.
        '''
        self.testFunctionValue = trueWavefunction(1, self.x,self.y,self.z)  
        
class Mesh(object):  
    '''
    Mesh object.  Will be constructed out of many cells, which will be constructed out of gridpoints.
    This will be used if we start with pre-existing mesh and want to set up the cell structure.
    Ideally, we start with a blank slate and can build the tree straight away, never using this mesh object.
    '''
    def __init__(self, xmin,xmax,nx,ymin,ymax,ny,zmin,zmax,nz):
        '''
        Mesh constructor:  
        First construct the gridpoints.  
        Then construct the cells that are composed of gridpoints. 
        Then construct the mesh as a 2D or 3D array of cells.
        '''
        # generate gridpoint objects.  2n+1 gridpoints if there are n cells in a given dimension.  
        # These will sit together in memory, and Cell objects will point to 9 of them (for 2D).
        xvec = np.linspace(xmin,xmax,2*nx+1)
        yvec = np.linspace(ymin,ymax,2*ny+1)
        zvec = np.linspace(zmin,zmax,2*nz+1)
        gridpoints = np.empty((2*nx+1,2*ny+1, 2*nz+1),dtype=object)
        for i in range(2*nx+1):
            for j in range(2*ny+1):
                for k in range(2*nz+1):
                    gridpoints[i,j,k] = GridPoint(xvec[i],yvec[j],zvec[k])
        
        # generate cells from the gridpoint objects  
        self.cells = np.empty((nx,ny,nz),dtype=object)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    self.cells[i,j,k] = Cell( gridpoints[2*i:2*i+3, 2*j:2*j+3, 2*k:2*k+3] )
                    
                    
                    
if __name__ == "__main__":
#     print('hi')
    xmin = ymin = zmin = -10
    xmax = ymax = zmax = -xmin
    tree = Tree(xmin,xmax,ymin,ymax,zmin,zmax)
    tree.buildTree( minLevels=1, maxLevels=6, divideTolerance=0.00125)
         
    tree.computePotentialOnTree(epsilon=0)
    print('\nPotential Error:        %.3g mHartree' %float((-1.0-tree.totalPotential)*1000.0))
         
    tree.computeKineticOnTree()
    print('Kinetic Error:           %.3g mHartree' %float((0.5-tree.totalKinetic)*1000.0))
             
    tree.visualizeMesh(attributeForColoring='psi')
    print('Visualization Complete.')
        
        
        