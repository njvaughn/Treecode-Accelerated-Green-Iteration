'''
Created on Mar 5, 2018

@author: nathanvaughn
'''
import unittest
import sys
import numpy as np
import time
sys.path.append('../src/dataStructures')
sys.path.append('../src/utilities')
import itertools
ThreeByThreeByThree = [element for element in itertools.product(range(3),range(3),range(3))]
from numpy import pi, sqrt, exp
from scipy.special import erf


from TreeStruct_CC import Tree
from convolution import gpuPoissonConvolution, gpuPoissonConvolutionRegularized, gpuHartreeIterativeSubractSingularity, gpuHartreeIterative

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
            
            # set density on the secondary mesh 
            for i,j,k in cell.PxByPyByPz_density:
                dp = cell.densityPoints[i,j,k]
                r = sqrt( dp.x**2 + dp.y**2 + dp.z**2 )
                dp.rho = gaussianDensity(r,alpha)

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
    

class TestEnergyComputation(unittest.TestCase):


    @classmethod
    def setUpClass(self):
        '''
        setUp() gets called before doing the tests below.
        '''
        inputFile ='../src/utilities/molecularConfigurations/dummyAtomAuxiliary.csv'
#         inputFile ='../src/utilities/molecularConfigurations/oxygenAtomAuxiliary.csv'
#         inputFile ='../src/utilities/molecularConfigurations/carbonMonoxideAuxiliary.csv'
#         inputFile ='../src/utilities/molecularConfigurations/carbonAtomAuxiliary.csv'
        xmin = ymin = zmin = -20
        xmax = ymax = zmax =  20
        order=5
        minDepth=3
        maxDepth=20
        divideCriterion='LW5'
        divideParameter=1000
        self.alpha = 2
        
        [coordinateFile, outputFile] = np.genfromtxt(inputFile,dtype="|U100")[:2]
        [nElectrons, nOrbitals, Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal, gaugeShift] = np.genfromtxt(inputFile)[2:]
        nElectrons = int(nElectrons)
        nOrbitals = int(nOrbitals)
        
        print([coordinateFile, outputFile, nElectrons, nOrbitals, 
     Etotal, Eexchange, Ecorrelation, Eband, gaugeShift])
        self.tree = Tree(xmin,xmax,order,ymin,ymax,order,zmin,zmax,order,nElectrons,nOrbitals,maxDepthAtAtoms=maxDepth,gaugeShift=gaugeShift,
                coordinateFile=coordinateFile,inputFile=inputFile)#, iterationOutFile=outputFile)
    
        self.tree.buildTree( minLevels=minDepth, maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, divideParameter=divideParameter, printTreeProperties=True)
#         self.tree.occupations = np.array([2,2,4/3,4/3,4/3])
        self.tree.occupations = np.array([2,2])
        
        setDensityToGaussian(self.tree,self.alpha)
        setTrueHartree(self.tree,self.alpha)
        print()
        print()
    
    @unittest.skip('Skipping because the initialization is solid.')
    def testTheSetupOfDensity(self):
        print('Integrating the Gaussian density on both meshes.')
        self.tree.integrateDensityBothMeshes()

    @unittest.skip('Skipping because this is included in the test with numerical potential.')        
    def testEnergyWithAnalyticPotential(self):
        
        HartreeEnergyFromAnalyticPotential = computeHartreeEnergyFromAnalyticPotential(self.tree)
        TrueHartreeEnergy = hartreeEnergy(self.alpha)
        print('True Hartree Energy:  ',TrueHartreeEnergy)
        print('Hartree Energy computed from analytic potential:  ',HartreeEnergyFromAnalyticPotential)
        print('Error: ', (HartreeEnergyFromAnalyticPotential-TrueHartreeEnergy))
        self.assertAlmostEqual(TrueHartreeEnergy, HartreeEnergyFromAnalyticPotential, 3, 
                               "Analytic Energy and Energy computed from analytic potential not agreeing well enough")

    @unittest.skip('Skipping single and staggered mesh comparison')
    def testHartreeSolve_staggeredMesh(self):
        print()
        targets = self.tree.extractLeavesDensity()  
        sources = targets   # extract density on secondary mesh
        
        
        threadsPerBlock = 512
        blocksPerGrid = (self.tree.numberOfGridpoints + (threadsPerBlock - 1)) // threadsPerBlock  # compute the number of blocks based on N and threadsPerBlock
    
        V_HartreeNew = np.zeros((len(targets)))
        gpuPoissonConvolution[blocksPerGrid, threadsPerBlock](targets,sources,V_HartreeNew)  # call the GPU convolution
        self.tree.importVcoulombOnLeaves(V_HartreeNew)
        self.tree.updateVxcAndVeffAtQuadpoints()
        
        computedFromNumericalPotential = computeHartreeEnergyFromNumericalPotential(self.tree)
        computedFromAnalyticPotential = computeHartreeEnergyFromAnalyticPotential(self.tree)
        trueHartreeEnergy = hartreeEnergy(self.alpha)
        
        # Now perform the calculation using the secondary mesh so that the Convolution doesn't have any r=r' singularity
        sources = self.tree.extractDenstiySecondaryMesh()   # extract density on secondary mesh
        V_HartreeNew = np.zeros((len(targets)))
        gpuPoissonConvolution[blocksPerGrid, threadsPerBlock](targets,sources,V_HartreeNew)  # call the GPU convolution
        self.tree.importVcoulombOnLeaves(V_HartreeNew)
        self.tree.updateVxcAndVeffAtQuadpoints()
        computedFromNumericalPotentialSecondaryMesh = computeHartreeEnergyFromNumericalPotential(self.tree)

        
        print('True Hartree Energy:                                 ', trueHartreeEnergy)
        print('Computed from Analytic Potential:                    ', computedFromAnalyticPotential)
        print('Computed from Numerical Potential (same mesh):       ', computedFromNumericalPotential)
        print('Computed from Numerical Potential (secondary mesh) : ', computedFromNumericalPotentialSecondaryMesh)
        print()
        print('Error for Analytic Potential:                   %1.3e' %(computedFromAnalyticPotential - trueHartreeEnergy))
        print('Error for Computed Potential (same mesh):      %1.3e' %(computedFromNumericalPotential - trueHartreeEnergy))
        print('Error for Computed Potential (secondary mesh): %1.3e' %(computedFromNumericalPotentialSecondaryMesh - trueHartreeEnergy))
       
    @unittest.skip('Skipping uniform epsilon test') 
    def testHartreeSolve_uniformEpsilon(self):
        print()
        
        computedFromAnalyticPotential = computeHartreeEnergyFromAnalyticPotential(self.tree)
        trueHartreeEnergy = hartreeEnergy(self.alpha)
        print('True Hartree Energy:                                       ', trueHartreeEnergy, '\n')
        print('Computed from Analytic Potential:                          ', computedFromAnalyticPotential)
        print('Error for Analytic Potential:                               %1.3e \n' %(computedFromAnalyticPotential - trueHartreeEnergy))
        
        
        targets = self.tree.extractLeavesDensity()  
        sources = targets   # extract density on secondary mesh
        threadsPerBlock = 512
        blocksPerGrid = (self.tree.numberOfGridpoints + (threadsPerBlock - 1)) // threadsPerBlock  # compute the number of blocks based on N and threadsPerBlock

        epsilon = 2    
        while epsilon > 0.0001:
            V_HartreeNew = np.zeros((len(targets)))
            gpuPoissonConvolutionRegularized[blocksPerGrid, threadsPerBlock](targets,sources,V_HartreeNew, epsilon)  # call the GPU convolution
            self.tree.importVcoulombOnLeaves(V_HartreeNew)
            self.tree.updateVxcAndVeffAtQuadpoints()
            computedFromNumericalPotential = computeHartreeEnergyFromNumericalPotential(self.tree)
            print('Computed from Numerical Potential (epsilon = %f):       %1.3e' %(epsilon,computedFromNumericalPotential) )
            print('Error for Computed Potential:                                %1.3e \n' %(computedFromNumericalPotential - trueHartreeEnergy))
            epsilon = epsilon/2
            
    def testHartreeSolve_iterativeHelmholtz(self):
        print()
        
        computedFromAnalyticPotential = computeHartreeEnergyFromAnalyticPotential(self.tree)
        trueHartreeEnergy = hartreeEnergy(self.alpha)
        print('True Hartree Energy:                                       ', trueHartreeEnergy, '\n')
        print('Computed from Analytic Potential:                          ', computedFromAnalyticPotential)
        print('Error for Analytic Potential:                               %1.3e \n' %(computedFromAnalyticPotential - trueHartreeEnergy))
        
        
        targets = self.tree.extractLeavesDensity()  
        sources = targets   # extract density on secondary mesh
        weights = np.copy(targets[:,4])
#         print(np.shape(targets[:,3]))
#         print(np.shape(weights))
        threadsPerBlock = 512
        blocksPerGrid = (self.tree.numberOfGridpoints + (threadsPerBlock - 1)) // threadsPerBlock  # compute the number of blocks based on N and threadsPerBlock

        # Compute an initial guess for V_Hartree just using singularity skipping
        V_HartreeOld = np.zeros((len(targets)))
        gpuPoissonConvolution[blocksPerGrid, threadsPerBlock](targets,sources,V_HartreeOld)  # call the GPU convolution
        self.tree.importVcoulombOnLeaves(V_HartreeOld)
        self.tree.updateVxcAndVeffAtQuadpoints()
        computedFromNumericalPotential = computeHartreeEnergyFromNumericalPotential(self.tree)
        print('Computed from Numerical Potential (iteration 0):             %1.3e' %(computedFromNumericalPotential) )
        print('Error for Computed Potential:                                %1.3e\n' %(computedFromNumericalPotential - trueHartreeEnergy))
        
        print(np.min(V_HartreeOld ))
        print(np.max(V_HartreeOld ))
        print('Location of max')
        idx = np.argmax(np.abs(V_HartreeOld))
        print(sources[idx,0:3])
        print('Location of min')
        idx = np.argmin(np.abs(V_HartreeOld))
        print(sources[idx,0:3])
        print()
        
        minVal = gaussianHartree(np.sqrt(3*19.8**2), self.alpha)
        
#         print('Randomizing V_HartreeOld')
#         V_HartreeOld = np.random.rand((len(targets)))
        
        beta = 1
        hartreeResidual = 1
        count = 1
        while hartreeResidual > 1e-6:
            
            ## modify the integrand to be (rho - beta**2 V_hartree) for the previous iterate of V_Hartree
            ## gpuHartreeIterativeSubractSingularity expects the integrand as t 
            
            modifiedTargets = np.copy(targets)
            modifiedSources = np.copy(sources)
            
            r = np.sqrt(modifiedSources[:,0]**2 + modifiedSources[:,1]**2 + modifiedSources[:,2]**2)
            trueVhartree = gaussianHartree(r,self.alpha)
            
            
#             print(modifiedSources[0:5,3])
#             print(np.min(modifiedSources[:,3] ))
#             print(np.max(modifiedSources[:,3] ))
            modifiedTargets[:,3] *= -4*pi
            modifiedSources[:,3] *= -4*pi
            modifiedTargets[:,3] -= beta**2 * V_HartreeOld
            modifiedSources[:,3] -= beta**2 * V_HartreeOld
#             modifiedTargets[:,3] -= beta**2 * trueVhartree
#             modifiedSources[:,3] -= beta**2 * trueVhartree
            

            

            ##
            V_HartreeNew = np.zeros((len(targets)))
#             gpuHartreeIterative[blocksPerGrid, threadsPerBlock](modifiedTargets,modifiedSources,V_HartreeNew, beta)  # call the GPU convolution
            gpuHartreeIterativeSubractSingularity[blocksPerGrid, threadsPerBlock](modifiedTargets,modifiedSources,V_HartreeNew, beta)  # call the GPU convolution
#             print(V_HartreeNew[0:5])
           
#             self.tree.importVcoulombOnLeaves(( V_HartreeNew + V_HartreeOld ) /2 )
            self.tree.importVcoulombOnLeaves(V_HartreeNew)
            self.tree.updateVxcAndVeffAtQuadpoints()
            computedFromNumericalPotential = computeHartreeEnergyFromNumericalPotential(self.tree)
            print('Computed from Numerical Potential (iteration %i):             %1.3e' %(count,computedFromNumericalPotential) )
            print('Error for Computed Potential:                                %1.3e' %(computedFromNumericalPotential - trueHartreeEnergy))
#             print('Shift to match min val')
#             currentMinVal = np.min(V_HartreeNew )
#             V_HartreeNew *= (minVal / currentMinVal)
#             self.tree.importVcoulombOnLeaves(V_HartreeNew)
#             self.tree.updateVxcAndVeffAtQuadpoints()
#             computedFromNumericalPotential = computeHartreeEnergyFromNumericalPotential(self.tree)
#             print('After scaling...')
#             print('Computed from Numerical Potential (iteration %i):             %1.3e' %(count,computedFromNumericalPotential) )
#             print('Error for Computed Potential:                                %1.3e' %(computedFromNumericalPotential - trueHartreeEnergy))
            
            
            # Compute residual            
            hartreeResidual = np.sqrt( np.sum( (V_HartreeOld-V_HartreeNew)**2*weights ) )
            print('Residual:                                                   ', hartreeResidual)
            print()
            
            print(np.min(V_HartreeNew ))
            print(np.max(V_HartreeNew ))
            print('Location of max')
            idx = np.argmax(np.abs(V_HartreeNew))
            print(modifiedSources[idx,0:3])
            print('Location of min')
            idx = np.argmin(np.abs(V_HartreeNew))
            print(modifiedSources[idx,0:3])
            print()
            V_HartreeOld = np.copy(V_HartreeNew)
            count += 1
        
      
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()