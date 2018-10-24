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
from numpy import pi, sqrt, exp
from scipy.special import erf


from TreeStruct_CC import Tree
from convolution import gpuPoissonConvolution, gpuPoissonConvolutionRegularized, gpuHartreeIterativeSubractSingularity, gpuHartreeIterative, gpuHartreeShiftedPoisson, gpuHartreeShiftedPoisson_singularitySubtract, gpuHartreeGaussianSingularitySubract
from meshUtilities import ChebLaplacian3D

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

def evaluateLaplacianOfPhi(tree):
    rhoError = 0.0
    relRhoError = 0.0
    
    maxTrueRho = 0.0
    maxComputedRho = 0.0
    minTrueRho = 0.0
    minComputedRho = 0.0
    
    phiError = 0.0
    relPhiError = 0.0
    
    maxTruePhi = 0.0
    maxComputedPhi = 0.0
    minTruePhi = 0.0
    minComputedPhi = 0.0
    worstCell = 0.0
    
    for _,cell in tree.masterList:
        if cell.leaf == True:
            rho = np.zeros((cell.px,cell.py,cell.pz))
            rhoTrue = np.zeros((cell.px,cell.py,cell.pz))
            phi = np.zeros((cell.px,cell.py,cell.pz))
            phiTrue = np.zeros((cell.px,cell.py,cell.pz))
            
            for i,j,k in cell.PxByPyByPz:
                phiTrue[i,j,k] = cell.gridpoints[i,j,k].trueHartree
                rhoTrue[i,j,k] = cell.gridpoints[i,j,k].rho
                maxTrueRho = max(maxTrueRho , (4*pi)*cell.gridpoints[i,j,k].rho)
                minTrueRho = min(minTrueRho , (4*pi)*cell.gridpoints[i,j,k].rho)
                maxTruePhi = max(maxTruePhi , cell.gridpoints[i,j,k].trueHartree)
                minTruePhi = min(minTrueRho , cell.gridpoints[i,j,k].trueHartree)
            
#             phix = np.dot( cell.DopenX, phiTrue )
#             phixx = np.dot( cell.DopenX, phix )
#             phiy = np.dot( cell.DopenY, phiTrue )
#             phiyy = np.dot( cell.DopenY, phiy )
#             phiz = np.dot( cell.DopenZ, phiTrue )
#             phizz = np.dot( cell.DopenZ, phiz )
#             rho = ( phixx + phiyy + phizz ) / (-4*pi) 
            rho = ChebLaplacian3D(cell.DopenX, cell.DopenY, cell.DopenZ, cell.px, phiTrue) / (-4*pi)

            rhoError += np.sum( (rho-rhoTrue)**2 * cell.w )
            relRhoError += np.sum( (rhoTrue)**2 * cell.w )
            
            if np.sum( (rho-rhoTrue)**2 * cell.w ) > worstCell:
                worstCell = np.sum( (rho-rhoTrue)**2 * cell.w )
                print('New worst cell: x,y,z, error:',cell.xmid, cell.ymid, cell.zmid, np.sum( (rho-rhoTrue)**2 * cell.w ))
#                 print()
#                 print('computed rho: ', rho)
#                 print()
#                 print('4pi*true rho: ', rhoTrue)
#                 print()
                      
            
            maxComputedRho = max( maxComputedRho, np.max(rho)) 
            minComputedRho = min( maxComputedRho, np.min(rho)) 
            
            
#             phi = -np.dot(cell.inverseLaplacian,-4*pi*phiTrue) 
#             phiError += np.sum( (phi-phiTrue)**2 * cell.w )
#             relPhiError += np.sum( (phiTrue)**2 * cell.w )
#             
#             maxComputedPhi = max( maxComputedPhi, np.max(phi)) 
#             minComputedPhi = min( maxComputedPhi, np.min(phi)) 
            
            
            
            
    relRhoError = rhoError / relRhoError
    relRhoError = np.sqrt(relRhoError)
    rhoError = np.sqrt(rhoError)
    
#     relPhiError = phiError / relPhiError
#     relPhiError = np.sqrt(relPhiError)
#     phiError = np.sqrt(phiError)
    
    
    print('~'*50)
    print('Density Calculation Results')
    print('Absolute L2 error in computed density: ', rhoError)
    print('Relative L2 error in computed density: ', relRhoError)
    print('Maximum true rho:                      ', maxTrueRho)
    print('Maximum computed rho:                  ', maxComputedRho)
    print('Minimum true rho:                      ', minTrueRho)
    print('Minimum computed rho:                  ', minComputedRho)
    print('~'*50)
#     print('~'*50)
#     print('Potential Calculation Results')
#     print('Absolute L2 error in computed potential: ', phiError)
#     print('Relative L2 error in computed potential: ', relPhiError)
#     print('Maximum true phi:     ', maxTruePhi)
#     print('Maximum computed phi: ', maxComputedPhi)
#     print('Minimum true phi:     ', minTruePhi)
#     print('Minimum computed phi: ', minComputedPhi)      
#     print('~'*50)
     
    

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
        self.alpha = 2  # alpha for the Gaussian charge density.  NOT THE SAME AS THE ALPHA FOR THE SINGULARITY SUBTRACTION OR THE POISSON-REGULARIZATION
        
        
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
        alpha = 1.5 # for the Gaussian singularity subtraction 
        alphasq = alpha*alpha
        print('Using Gaussian singularity subtraction, alpha = ', alpha)
        gpuHartreeGaussianSingularitySubract[blocksPerGrid, threadsPerBlock](targets,sources,V_HartreeNew, alphasq)  # call the GPU convolution
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
      
    @unittest.skip('Skipping Iterative solve')  
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

#         r = np.sqrt(sources[:,0]**2 + sources[:,1]**2 + sources[:,2]**2)
#         trueVhartree = gaussianHartree(r,self.alpha)
            
        # Compute an initial guess for V_Hartree just using singularity skipping            
        V_HartreeOld = np.zeros((len(targets)))
        gpuPoissonConvolution[blocksPerGrid, threadsPerBlock](targets,sources,V_HartreeOld)  # call the GPU convolution
#         print('Initializing with analytic phiH')
#         V_HartreeOld = trueVhartree
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
        
        beta = 4
        
        
        """ Rework to match the x^n+1 = Bx^n + c framework """
        modifiedTargets = np.copy(targets)
        modifiedSources = np.copy(sources)
        modifiedTargets[:,3] *= -4*pi
        modifiedSources[:,3] *= -4*pi
        
        c = np.zeros((len(targets)))
        gpuHartreeIterative[blocksPerGrid, threadsPerBlock](modifiedTargets,modifiedSources,c, beta)
#         gpuHartreeIterativeSubractSingularity[blocksPerGrid, threadsPerBlock](modifiedTargets,modifiedSources,c, beta)  # call the GPU convolution
#         print('Sum of c: ', np.sum(c))
            
        hartreeResidual = 1
        count = 1
        
        
        while hartreeResidual > 1e-6:
            
            ## modify the integrand to be (rho - beta**2 V_hartree) for the previous iterate of V_Hartree
            ## gpuHartreeIterativeSubractSingularity expects the integrand as t 
            
#             modifiedTargets = np.copy(targets)
#             modifiedSources = np.copy(sources)
            
#             r = np.sqrt(modifiedSources[:,0]**2 + modifiedSources[:,1]**2 + modifiedSources[:,2]**2)
#             trueVhartree = gaussianHartree(r,self.alpha)
# #             modifiedTargets[:,3] -= beta**2 * trueVhartree
# #             modifiedSources[:,3] -= beta**2 * trueVhartree
#             
            
# #             print(modifiedSources[0:5,3])
# #             print(np.min(modifiedSources[:,3] ))
# #             print(np.max(modifiedSources[:,3] ))
#             modifiedTargets[:,3] *= -4*pi
#             modifiedSources[:,3] *= -4*pi
#             modifiedTargets[:,3] -= beta**2 * V_HartreeOld
#             modifiedSources[:,3] -= beta**2 * V_HartreeOld



            modifiedTargets[:,3] = -beta**2 * V_HartreeOld
            modifiedSources[:,3] = -beta**2 * V_HartreeOld
            

            

            ##
            V_HartreeNew = np.zeros((len(targets)))
            gpuHartreeIterative[blocksPerGrid, threadsPerBlock](modifiedTargets,modifiedSources,V_HartreeNew, beta)  # call the GPU convolution
#             gpuHartreeIterativeSubractSingularity[blocksPerGrid, threadsPerBlock](modifiedTargets,modifiedSources,V_HartreeNew, beta)  # call the GPU convolution
            V_HartreeNew += c  # add in the c 
#             print('Sum of V_HartreeNew: ', np.sum(V_HartreeNew))
#             print(V_HartreeNew[0:5])
           
#             self.tree.importVcoulombOnLeaves(( 0.25*V_HartreeNew + 0.75*V_HartreeOld ) )
            self.tree.importVcoulombOnLeaves(V_HartreeNew)
            self.tree.updateVxcAndVeffAtQuadpoints()
            computedFromNumericalPotential = computeHartreeEnergyFromNumericalPotential(self.tree)
            print('Computed from Numerical Potential (iteration %i):             %1.3e' %(count,computedFromNumericalPotential) )
            print('Error for Computed Potential:                                %1.3e' %(computedFromNumericalPotential - trueHartreeEnergy))
#             print('Shift to match min val')
#             currentMinVal = np.min(V_HartreeNew )
#             V_HartreeNew += (minVal - currentMinVal)
# #             V_HartreeNew *= (minVal / currentMinVal)
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
    
    @unittest.skip('Skipping power iteration')       
    def testPowerIteration(self):
        print()
    
        
        targets = self.tree.extractLeavesDensity()  
        sources = targets   # extract density on secondary mesh
        weights = np.copy(targets[:,4])

        threadsPerBlock = 512
        blocksPerGrid = (self.tree.numberOfGridpoints + (threadsPerBlock - 1)) // threadsPerBlock  # compute the number of blocks based on N and threadsPerBlock
                
        
        print('Randomizing initial vector')
        x = np.random.rand((len(targets)))
        x /= np.sqrt( x**2 *weights )
        y = np.random.rand((len(targets)))
#         y -= np.sqrt( y*x *weights ) * x
        y /= np.sqrt( y**2 *weights )
        beta = 1/2
        
        
        """ Rework to match the x^n+1 = Bx^n + c framework """
        modifiedTargets = np.copy(targets)
        modifiedSources = np.copy(sources)
        
        
            
        eigenvalueResidual = 1
        xoldEigenvalue = 0
        yoldEigenvalue = 0
        count = 1
        
        
        while eigenvalueResidual > 1e-6:
            
            # Apply the operator
            modifiedTargets[:,3] = -beta**2*x
            modifiedSources[:,3] = -beta**2*x
#             gpuHartreeIterative[blocksPerGrid, threadsPerBlock](modifiedTargets,modifiedSources,x, beta)
            gpuHartreeIterativeSubractSingularity[blocksPerGrid, threadsPerBlock](modifiedTargets,modifiedSources,x, beta)
            
            # Compute the Rayleigh Quotient
            xeigenvalue = np.sqrt( np.sum( x**2*weights ) )
            print('Iteration %i' %count)
            print('<x|x> =     ', xeigenvalue)
            x /= xeigenvalue
            
            
#             # Apply the the operator, orthogonalize against x
#             modifiedTargets[:,3] = -beta**2*y
#             modifiedSources[:,3] = -beta**2*y
#             gpuHartreeIterative[blocksPerGrid, threadsPerBlock](modifiedTargets,modifiedSources,y, beta)
#             y -= np.sqrt( y*x *weights ) * x
#             # Compute the Rayleigh Quotient
#             yeigenvalue = np.sqrt( np.sum( y**2*weights ) )
#             print('<y|y> =     ', yeigenvalue)
#             y /= yeigenvalue
            
            # Update residual and counter
#             eigenvalueResidual = max( abs(xoldEigenvalue - xeigenvalue), abs(yoldEigenvalue - yeigenvalue) )
            eigenvalueResidual = abs(xoldEigenvalue - xeigenvalue)
            print('Residual =  ', eigenvalueResidual)
            print()
            
            xoldEigenvalue = xeigenvalue
#             yoldEigenvalue = yeigenvalue
            
            
            count += 1
            
    @unittest.skip('Skipping the Laplacian evaluation')       
    def testLaplacian(self):
        print()
        """ Load in true phi and rho, then compute using Laplacian and inverse Laplacian """
        
#         targets = self.tree.extractLeavesDensity()  
#         sources = targets   # extract density on secondary mesh
#         weights = np.copy(targets[:,4])

#         r = np.sqrt(sources[:,0]**2 + sources[:,1]**2 + sources[:,2]**2)
#         phiTrue = gaussianHartree(r,self.alpha)
#         rhoTrue = gaussianDensity(r,self.alpha)
        
#         """  Compute the Laplacian matrices for all leaf cells """
#         start = time.time()
#         for _,cell in self.tree.masterList:
#             if cell.leaf==True:
#                 cell.computeLaplacianAndInverse()
#         end = time.time()
#         print('Computing laplacians took ', (end-start), ' seconds.')
                
        computedRho = evaluateLaplacianOfPhi(self.tree)

    @unittest.skip('Skipping the alpha-regularization test')       
    def testAlphaRegularization(self):
        """ Idea: Instead of Poisson, solve the Helmholtz equation with a sequence of alphas approaching zero """
        
        
        print()
        
        computedFromAnalyticPotential = computeHartreeEnergyFromAnalyticPotential(self.tree)
        trueHartreeEnergy = hartreeEnergy(self.alpha)
        print('True Hartree Energy:                                       ', trueHartreeEnergy, '\n')
        print('Computed from Analytic Potential:                          ', computedFromAnalyticPotential)
        print('Error for Analytic Potential:                               %1.3e \n' %(computedFromAnalyticPotential - trueHartreeEnergy))
        
        targets = self.tree.extractLeavesDensity()  
        sources = targets   # extract density on secondary mesh
        weights = np.copy(targets[:,4])

        threadsPerBlock = 512
        blocksPerGrid = (self.tree.numberOfGridpoints + (threadsPerBlock - 1)) // threadsPerBlock  # compute the number of blocks based on N and threadsPerBlock
        
        for alpha in [2.0, 1.0, 0.5, 0.25, 0.125]:
            print('alpha = ', alpha)
            
            
            
    
    #         r = np.sqrt(sources[:,0]**2 + sources[:,1]**2 + sources[:,2]**2)
    #         trueVhartree = gaussianHartree(r,self.alpha)
                
            # Compute an initial guess for V_Hartree just using singularity skipping            
            V_Hartree = np.zeros((len(targets)))
#             gpuHartreeShiftedPoisson[blocksPerGrid, threadsPerBlock](targets,sources,V_Hartree,alpha)  # call the GPU convolution
            gpuHartreeShiftedPoisson_singularitySubtract[blocksPerGrid, threadsPerBlock](targets,sources,V_Hartree,alpha)  # call the GPU convolution
#             gpuHartreeIterativeSubractSingularity[blocksPerGrid, threadsPerBlock](targets,sources,V_HartreeOld)
#             gpuPoissonConvolution[blocksPerGrid, threadsPerBlock](targets,sources,V_Hartree)  # call the GPU convolution
            #gpuHelmholtzConvolution[blocksPerGrid, threadsPerBlock](targets,sources,V_HartreeOld,alpha)

            self.tree.importVcoulombOnLeaves(V_Hartree)
            self.tree.updateVxcAndVeffAtQuadpoints()
            computedFromNumericalPotential = computeHartreeEnergyFromNumericalPotential(self.tree)
            print('Computed from Numerical Potential (iteration 0):             %1.3e' %(computedFromNumericalPotential) )
            print('Error for Computed Potential:                                %1.3e\n' %(computedFromNumericalPotential - trueHartreeEnergy))
            
                
        
        
        
      
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()