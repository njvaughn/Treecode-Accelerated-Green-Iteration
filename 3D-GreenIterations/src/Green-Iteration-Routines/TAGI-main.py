'''
testGreenIterations.py
This is a unitTest module for testing Green iterations.  It begins by building the tree-based
adaotively refined mesh, then performs Green iterations to obtain the ground state energy
and wavefunction for the single electron hydrogen atom.  -- 03/20/2018 NV

Created on Mar 13, 2018
@author: nathanvaughn
'''
import os
import sys
import time
import inspect
import resource
import unittest
import numpy as np
import pylibxc
import itertools
import csv
from scipy.optimize import anderson
from scipy.optimize import root as scipyRoot
from scipy.special import sph_harm
from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkTriangle, VtkQuad, VtkPolygon, VtkVoxel, VtkHexahedron
import mpi4py.MPI as MPI



sys.path.insert(1, '/Users/nathanvaughn/Documents/GitHub/TAGI/3D-GreenIterations/src/utilities')
sys.path.insert(1, '/Users/nathanvaughn/Documents/GitHub/TAGI/3D-GreenIterations/src/dataStructures')
sys.path.insert(1, '/home/njvaughn/TAGI/3D-GreenIterations/src/utilities')
# from loadBalancer import loadBalance
from mpiUtilities import global_dot, scatterArrays, rprint
from mpiMeshBuilding import  buildMeshFromMinimumDepthCells
# try:
#     import treecodeWrappers_distributed as treecodeWrappers
# except ImportError:
#     rprint(rank,'Unable to import treecodeWrapper due to ImportError')
# except OSError:
#     print('Unable to import treecodeWrapper due to OSError')
#     import treecodeWrappers_distributed as treecodeWrappers
# from TreeStruct_CC import Tree
from CellStruct_CC import Cell
from GridpointStruct import GridPoint
import densityMixingSchemes as densityMixing
from scfFixedPoint import scfFixedPointClosure
from scfFixedPointSimultaneous import scfFixedPointClosureSimultaneous
from scfFixedPointGreedy import scfFixedPointClosureGreedy, sortByEigenvalue, fermiObjectiveFunctionClosure
import moveData_wrapper as MOVEDATA

# Temperature = 500
# KB = 1/315774.6
# Sigma = Temperature*KB
# def fermiObjectiveFunctionClosure(Energies,nElectrons):
#     def fermiObjectiveFunction(fermiEnergy):
#                 exponentialArg = (Energies['orbitalEnergies']-fermiEnergy)/Sigma
#                 temp = 1/(1+np.exp( exponentialArg ) )
#                 return nElectrons - 2 * np.sum(temp)
#     return fermiObjectiveFunction



# if os.uname()[1] == 'Nathans-MacBook-Pro.local':
#     rootDirectory = '/Users/nathanvaughn/Documents/GitHub/TAGI/3D-GreenIterations/'
# else:
#     rprint(rank,'os.uname()[1] = ', os.uname()[1])


## Read in command line arguments


n=1
domainSize          = int(sys.argv[n]); n+=1
maxSideLength       = float(sys.argv[n]); n+=1
order               = int(sys.argv[n]); n+=1
fine_order          = int(sys.argv[n]); n+=1
subtractSingularity = int(sys.argv[n]); n+=1
gaussianAlpha       = float(sys.argv[n]); n+=1
gaugeShift          = float(sys.argv[n]); n+=1
divideCriterion     = str(sys.argv[n]); n+=1
divideParameter1    = float(sys.argv[n]); n+=1 
divideParameter2    = float(sys.argv[n]); n+=1
scfTolerance        = float(sys.argv[n]); n+=1
initialGItolerance  = float(sys.argv[n]); n+=1
finalGItolerance    = float(sys.argv[n]); n+=1
gradualSteps        = int(sys.argv[n]); n+=1
outputFile          = str(sys.argv[n]); n+=1
inputFile           = str(sys.argv[n]); n+=1
coreRepresentation  = str(sys.argv[n]); n+=1
srcdir              = str(sys.argv[n]); n+=1
vtkDir              = str(sys.argv[n]); n+=1
noGradients         = str(sys.argv[n]) ; n+=1
symmetricIteration  = str(sys.argv[n]) ; n+=1
mixingScheme        = str(sys.argv[n]); n+=1
mixingParameter     = float(sys.argv[n]); n+=1
mixingHistoryCutoff = int(sys.argv[n]) ; n+=1
GPUpresent          = str(sys.argv[n]); n+=1
treecode            = str(sys.argv[n]); n+=1
treecodeOrder       = int(sys.argv[n]); n+=1
theta               = float(sys.argv[n]); n+=1
maxParNode          = int(sys.argv[n]); n+=1
batchSize           = int(sys.argv[n]); n+=1
divideParameter3    = float(sys.argv[n]); n+=1
divideParameter4    = int(sys.argv[n]); n+=1
restart             = str(sys.argv[n]); n+=1
savedMesh           = str(sys.argv[n]); n+=1
singularityHandling = str(sys.argv[n]); n+=1
approximationName   = str(sys.argv[n]); n+=1
regularize          = str(sys.argv[n]); n+=1
epsilon             = float(sys.argv[n]); n+=1
TwoMeshStart        = int(sys.argv[n]); n+=1
GI_form             = str(sys.argv[n]); n+=1



# Set up paths based on srcdir
inputFile = srcdir+inputFile
sys.path.append(srcdir+'dataStructures')
sys.path.append(srcdir+'Green-Iteration-Routines')
sys.path.append(srcdir+'utilities')
sys.path.append(srcdir+'../ctypesTests/src')

sys.path.append(srcdir+'../ctypesTests')
sys.path.append(srcdir+'../ctypesTests/lib') 







if savedMesh == 'None':
    savedMesh=''
    
if regularize=='True':
    regularize=True
elif regularize=='False':
    regularize=False
else:
    print("What should regularize input be set to?")
    exit(-1)

if noGradients=='True':
    gradientFree=True

    
elif noGradients=='False':
    gradientFree=False
elif noGradients=='Laplacian':
    gradientFree='Laplacian'
else:
    rprint(rank,'Warning, not correct input for gradientFree')
    
if symmetricIteration=='True':
    symmetricIteration=True
elif symmetricIteration=='False':
    symmetricIteration=False
else:
    rprint(rank,'Warning, not correct input for gradientFree')

if restart=='True':
    restart=True
elif restart=='False':
    restart=False
else:
    rprint(rank,'Warning, not correct input for restart')
    
if GPUpresent=='True':
    GPUpresent=True
elif GPUpresent=='False':
    GPUpresent=False
else:
    rprint(rank,'Warning, not correct input for GPUpresent')
if treecode=='True':
    treecode=True
elif treecode=='False':
    treecode=False
else:
    rprint(rank,'Warning, not correct input for treecode')


vtkFileBase='/home/njvaughn/results_CO/orbitals'
# print("Converting command line arguments to correct types.")

def global_dot(u,v,comm):
    local_dot = np.dot(u,v)
    global_dot = comm.allreduce(local_dot)
    return global_dot


def clenshawCurtisNormClosure(W):
    def clenshawCurtisNorm(psi):
        norm = np.sqrt( global_dot( psi, psi*W, comm ) )
        return norm
    return clenshawCurtisNorm


def initializeOrbitalsRandomly(atoms,coreRepresentation,orbitals,nOrbitals,X,Y,Z):
    print("INITIALIZING ORBITALS RANDOMLY")
#     orbitals = np.zeros(nOrbitals,len(X))
    R2=X*X+Y*Y+Z*Z
    R = np.sqrt(R2)
    for m in range(nOrbitals):
        orbitals[m,:] = np.exp(-R)*np.sin(m*R)
    return orbitals

    
    
def initializeOrbitalsFromAtomicDataExternally(atoms,coreRepresentation,orbitals,nOrbitals,X,Y,Z,W): 
        aufbauList = ['10',                                     # n+ell = 1
                      '20',                                     # n+ell = 2
                      '21', '30',                               # n+ell = 3
                      '31', '40', 
                      '32', '41', '50'
                      '42', '51', '60'
                      '43', '52', '61', '70']

        orbitalIndex=0
        initialOccupations=[]
        initialEnergies=[]
        
        for atom in atoms:
            nAtomicOrbitals = atom.nAtomicOrbitals
            electronCount=0
                
            
            
            rprint(rank,'Initializing orbitals for atom Z = %i (%i atomic orbitals) located at (x, y, z) = (%6.3f, %6.3f, %6.3f)' 
                      %(atom.atomicNumber, atom.nAtomicOrbitals, atom.x,atom.y,atom.z))
            rprint(rank,'Orbital index = %i'%orbitalIndex)            
            singleAtomOrbitalCount=0
            for nell in aufbauList:
                
                if singleAtomOrbitalCount< nAtomicOrbitals:  
                    n = int(nell[0])
                    ell = int(nell[1])
                    psiID = 'psi'+str(n)+str(ell)
#                     print('Using ', psiID)
                    # filling these requires 2 * (2*ell+1) electrons
                    if atom.nuclearCharge - electronCount - 2*(2*ell+1) >= 0:
                        occupation=2.0
                    else:
                        occupation = (atom.nuclearCharge - electronCount) / ((2*ell+1))
                    for m in range(-ell,ell+1):
                            
                        
                        if psiID in atom.interpolators:  # pseudopotentials don't start from 10, 20, 21,... they start from the valence, such as 30, 31, ...
                            
                            initialEnergies.append( -atom.nuclearCharge / (n+ell))  # provide initial eigenvalue guess.  Higher (n,ell) closer to zero.
                            initialOccupations.append(occupation)
                            electronCount+=occupation
                            print("Initial (n,ell,m)=(%i,%i,%i) wavefunction gets occupation %f" %(n,ell,m,occupation))
                            print("Accounted for %f of %i electrons now." %(electronCount,atom.nuclearCharge))
                            #input()
                            
                            dx = X-atom.x
                            dy = Y-atom.y
                            dz = Z-atom.z
                            phi = np.zeros(len(dx))
                            r = np.sqrt( dx**2 + dy**2 + dz**2 )
                            inclination = np.arccos(dz/r)
    #                         print('Type(dx): ', type(dx))
    #                         print('Type(dy): ', type(dy))
    #                         print('Shape(dx): ', np.shape(dx))
    #                         print('Shape(dy): ', np.shape(dy))
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
                                print('imag(Y) ', np.imag(Ysp))
                                return
    #                                     Y = np.real(sph_harm(m,ell,azimuthal,inclination))
    #                         phi = atom.interpolators[psiID](r)*np.real(Y)
                            try:
                                phi = atom.interpolators[psiID](r)*np.real(Ysp) / np.sqrt(4*np.pi*r*r) 
                            except ValueError:
                                phi = 0.0   # if outside the interpolation range, assume 0.
                            except KeyError:
                                for key, value in atom.interpolators.items() :
                                    print (key, value)
                                exit(-1)
                            
                            
                            integralPhiSquared = global_dot(phi**2,W,comm)
                            phi /= np.sqrt(integralPhiSquared)
                            
                            orbitals[orbitalIndex,:] = np.copy(phi)
    #                         self.importPhiOnLeaves(phi, orbitalIndex)
    #                         self.normalizeOrbital(orbitalIndex)
                            
                            rprint(rank,'Orbital %i filled with (n,ell,m) = (%i,%i,%i) ' %(orbitalIndex,n,ell,m))
                            orbitalIndex += 1
                            singleAtomOrbitalCount += 1
                    
#                 else:
#                     n = int(nell[0])
#                     ell = int(nell[1])
#                     psiID = 'psi'+str(n)+str(ell)
#                     print('Not using ', psiID)
                        
        if orbitalIndex < nOrbitals:
            rprint(rank,"Didn't fill all the orbitals.  Should you initialize more?  Randomly, or using more single atom data?")
#             print('Filling extra orbitals with decaying exponential.')
            rprint(rank,'Filling extra orbitals with random initial data.')
            for ii in range(orbitalIndex, nOrbitals):
                R = np.sqrt(X*X+Y*Y+Z*Z)
#                 orbitals[:,ii] = np.exp(-R)*np.sin(R)
                orbitals[ii,:] = np.random.rand(len(R))
                initialOccupations.append(0.0)
                initialEnergies.append(-0.1)
#                 self.initializeOrbitalsRandomly(targetOrbital=ii)
#                 self.initializeOrbitalsToDecayingExponential(targetOrbital=ii)
#                 self.orthonormalizeOrbitals(targetOrbital=ii)
        if orbitalIndex > nOrbitals:
            rprint(rank,"Filled too many orbitals, somehow.  That should have thrown an error and never reached this point.")
                        

#         
#         for m in range(self.nOrbitals):
#             self.normalizeOrbital(m)

        rprint(rank, "initial occupations: ",initialOccupations)
        #input()
        return orbitals, initialOccupations, initialEnergies
    
def initializeDensityFromAtomicDataExternally(x,y,z,w,atoms,coreRepresentation):
        
    rho = np.zeros(len(x))
    
    totalElectrons = 0
    for atom in atoms:
        
        r = np.sqrt( (x-atom.x)**2 + (y-atom.y)**2 + (z-atom.z)**2 )
        
        if coreRepresentation=="AllElectron":
            totalElectrons += atom.atomicNumber
            try:
                rho += atom.interpolators['density'](r)
            except ValueError:
                rho += 0.0   # if outside the interpolation range, assume 0.
        elif coreRepresentation=="Pseudopotential":
            totalElectrons += atom.PSP.psp['header']['z_valence']
            rho += atom.PSP.evaluateDensityInterpolator(r)
            
        rprint(rank,"max density: ", max(abs(rho)))
        rprint(rank,"cumulative number of electrons: ", totalElectrons)


#     print("NOT NORMALIZING INITIAL DENSITY.")
    rho *= totalElectrons / global_dot(rho,w,comm)
    
    return rho





def testGreenIterationsGPU_rootfinding(X,Y,Z,W,Xf,Yf,Zf,Wf,pointsPerCell_coarse, pointsPerCell_fine,RHO,orbitals,eigenvalues,initialOccupations,atoms,coreRepresentation,nPoints,nOrbitals,nElectrons,referenceEigenvalues,vtkExport=False,onTheFlyRefinement=False, maxOrbitals=None, maxSCFIterations=None, restartFile=None):
    
    startTime = time.time()
    

    
    Energies, Rho, Times = greenIterations_KohnSham_SCF_rootfinding(X,Y,Z,W,Xf,Yf,Zf,Wf,pointsPerCell_coarse, pointsPerCell_fine,RHO,orbitals,eigenvalues,initialOccupations,atoms,coreRepresentation,nPoints,nOrbitals,nElectrons,referenceEigenvalues,
                                scfTolerance, initialGItolerance, finalGItolerance, gradualSteps,
                                gradientFree, symmetricIteration, GPUpresent, treecode, treecodeOrder, theta, maxParNode, batchSize, 
                                singularityHandling,approximationName,
                                 mixingScheme, mixingParameter, mixingHistoryCutoff,
                                 subtractSingularity, gaussianAlpha, gaugeShift,
                                 inputFile=inputFile,outputFile=outputFile, restartFile=restart,
                                 onTheFlyRefinement=onTheFlyRefinement, vtkExport=False, maxOrbitals=maxOrbitals, maxSCFIterations=maxSCFIterations,
                                 regularize=regularize,epsilon=epsilon)


    if rank==0:
        Times['totalKohnShamTime'] = time.time()-startTime
        print('Total Time: ', Times['totalKohnShamTime'])
    
        header = ['domainSize','maxSideLength','order','fineOrder','numberOfCells','numberOfPoints','gradientFree',
                  'divideCriterion','divideParameter1','divideParameter2','divideParameter3','divideParameter4',
                  'gaussianAlpha','gaugeShift','finalGItolerance',
                  'GreenSingSubtracted', 'regularize', 'epsilon',
                  'orbitalEnergies', 'BandEnergy', 'KineticEnergy',
                  'ExchangeEnergy','CorrelationEnergy','HartreeEnergy','TotalEnergy',
                  'Treecode','approximationName','treecodeOrder','theta','maxParNode','batchSize','totalTime','timePerConvolution','totalIterationCount']
        
        myData = [domainSize,maxSideLength,order,fine_order, nPoints/order**3,nPoints,gradientFree,
                  divideCriterion,divideParameter1,divideParameter2,divideParameter3,divideParameter4,
                  gaussianAlpha,gaugeShift,finalGItolerance,
                  subtractSingularity, regularize, epsilon,
                  Energies['orbitalEnergies']-Energies['gaugeShift'], Energies['Eband'], Energies['kinetic'], Energies['Ex'], Energies['Ec'], Energies['Ehartree'], Energies['Etotal'],
                  treecode,approximationName,treecodeOrder,theta,maxParNode,batchSize, Times['totalKohnShamTime'],Times['timePerConvolution'],Times['totalIterationCount'] ]
    #               Energies['Etotal'], tree.
    #               Energies['Etotal'], Energies['orbitalEnergies'][0], abs(Energies['Etotal']+1.1373748), abs(Energies['orbitalEnergies'][0]+0.378665)]
        
    
        runComparisonFile = os.path.split(outputFile)[0] + '/runComparison.csv'
        
        if not os.path.isfile(runComparisonFile):
            myFile = open(runComparisonFile, 'a')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerow(header) 
            
            
        
        myFile = open(runComparisonFile, 'a')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerow(myData)    
    
    
    return Rho


    



def greenIterations_KohnSham_SCF_rootfinding(X,Y,Z,W,Xf,Yf,Zf,Wf,pointsPerCell_coarse, pointsPerCell_fine,RHO,orbitals,eigenvalues,initialOccupations,atoms,coreRepresentation,nPoints,nOrbitals,nElectrons,referenceEigenvalues,
                                             SCFtolerance, initialGItolerance, finalGItolerance, gradualSteps, 
                                             gradientFree, symmetricIteration, GPUpresent, 
                                 treecode, treecodeOrder, theta, maxParNode, batchSize, singularityHandling, approximationName,
                                 mixingScheme, mixingParameter, mixingHistoryCutoff,
                                subtractSingularity, gaussianAlpha, gaugeShift, inputFile='',outputFile='',restartFile=False,
                                onTheFlyRefinement = False, vtkExport=False, outputErrors=False, maxOrbitals=None, maxSCFIterations=None,
                                regularize=False, epsilon=0.0): 
    '''
    Green Iterations for Kohn-Sham DFT using Clenshaw-Curtis quadrature.
    '''

    ### Determine nearby atoms.  Note, this does not work due to the global dot products in the atom nonlocal potential evaluation.
    nearbyAtoms = atoms
#     xbounds = [np.min(X), np.max(X)]
#     ybounds = [np.min(Y), np.max(Y)]
#     zbounds = [np.min(Z), np.max(Z)]
#     nearbyAtoms = np.empty((0,),dtype=object)
#     for atom in atoms:
#         if ((atom.x>xbounds[0]) and (atom.x<xbounds[1]) and (atom.y>ybounds[0]) and (atom.y<ybounds[1]) and (atom.z>zbounds[0]) and (atom.z<zbounds[1])):
#             # atom is in this processor's local domain
#             nearbyAtoms = np.append(nearbyAtoms,atom)
# #             print("Trying to append")
#         else:
#             added=False
# #             radius = np.sqrt( (xbounds[1]-xbounds[0])**2 + (ybounds[1]-ybounds[0])**2 + (zbounds[1]-zbounds[0])**2 )
# 
#             xmid = (xbounds[1]+xbounds[0])/2
#             ymid = (ybounds[1]+ybounds[0])/2
#             zmid = (zbounds[1]+zbounds[0])/2
#             
#             xwidth = xbounds[1]-xbounds[0]
#             ywidth = ybounds[1]-ybounds[0]
#             zwidth = zbounds[1]-zbounds[0]
#             
# 
#             dx = max(abs(atom.x - xmid) - xwidth / 2, 0);
#             dy = max(abs(atom.y - ymid) - ywidth / 2, 0);
#             dz = max(abs(atom.z - zmid) - zwidth / 2, 0);
#             
#             dist = np.sqrt( dx*dx + dy*dy + dz*dz )
#             print("dist = ", dist)
#             if ( (dist<4) and (added==False) ):
#                 
# #                 print("Trying to append")
#                 nearbyAtoms = np.append(nearbyAtoms,atom)
#                 added=True
#             
#             
#     print(np.shape(nearbyAtoms))
             
            
    
    verbosity=0
    polarization="unpolarized"
    exchangeFunctional="LDA_X"
    correlationFunctional="LDA_C_PZ"
    exchangeFunctional = pylibxc.LibXCFunctional(exchangeFunctional, polarization)
    correlationFunctional = pylibxc.LibXCFunctional(correlationFunctional, polarization)
    
    Vext_local = np.zeros(nPoints)
    Vext_local_fine = np.zeros(len(Xf))
    atomCount=1
#     for atom in nearbyAtoms:
    for atom in atoms:
        if coreRepresentation=="Pseudopotential":
            atom.generateChi(X,Y,Z)
            atom.generateFineChi(Xf,Yf,Zf)
            rprint(rank,"Generated projectors and set V_ext_local for atom %i" %atomCount)
        atomCount+=1
    atomCount=1       
    for atom in atoms:
        if coreRepresentation=="AllElectron":
            Vext_local += atom.V_all_electron(X,Y,Z)
        elif coreRepresentation=="Pseudopotential":
            Vext_local += atom.V_local_pseudopotential(X,Y,Z)
            Vext_local_fine += atom.V_local_pseudopotential(Xf,Yf,Zf)
            
#             print("NORMALIZING CHIS")
#             atom.normalizeChi(W,comm)
#             atom.normalizeFineChi(Wf,comm)
#             for i in range(atom.numberOfChis):
#                 norm_of_chi = global_dot(W,atom.Chi[str(i)]**2,comm)
#                 atom.Chi[str(i)] *= np.sqrt( 1.0/norm_of_chi )
#                 print("NORMALIZING CHI TO 1")
        else:
            print("Error: what should coreRepresentation be?")
            exit(-1)
        atomCount+=1
        
#     print('Does X exist in greenIterations_KohnSham_SCF_rootfinding()? ', len(X))
#     print('Does RHO exist in greenIterations_KohnSham_SCF_rootfinding()? ', len(RHO))
    
    Energies={}
#     Energies['orbitalEnergies'] = -1*np.ones(nOrbitals)
    Energies['orbitalEnergies'] = eigenvalues
    Energies['gaugeShift'] = gaugeShift
    Energies['kinetic'] = 0.0
    Energies['Enuclear'] = 0.0
    
    for atom1 in atoms:
        for atom2 in atoms:
            if atom1!=atom2:
                r = np.sqrt( (atom1.x-atom2.x)**2 + (atom1.y-atom2.y)**2 + (atom1.z-atom2.z)**2 )
#                 Energies['Enuclear'] += atom1.atomicNumber*atom2.atomicNumber/r
                Energies['Enuclear'] += atom1.nuclearCharge*atom2.nuclearCharge/r
    Energies['Enuclear'] /= 2 # because of double counting
    
    Times={}
    
    
    
    
#     return
    if verbosity>0: rprint(rank,'MEMORY USAGE: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    # Store Tree variables locally
    gaugeShift = Energies['gaugeShift']
    
    

    greenIterationOutFile = outputFile[:-4]+'_GREEN_'+str(nPoints)+'_'+str(len(Xf))+outputFile[-4:]
    SCFiterationOutFile =   outputFile[:-4]+'_SCF_'+str(nPoints)+'_'+str(len(Xf))+outputFile[-4:]
    densityPlotsDir =       outputFile[:-4]+'_SCF_'+str(nPoints)+'_plots'
    restartFilesDir =       '/home/njvaughn/restartFiles/'+'restartFiles_'+str(nPoints)
#     restartFilesDir =       '/home/njvaughn/restartFiles/restartFiles_1416000_after25'
#     restartFilesDir =       '/Users/nathanvaughn/Documents/synchronizedDataFiles/restartFiles_1416000_after25'
    wavefunctionFile =      restartFilesDir+'/wavefunctions'
    densityFile =           restartFilesDir+'/density'
    inputDensityFile =      restartFilesDir+'/inputdensity'
    outputDensityFile =     restartFilesDir+'/outputdensity'
    vHartreeFile =          restartFilesDir+'/vHartree'
    auxiliaryFile =         restartFilesDir+'/auxiliary'
    
    plotSliceOfDensity=False
    if plotSliceOfDensity==True:
        try:
            os.mkdir(densityPlotsDir)
        except OSError:
            print('Unable to make directory ', densityPlotsDir)
        
    if rank==0:
        try:
            os.mkdir(restartFilesDir)
        except OSError:
            rprint(rank,'Unable to make restart directory ', restartFilesDir)
    
    
#     tr = tracker.SummaryTracker()   
    if restartFile!=False:
        print("Not ready to handle restarts in mpi version.")
        return
#         orbitals = np.load(wavefunctionFile+'.npy')
#         oldOrbitals = np.copy(orbitals)
# #         for m in range(nOrbitals): 
# #             tree.importPhiOnLeaves(orbitals[:,m], m)
#         RHO = np.load(densityFile+'.npy')
#         
#         inputDensities = np.load(inputDensityFile+'.npy')
#         outputDensities = np.load(outputDensityFile+'.npy')
#         
#         V_hartreeNew = np.load(vHartreeFile+'.npy')
#         
#         
#         # make and save dictionary
#         auxiliaryRestartData = np.load(auxiliaryFile+'.npy').item()
#         rprint(rank,'type of aux: ', type(auxiliaryRestartData))
#         SCFcount = auxiliaryRestartData['SCFcount']
#         Times['totalIterationCount'] = auxiliaryRestartData['totalIterationCount']
#         Energies['orbitalEnergies'] = auxiliaryRestartData['eigenvalues'] 
#         Energies['Eold'] = auxiliaryRestartData['Eold']
#         
#         
#         
#         Energies['Ehartree'] = 1/2*np.sum(W * RHO * V_hartreeNew)
#         exchangeOutput = exchangeFunctional.compute(RHO)
#         correlationOutput = correlationFunctional.compute(RHO)
#         Energies['Ex'] = np.sum( W * RHO * np.reshape(exchangeOutput['zk'],np.shape(RHO)) )
#         Energies['Ec'] = np.sum( W * RHO * np.reshape(correlationOutput['zk'],np.shape(RHO)) )
#         
#         Vx = np.reshape(exchangeOutput['vrho'],np.shape(RHO))
#         Vc = np.reshape(correlationOutput['vrho'],np.shape(RHO))
#         
#         Energies['Vx'] = np.sum(W * RHO * Vx)
#         Energies['Vc'] = np.sum(W * RHO * Vc)
#          
#         Veff = V_hartreeNew + Vx + Vc + Vext_local + gaugeShift
        
        
    
    else: 
        Eold = -10
        SCFcount=0
        Times['totalIterationCount'] = 0

        inputDensities = np.zeros((nPoints,1))
        outputDensities = np.zeros((nPoints,1))
        
        inputDensities[:,0] = np.copy(RHO)
        oldOrbitals = np.copy(orbitals)

#     tr.print_diff()
    
    if plotSliceOfDensity==True:
        densitySliceSavefile = densityPlotsDir+'/densities'
        r, rho = tree.interpolateDensity(xi,yi,zi,xf,yf,zf, numpts, plot=False, save=False)
        
        densities = np.concatenate( (np.reshape(r, (numpts,1)), np.reshape(rho, (numpts,1))), axis=1)
        np.save(densitySliceSavefile,densities)

    
    ## Barrier...
    comm.barrier()
    rprint(rank,'\nEntering greenIterations_KohnSham_SCF()')
    print('Number of targets on proc %i:   %i' %(rank,nPoints) )

    
    densityResidual = 10                                   # initialize the densityResidual to something that fails the convergence tolerance

    [Eband, Ekinetic, Eexchange, Ecorrelation, Ehartree, Etotal] = np.genfromtxt(inputFile)[3:9]
    rprint(rank,[Eband, Ekinetic, Eexchange, Ecorrelation, Ehartree, Etotal])

 

    energyResidual=10
    residuals = 10*np.ones_like(Energies['orbitalEnergies'])
    
    referenceEnergies = {'Etotal':Etotal,'Eband':Eband,'Ehartree':Ehartree,'Eexchange':Eexchange,'Ecorrelation':Ecorrelation}
    referenceEnergies["Eelectrostatic"] = -4.0103432928895426 # DFT-FE value for single Beryllium atom
    scf_args={'inputDensities':inputDensities,'outputDensities':outputDensities,'SCFcount':SCFcount,'nPoints':nPoints,'nOrbitals':nOrbitals,'mixingHistoryCutoff':mixingHistoryCutoff,
               'GPUpresent':GPUpresent,'treecode':treecode,'treecodeOrder':treecodeOrder,'theta':theta,'maxParNode':maxParNode,'batchSize':batchSize,'gaussianAlpha':gaussianAlpha,
               'Energies':Energies,'Times':Times,'exchangeFunctional':exchangeFunctional,'correlationFunctional':correlationFunctional,
               'Vext_local':Vext_local,'Vext_local_fine':Vext_local_fine,'gaugeShift':gaugeShift,'orbitals':orbitals,'oldOrbitals':oldOrbitals,'subtractSingularity':subtractSingularity,
               'X':X,'Y':Y,'Z':Z,'W':W,'Xf':Xf,'Yf':Yf,'Zf':Zf,'Wf':Wf,'gradientFree':gradientFree,'residuals':residuals,'greenIterationOutFile':greenIterationOutFile,
               'referenceEigenvalues':referenceEigenvalues,'symmetricIteration':symmetricIteration,
               'SCFtolerance':SCFtolerance,'initialGItolerance':initialGItolerance, 'finalGItolerance':finalGItolerance, 'gradualSteps':gradualSteps, 'nElectrons':nElectrons,'referenceEnergies':referenceEnergies,'SCFiterationOutFile':SCFiterationOutFile,
               'wavefunctionFile':wavefunctionFile,'densityFile':densityFile,'outputDensityFile':outputDensityFile,'inputDensityFile':inputDensityFile,'vHartreeFile':vHartreeFile,
               'auxiliaryFile':auxiliaryFile,
               'GItolerancesIdx':0,
               'singularityHandling':singularityHandling, 'approximationName':approximationName, 
               'atoms':atoms,'nearbyAtoms':nearbyAtoms,'coreRepresentation':coreRepresentation,
               'order':order,'fine_order':fine_order,
               'regularize':regularize,'epsilon':epsilon,
               'pointsPerCell_coarse':pointsPerCell_coarse, 
               'pointsPerCell_fine':pointsPerCell_fine,
               'TwoMeshStart':TwoMeshStart,
               'occupations':initialOccupations}
    

    """
    clenshawCurtisNorm = clenshawCurtisNormClosure(W)
    method='anderson'
    jacobianOptions={'alpha':1.0, 'M':mixingHistoryCutoff, 'w0':0.01} 
    solverOptions={'fatol':interScfTolerance, 'tol_norm':clenshawCurtisNorm, 'jac_options':jacobianOptions,'maxiter':1000, 'line_search':None, 'disp':True}
 
     
    print('Calling scipyRoot with %s method' %method)
    scfFixedPoint, scf_args = scfFixedPointClosure(scf_args)
#     print(np.shaoe(RHO))
    sol = scipyRoot(scfFixedPoint,RHO, args=scf_args, method=method, options=solverOptions)
    print(sol.success)
    print(sol.message)
    RHO = sol.x
     
     
    """
    comm.barrier()
    rprint(rank,"Copying data to GPU and starting while loop for density...")
    MOVEDATA.callCopyVectorToDevice(orbitals)
    MOVEDATA.callCopyVectorToDevice(W)
    comm.barrier()
    while ( (densityResidual > SCFtolerance) or (energyResidual > SCFtolerance) ):  # terminate SCF when both energy and density are converged.
          
        ## CALL SCF FIXED POINT FUNCTION
#         if SCFcount > 0:
#             print('Exiting before first SCF (for testing initialized mesh accuracy)')
#             return
        abortAfterInitialHartree=False
        
        MOVEDATA.callRemoveVectorFromDevice(orbitals)
        MOVEDATA.callCopyVectorToDevice(orbitals)
        
        
        if GI_form=="Sequential":
            scfFixedPoint, scf_args = scfFixedPointClosure(scf_args)
            densityResidualVector = scfFixedPoint(RHO,scf_args,abortAfterInitialHartree)
        elif GI_form=="Simultaneous":
            scfFixedPointSimultaneous, scf_args = scfFixedPointClosureSimultaneous(scf_args)
            densityResidualVector = scfFixedPointSimultaneous(RHO,scf_args,abortAfterInitialHartree)
        elif GI_form=="Greedy":
            scfFixedPointGreedy, scf_args = scfFixedPointClosureGreedy(scf_args)
            densityResidualVector = scfFixedPointGreedy(RHO,scf_args,abortAfterInitialHartree)
        else:
            print("What should GI_form be?")
            exit(-1)
            
        densityResidual=scf_args['densityResidual']
        energyResidual=scf_args['energyResidual'] 
        SCFcount=scf_args['SCFcount']
          
#         densityResidual = np.sqrt( np.sum( (outputDensities[:,SCFcount-1] - inputDensities[:,SCFcount-1])**2*weights ) )
#         print('Density Residual from arrays ', densityResidual)
        rprint(rank,'Shape of density histories: ', np.shape(scf_args['inputDensities']), np.shape(scf_args['outputDensities']))
        rprint(rank,'outputDensities[0,:] = ', scf_args['outputDensities'][0,:])
        # Now compute new mixing with anderson scheme, then import onto tree. 
    
      
        SCFindex = (SCFcount-1)%scf_args['mixingHistoryCutoff']
        if mixingScheme == 'Simple':
            print('Using simple mixing, from the input/output arrays')
            simpleMixingDensity = mixingParameter*scf_args['outputDensities'][:,SCFindex] + (1-mixingParameter)*scf_args['inputDensities'][:,SCFindex]
#             integratedDensity = np.sum( simpleMixingDensity*W )
            integratedDensity = global_dot( simpleMixingDensity, W, comm )
            rprint(rank,'Integrated simple mixing density: ', integratedDensity)  
    #             tree.importDensityOnLeaves(simpleMixingDensity)
            RHO = np.copy(simpleMixingDensity)
          
        elif mixingScheme == 'Anderson':
            if verbosity>0: rprint(rank, 'Using anderson mixing.')
            andersonDensity = densityMixing.computeNewDensity(scf_args['inputDensities'], scf_args['outputDensities'], mixingParameter,W)
#             integratedDensity = np.sum( andersonDensity*W )
            integratedDensity = global_dot( andersonDensity, W, comm )
            if verbosity>0: rprint(rank,'Integrated anderson density: ', integratedDensity)
    #             tree.importDensityOnLeaves(andersonDensity)
            RHO = np.copy(andersonDensity)
          
        elif mixingScheme == 'None':
            rprint(rank,'Using no mixing.')
#             RHO += densityResidualVector
            RHO = np.copy( scf_args['outputDensities'][:,SCFindex] )
               
          
          
        else:
            print('Mixing must be set to either Simple, Anderson, or None')
            return
      
                  
       
          
        if Energies['Etotal'] > 0.0:                       # Check that the current guess for energy didn't go positive.  Reset it if it did. 
            print('Warning, Energy is positive')
            Energies['Etotal'] = -0.5
              
          
        if SCFcount >= 100:
            print('Setting density residual to -1 to exit after the 150th SCF')
            densityResidual = -1
        
    
    ## DO ONE FINAL ITERATION WITH TWO LEVEL MESH
    print("\n\n\n\nDoing one final SCF iteration with Two Level Mesh.\n\n\n")
    MOVEDATA.callRemoveVectorFromDevice(orbitals)
    MOVEDATA.callCopyVectorToDevice(orbitals)
    scf_args["TwoMeshStart"]=SCFcount
    
    if GI_form=="Sequential":
        scfFixedPoint, scf_args = scfFixedPointClosure(scf_args)
        densityResidualVector = scfFixedPoint(RHO,scf_args,abortAfterInitialHartree)
              
#         if SCFcount >= 1:
#             print('Setting density residual to -1 to exit after the First SCF just to test treecode or restart')
#             energyResidual = -1
#             densityResidual = -1
          
  
   
  
      
      
    rprint(rank,'\nConvergence to a tolerance of %f took %i iterations' %(SCFtolerance, SCFcount))
#     """
    return Energies, RHO, Times
    
    

    

    
if __name__ == "__main__": 
    #import sys;sys.argv = ['', 'Test.testName']
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size() 

    rprint(rank,'='*70) 
    rprint(rank,'='*70) 
    rprint(rank,'='*70,'\n') 
    
    
    
    rprint(rank,'inputFile = ', inputFile)

    rprint(rank,'gradientFree = ', noGradients)
    rprint(rank,'Mixing scheme = ', mixingScheme)
    rprint(rank,'vtk directory = ', vtkDir)
    
    
    # Set domain to be an integrer number of far-field cells
    rprint(rank,"original domain size: ", domainSize)
    remainder=(2*domainSize)%maxSideLength
    print("remainder: ", remainder)
    if remainder>0:
        domainSize+=1/2*(maxSideLength-remainder)
    rprint(rank,"Max side length: ",maxSideLength)
    rprint(rank,"Domain length after adjustment: ", domainSize)
    rprint(rank," Far field nx, ny, nz = ", 2*domainSize/maxSideLength)
        
    

    X,Y,Z,W,Xf,Yf,Zf,Wf,pointsPerCell_coarse, pointsPerCell_fine, atoms,PSPs,nPoints,nOrbitals,nElectrons,referenceEigenvalues = buildMeshFromMinimumDepthCells(domainSize,domainSize,domainSize,maxSideLength,coreRepresentation,
                                                                                                     inputFile,outputFile,srcdir,order,fine_order,gaugeShift,
                                                                                                     divideCriterion,divideParameter1,divideParameter2,divideParameter3,divideParameter4)
    
    
    ## NO LONGER PERFORMING LOAD BALANCING AFTER CREATING MESH POINTS.  IT IS TAKEN CARE OF WHEN CHOPPING THE DOMAIN INTO INITIAL BASE MESH.
#     comm.barrier()
#     xSum = np.sqrt( global_dot(X,X,comm) ) 
#     ySum = np.sqrt( global_dot(Y,Y,comm) ) 
#     zSum = np.sqrt( global_dot(Z,Z,comm) ) 
#     wSum = np.sqrt( global_dot(W,W,comm) ) 
#     
# #     print("NOT CALLING LOAD BALANCER.")
#     print('Before load balancing, nPoints and nCells on proc %i: %i, %i' %(rank,len(X),len(X)/(order+1)**3) )
#     print("BEFORE BALANCING: rank %i, xmin, xmax, ymin, ymax, zmin, zmax: " %(rank), np.min(X),np.max(X),np.min(Y),np.max(Y),np.min(Z),np.max(Z) )
# 
# #     X,Y,Z,W = scatterArrays(X,Y,Z,W,comm)
# #     print('After scattering, nPoints on proc %i: %i' %(rank,len(X)) )
#     comm.barrier()
#     start=MPI.Wtime()
#     X,Y,Z,W = loadBalance(X,Y,Z,W)
#     comm.barrier()
#     end=MPI.Wtime()
#     print("LOAD BALANCING TIME WHEN NOT USING RANDOM FIRST: ", end-start)
#     print('After load balancing, nPoints on proc %i: %i' %(rank,len(X)) )
#     print("proc %i: average x, y, z: %f,%f,%f"%(rank, np.mean(X), np.mean(Y), np.mean(Z)))
# #     atoms = comm.bcast(atoms, root=0)
# #     nOrbitals = comm.bcast(nOrbitals, root=0)
# #     nElectrons = comm.bcast(nElectrons, root=0)
# #     eigenvalues = comm.bcast(eigenvalues, root=0)
# #     referenceEigenvalues = comm.bcast(referenceEigenvalues, root=0)

#     xSum2 = np.sqrt( global_dot(X,X,comm) ) 
#     ySum2 = np.sqrt( global_dot(Y,Y,comm) ) 
#     zSum2 = np.sqrt( global_dot(Z,Z,comm) ) 
#     wSum2 = np.sqrt( global_dot(W,W,comm) )
#     assert abs(xSum-xSum2)/xSum<1e-12, "xSum not matching after DD. xSum=%f, xSum2=%f"%(xSum,xSum2)
#     assert abs(ySum-ySum2)/ySum<1e-12, "ySum not matching after DD. ySum=%f, ySum2=%f"%(ySum,ySum2)
#     assert abs(zSum-zSum2)/zSum<1e-12, "zSum not matching after DD. zSum=%f, zSum2=%f"%(zSum,zSum2)
#     assert abs(wSum-wSum2)/wSum<1e-12, "wSum not matching after DD. wSum=%f, wSum2=%f"%(wSum,wSum2)
#     
#     comm.barrier()
#     
#     ## Test if load balancing worked...
#     print("AFTER BALANCING: rank %i, xmin, xmax, ymin, ymax, zmin, zmax: " %(rank), np.min(X),np.max(X),np.min(Y),np.max(Y),np.min(Z),np.max(Z) )
# #     exit(-1)
    
#     eigenvalues = -2*np.ones(nOrbitals)
    RHO = initializeDensityFromAtomicDataExternally(X,Y,Z,W,atoms,coreRepresentation)
    densityIntegral = global_dot( RHO, W, comm)
    rprint(rank,"Initial density integrates to ", densityIntegral)
    nPointsLocal = len(X)
#     assert abs(2-global_dot(RHO,W,comm)) < 1e-12, "Initial density not integrating to 2"
    orbitals = np.zeros((nOrbitals,nPointsLocal))
    if coreRepresentation=="AllElectron":
        orbitals,initialOccupations,initialEnergies = initializeOrbitalsFromAtomicDataExternally(atoms,coreRepresentation,orbitals,nOrbitals,X,Y,Z,W)
    elif coreRepresentation=="Pseudopotential":
        orbitals,initialOccupations,initialEnergies = initializeOrbitalsFromAtomicDataExternally(atoms,coreRepresentation,orbitals,nOrbitals,X,Y,Z,W)
#         orbitals = initializeOrbitalsRandomly(atoms,coreRepresentation,orbitals,nOrbitals,X,Y,Z)
#         initialOccupations=2*np.ones(nOrbitals)
#         initialEnergies=-1*np.ones(nOrbitals)
    print("Max of first wavefunction: ", np.max(np.abs(orbitals[0,:])))
#     print('nOrbitals: ', nOrbitals)
    comm.barrier()
    
    # Sort orbitals
#     orbitals, initialEnergies = sortByEigenvalue(orbitals, initialEnergies)
#     fermiObjectiveFunction = fermiObjectiveFunctionClosure(Energies,nElectrons)        
#     eF = brentq(fermiObjectiveFunction, Energies['orbitalEnergies'][0], 1, xtol=1e-14)
#     rprint(rank,'Fermi energy: %f'%eF)
#     exponentialArg = (Energies['orbitalEnergies']-eF)/Sigma
#     initialOccupations = 2*1/(1+np.exp( exponentialArg ) )
    
    eigenvalues=np.array(initialEnergies)
    print("Energies:    ",initialEnergies)
    print("Occupations: ",initialOccupations)
#     input()

    

    
    initialRho = np.copy(RHO)
    finalRho = testGreenIterationsGPU_rootfinding(X,Y,Z,W,Xf,Yf,Zf,Wf,pointsPerCell_coarse, pointsPerCell_fine,RHO,orbitals,eigenvalues,initialOccupations,atoms,coreRepresentation,nPointsLocal,nOrbitals,nElectrons,referenceEigenvalues)

    MOVEDATA.callRemoveVectorFromDevice(orbitals)
    MOVEDATA.callRemoveVectorFromDevice(W)
