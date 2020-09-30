'''
TAGI-main.py
This is the main run file for TAGI.  

Created on Mar 13, 2018
@author: nathanvaughn
'''

import time
import resource
import numpy as np
import pylibxc
import csv
import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


### PATH SETUP ###
import os
import sys
from pathlib import Path
homePath = str(Path.home())
sys.path.insert(1, '/Users/nathanvaughn/Documents/GitHub/TAGI/3D-GreenIterations/src/utilities')
sys.path.insert(1, '/Users/nathanvaughn/Documents/GitHub/TAGI/3D-GreenIterations/src/dataStructures')
sys.path.insert(1, '/home/njvaughn/TAGI/3D-GreenIterations/src/utilities')

from mpiUtilities import global_dot, scatterArrays, rprint
from mpiMeshBuilding import  buildMeshFromMinimumDepthCells

from initializationRoutines import initializeDensityFromAtomicDataExternally, initializeOrbitalsFromAtomicDataExternally


from CellStruct_CC import Cell
from GridpointStruct import GridPoint
import densityMixingSchemes as densityMixing
from scfFixedPoint import scfFixedPointClosure
# from scfFixedPointSimultaneous import scfFixedPointClosureSimultaneous

import moveData_wrapper as MOVEDATA
from twoMeshCorrection import twoMeshCorrectionClosure





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
treecodeDegree       = int(sys.argv[n]); n+=1
theta               = float(sys.argv[n]); n+=1
maxPerSourceLeaf          = int(sys.argv[n]); n+=1
maxPerTargetLeaf           = int(sys.argv[n]); n+=1
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
    rprint(0, "What should regularize input be set to?")
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
# rprint(0, "Converting command line arguments to correct types.")

# def global_dot(u,v,comm):
#     local_dot = np.dot(u,v)
#     global_dot = comm.allreduce(local_dot)
#     return global_dot


def clenshawCurtisNormClosure(W):
    def clenshawCurtisNorm(psi):
        norm = np.sqrt( global_dot( psi, psi*W, comm ) )
        return norm
    return clenshawCurtisNorm







def testGreenIterationsGPU_rootfinding(X,Y,Z,W,Xf,Yf,Zf,Wf,pointsPerCell_coarse, pointsPerCell_fine,RHO, CORECHARGERHO,orbitals,eigenvalues,initialOccupations,atoms,coreRepresentation,nPoints,nOrbitals,nElectrons,referenceEigenvalues,vtkExport=False,onTheFlyRefinement=False, maxOrbitals=None, maxSCFIterations=None, restartFile=None):
    
    startTime = time.time()
    

    
    Energies, Rho, Times = greenIterations_KohnSham_SCF_rootfinding(X,Y,Z,W,Xf,Yf,Zf,Wf,pointsPerCell_coarse, pointsPerCell_fine,RHO, CORECHARGERHO,orbitals,eigenvalues,initialOccupations,atoms,coreRepresentation,nPoints,nOrbitals,nElectrons,referenceEigenvalues,
                                scfTolerance, initialGItolerance, finalGItolerance, gradualSteps,
                                gradientFree, symmetricIteration, GPUpresent, treecode, treecodeDegree, theta, maxPerSourceLeaf, maxPerTargetLeaf, 
                                singularityHandling,approximationName,
                                 mixingScheme, mixingParameter, mixingHistoryCutoff,
                                 subtractSingularity, gaussianAlpha, gaugeShift,
                                 inputFile=inputFile,outputFile=outputFile, restartFile=restart,
                                 onTheFlyRefinement=onTheFlyRefinement, vtkExport=False, maxOrbitals=maxOrbitals, maxSCFIterations=maxSCFIterations,
                                 regularize=regularize,epsilon=epsilon)
    
    global_nPoints = comm.allreduce(nPoints)

    if rank==0:
        Times['totalKohnShamTime'] = time.time()-startTime
        rprint(rank, 'Total Time: ', Times['totalKohnShamTime'])
        
        
    
        header = ['numProcs','domainSize','maxSideLength','order','fineOrder','numberOfCells','numberOfPoints','gradientFree',
                  'divideCriterion','divideParameter1','divideParameter2','divideParameter3','divideParameter4',
                  'gaussianAlpha','gaugeShift','finalGItolerance',
                  'GreenSingSubtracted', 'regularize', 'epsilon',
                  'orbitalEnergies', 'BandEnergy', 'KineticEnergy',
                  'ExchangeEnergy','CorrelationEnergy','ElectrostaticEnergy','TotalEnergy',
                  'Treecode','approximationName','treecodeDegree','theta','maxPerSourceLeaf','maxPerTargetLeaf','totalTime','timePerConvolution','totalIterationCount']
        
        myData = [size,domainSize,maxSideLength,order,fine_order, nPoints/(order+1)**3,global_nPoints,gradientFree,
                  divideCriterion,divideParameter1,divideParameter2,divideParameter3,divideParameter4,
                  gaussianAlpha,gaugeShift,finalGItolerance,
                  subtractSingularity, regularize, epsilon,
                  Energies['orbitalEnergies']-Energies['gaugeShift'], Energies['Eband'], Energies['kinetic'], Energies['Ex'], Energies['Ec'], Energies['totalElectrostatic'], Energies['Etotal'],
                  treecode,approximationName,treecodeDegree,theta,maxPerSourceLeaf,maxPerTargetLeaf, Times['totalKohnShamTime'],Times['timePerConvolution'],Times['totalIterationCount'] ]

    
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


    



def greenIterations_KohnSham_SCF_rootfinding(X,Y,Z,W,Xf,Yf,Zf,Wf,pointsPerCell_coarse, pointsPerCell_fine,RHO, CORECHARGERHO,orbitals,eigenvalues,initialOccupations,atoms,coreRepresentation,nPoints,nOrbitals,nElectrons,referenceEigenvalues,
                                             SCFtolerance, initialGItolerance, finalGItolerance, gradualSteps, 
                                             gradientFree, symmetricIteration, GPUpresent, 
                                 treecode, treecodeDegree, theta, maxPerSourceLeaf, maxPerTargetLeaf, singularityHandling, approximationName,
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
# #             rprint(0, "Trying to append")
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
#             rprint(0, "dist = ", dist)
#             if ( (dist<4) and (added==False) ):
#                 
# #                 rprint(0, "Trying to append")
#                 nearbyAtoms = np.append(nearbyAtoms,atom)
#                 added=True
#             
#             
#     rprint(0, np.shape(nearbyAtoms))
             
            
    
    verbosity=0
    polarization="unpolarized"
    exchangeFunctional="LDA_X"
    correlationFunctional="LDA_C_PW"
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
            
        else:
            rprint(0, "Error: what should coreRepresentation be?")
            exit(-1)
        atomCount+=1
        

    Energies={}
    Energies['orbitalEnergies'] = eigenvalues
    Energies['orbitalEnergies_corrected'] = np.copy(eigenvalues)
    Energies['gaugeShift'] = gaugeShift
    Energies['kinetic'] = 0.0
    Energies['Enuclear'] = 0.0
    Energies['Eold_corrected']=0.0
    
    for atom1 in atoms:
        for atom2 in atoms:
            if atom1!=atom2:
                r = np.sqrt( (atom1.x-atom2.x)**2 + (atom1.y-atom2.y)**2 + (atom1.z-atom2.z)**2 )
                Energies['Enuclear'] += atom1.nuclearCharge*atom2.nuclearCharge/r
    Energies['Enuclear'] /= 2 # because of double counting
    
    Times={}
    
    
    
    
    if verbosity>0: rprint(rank,'MEMORY USAGE: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    # Store Tree variables locally
    gaugeShift = Energies['gaugeShift']
    
    globalNumPoints = comm.allreduce(nPoints)
    
    nPointsF=len(Xf)
    globalFineMeshPoints = comm.allreduce(nPointsF)

    greenIterationOutFile = outputFile[:-4]+'_GREEN_'+str(globalNumPoints)+'_'+str(globalFineMeshPoints)+outputFile[-4:]
    SCFiterationOutFile =   outputFile[:-4]+'_SCF_'+str(globalNumPoints)+'_'+str(globalFineMeshPoints)+outputFile[-4:]
    densityPlotsDir =       outputFile[:-4]+'_SCF_'+str(globalNumPoints)+'_plots'

    comm.barrier()
    exampleDir = homePath+np.genfromtxt(inputFile,dtype=str)[2]
    if rank==0:
        try:
            os.mkdir(exampleDir)
        except OSError as e:
            rprint(rank,'Unable to make restart directory %s for this test case due to %s'  %(exampleDir,e))

    restartFilesDir =       exampleDir + "numPoints_" + str(globalNumPoints)
         
# #     restartFilesDir =       exampleDir + "backup_numPoints_" + str(globalNumPoints) + "/numPoints_1778176"
#     restartFilesDir = "/home/njvaughn/restartFiles/C60/backup_numPoints_1778176/numPoints_1778176"
#     rprint(0,"USING RESTART DATA FROM backup_numPoints_")
#     rprint(0,"USING RESTART DATA FROM backup_numPoints_")
#     rprint(0,"USING RESTART DATA FROM backup_numPoints_")
#     rprint(0,"USING RESTART DATA FROM backup_numPoints_")
#     rprint(0,"USING RESTART DATA FROM backup_numPoints_")
#     rprint(0,"USING RESTART DATA FROM backup_numPoints_")
#     rprint(0,"USING RESTART DATA FROM backup_numPoints_")
#     rprint(0,"USING RESTART DATA FROM backup_numPoints_")
#     rprint(0,"USING RESTART DATA FROM backup_numPoints_")
#     rprint(0,"USING RESTART DATA FROM backup_numPoints_") 
    if rank==0:
        try:
            os.mkdir(restartFilesDir)
        except OSError as e:
            rprint(rank,'Unable to make restart directory %s for this specific run due to %s' %(restartFilesDir,e) )
    
    rprint(rank,"restartFilesDir = ", restartFilesDir)

    wavefunctionFile =      restartFilesDir+'/wavefunctions_rank_%i_of_%i' %(rank,size)
    densityFile =           restartFilesDir+'/density_rank_%i_of_%i' %(rank,size)
    inputDensityFile =      restartFilesDir+'/inputdensity_rank_%i_of_%i' %(rank,size)
    outputDensityFile =     restartFilesDir+'/outputdensity_rank_%i_of_%i' %(rank,size)
    vHartreeFile =          restartFilesDir+'/vHartree_rank_%i_of_%i' %(rank,size)
    auxiliaryFile =         restartFilesDir+'/auxiliary_rank_%i_of_%i' %(rank,size)
    
    
    
    plotSliceOfDensity=False
    if plotSliceOfDensity==True:
        try:
            os.mkdir(densityPlotsDir)
        except OSError as e:
            rprint(rank,'Unable to make densityPlotsDir directory %s due to %s' %(densityPlotsDir,e) )
        

    
    
#     tr = tracker.SummaryTracker()   
    if restartFile!=False:
#         rprint(0, "Not ready to handle restarts in mpi version.")
#         return
        try:
            orbitals = np.load(wavefunctionFile+'.npy')
        except FileNotFoundError:
            rprint(0, "Rank %i could not find restart file " %rank, wavefunctionFile + ".npy.  Exiting.")
            exit(-1)
        oldOrbitals = np.copy(orbitals)
        RHO = np.load(densityFile+'.npy')
         
        inputDensities = np.load(inputDensityFile+'.npy')
        outputDensities = np.load(outputDensityFile+'.npy')
        
        if mixingScheme == 'Anderson':
            if verbosity>-1: rprint(rank, 'Using anderson mixing during restart data load.')
            andersonDensity = densityMixing.computeNewDensity(inputDensities, outputDensities, mixingParameter,W)
            integratedDensity = global_dot( andersonDensity, W, comm )
            if verbosity>0: rprint(rank,'Integrated anderson density: ', integratedDensity)
            RHO = np.copy(andersonDensity)
         
        V_hartreeNew = np.load(vHartreeFile+'.npy')
         
         
        auxiliaryRestartData = np.load(auxiliaryFile+'.npy',allow_pickle = True).item()
        rprint(rank,'type of aux: ', type(auxiliaryRestartData))
        SCFcount = auxiliaryRestartData['SCFcount']
        Times['totalIterationCount'] = auxiliaryRestartData['totalIterationCount']
        Energies['orbitalEnergies'] = auxiliaryRestartData['eigenvalues'] 
        Energies['Eold'] = auxiliaryRestartData['Eold']
        
        initialGItolerancesIdx = auxiliaryRestartData['GItolerancesIdx'] 
         
         
        NLCC_RHO = RHO+CORECHARGERHO
        Energies['Ehartree'] = 1/2*np.sum(W * RHO * V_hartreeNew)
        
        exchangeOutput    =    exchangeFunctional.compute(NLCC_RHO)
        correlationOutput = correlationFunctional.compute(NLCC_RHO) 
        
        Energies['Ex'] = np.sum( W * NLCC_RHO * np.reshape(   exchangeOutput['zk'],np.shape(NLCC_RHO)) )
        Energies['Ec'] = np.sum( W * NLCC_RHO * np.reshape(correlationOutput['zk'],np.shape(NLCC_RHO)) )
         
        Vx = np.reshape(   exchangeOutput['vrho'],np.shape(NLCC_RHO))
        Vc = np.reshape(correlationOutput['vrho'],np.shape(NLCC_RHO))
         
        Energies['Vx'] = np.sum(W * RHO * Vx)
        Energies['Vc'] = np.sum(W * RHO * Vc)        
        
    
    else: 
        Eold = -10
        SCFcount=0
        initialGItolerancesIdx=0
        Times['totalIterationCount'] = 0

        inputDensities = np.zeros((nPoints,1))
        outputDensities = np.zeros((nPoints,1))
        
        inputDensities[:,0] = np.copy(RHO)
        oldOrbitals = np.copy(orbitals)

    
#     if plotSliceOfDensity==True:
#         densitySliceSavefile = densityPlotsDir+'/densities'
#         r, rho = tree.interpolateDensity(xi,yi,zi,xf,yf,zf, numpts, plot=False, save=False)
#         
#         densities = np.concatenate( (np.reshape(r, (numpts,1)), np.reshape(rho, (numpts,1))), axis=1)
#         np.save(densitySliceSavefile,densities)

    
    ## Barrier...
    comm.barrier()
    rprint(rank,'Entering greenIterations_KohnSham_SCF()')
    rprint(0, 'Number of targets on proc %i:   %i' %(rank,nPoints) )
    comm.barrier()
    
    densityResidual = 10                                   # initialize the densityResidual to something that fails the convergence tolerance

    [Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal] = np.genfromtxt(inputFile)[3:9]
    rprint(rank,[Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal])

 

    energyResidual=10
    residuals = 5*np.ones_like(Energies['orbitalEnergies'])
    
    referenceEnergies = {'Etotal':Etotal,'Eband':Eband,'Eelectrostatic':Eelectrostatic,'Eexchange':Eexchange,'Ecorrelation':Ecorrelation}
    referenceEnergies["Ehartree"] = 0.0 # DFT-FE value for single Beryllium atom
    
    
    scf_args={'inputDensities':inputDensities,'outputDensities':outputDensities,'SCFcount':SCFcount,'nPoints':nPoints,'nOrbitals':nOrbitals,'mixingHistoryCutoff':mixingHistoryCutoff,
               'GPUpresent':GPUpresent,'treecode':treecode,'treecodeDegree':treecodeDegree,'theta':theta,'maxPerSourceLeaf':maxPerSourceLeaf,'maxPerTargetLeaf':maxPerTargetLeaf,'gaussianAlpha':gaussianAlpha,
               'Energies':Energies,'Times':Times,'exchangeFunctional':exchangeFunctional,'correlationFunctional':correlationFunctional,
               'Vext_local':Vext_local,'Vext_local_fine':Vext_local_fine,'gaugeShift':gaugeShift,'orbitals':orbitals,'oldOrbitals':oldOrbitals,'subtractSingularity':subtractSingularity,
               'X':X,'Y':Y,'Z':Z,'W':W,'Xf':Xf,'Yf':Yf,'Zf':Zf,'Wf':Wf,'gradientFree':gradientFree,'residuals':residuals,'greenIterationOutFile':greenIterationOutFile,
               'referenceEigenvalues':referenceEigenvalues,'symmetricIteration':symmetricIteration,
               'SCFtolerance':SCFtolerance,'initialGItolerance':initialGItolerance, 'finalGItolerance':finalGItolerance, 'gradualSteps':gradualSteps, 'nElectrons':nElectrons,'referenceEnergies':referenceEnergies,'SCFiterationOutFile':SCFiterationOutFile,
               'wavefunctionFile':wavefunctionFile,'densityFile':densityFile,'outputDensityFile':outputDensityFile,'inputDensityFile':inputDensityFile,'vHartreeFile':vHartreeFile,
               'auxiliaryFile':auxiliaryFile,
               'GItolerancesIdx':initialGItolerancesIdx,
               'singularityHandling':singularityHandling, 'approximationName':approximationName, 
               'atoms':atoms,'nearbyAtoms':nearbyAtoms,'coreRepresentation':coreRepresentation,
               'order':order,'fine_order':fine_order,
               'regularize':regularize,'epsilon':epsilon,
               'pointsPerCell_coarse':pointsPerCell_coarse, 
               'pointsPerCell_fine':pointsPerCell_fine,
               'TwoMeshStart':TwoMeshStart,
               'occupations':initialOccupations,
               'CORECHARGERHO': CORECHARGERHO}
    


    comm.barrier()
    rprint(rank,"Copying data to GPUs.")
    if GPUpresent: MOVEDATA.callCopyVectorToDevice(orbitals)
    if GPUpresent: MOVEDATA.callCopyVectorToDevice(W)
    comm.barrier()
    rprint(rank,"Copied data to GPUs.  Starting while loop for density...")
    while ( (densityResidual > SCFtolerance) or (energyResidual > SCFtolerance) ):  # terminate SCF when both energy and density are converged.
          
        ## CALL SCF FIXED POINT FUNCTION
#         if SCFcount > 0:
#             rprint(0, 'Exiting before first SCF (for testing initialized mesh accuracy)')
#             return
        abortAfterInitialHartree=False 
        
        if GPUpresent: MOVEDATA.callRemoveVectorFromDevice(orbitals)
        if GPUpresent: MOVEDATA.callCopyVectorToDevice(orbitals)
        
        
        if GI_form=="Sequential":
            scfFixedPoint, scf_args = scfFixedPointClosure(scf_args)
            densityResidualVector = scfFixedPoint(RHO,scf_args,abortAfterInitialHartree)
#             rprint(rank,"******** REPEATING GREEN ITERATION FOR SAME INPUT DENSITY AFTER SORTING *******")
#             scf_args['GItolerancesIdx']-=1
#             densityResidualVector = scfFixedPoint(RHO,scf_args,abortAfterInitialHartree)  # call this a second time, will have wavefunctions sorted from the first call.
        elif GI_form=="Simultaneous":
            scfFixedPointSimultaneous, scf_args = scfFixedPointClosureSimultaneous(scf_args)
            densityResidualVector = scfFixedPointSimultaneous(RHO,scf_args,abortAfterInitialHartree)
        elif GI_form=="Greedy":
            scfFixedPointGreedy, scf_args = scfFixedPointClosureGreedy(scf_args)
            densityResidualVector = scfFixedPointGreedy(RHO,scf_args,abortAfterInitialHartree)
        else:
            rprint(0, "What should GI_form be?")
            exit(-1)
            
        densityResidual=scf_args['densityResidual']
        energyResidual=scf_args['energyResidual'] 
        SCFcount=scf_args['SCFcount']
          

        if verbosity>0: rprint(rank,'Shape of density histories: ', np.shape(scf_args['inputDensities']), np.shape(scf_args['outputDensities']))
        if verbosity>0: rprint(rank,'outputDensities[0,:] = ', scf_args['outputDensities'][0,:])
        # Now compute new mixing with anderson scheme, then import onto tree. 
    
      
        SCFindex = (SCFcount-1)%scf_args['mixingHistoryCutoff']
        if mixingScheme == 'Simple':
            rprint(0, 'Using simple mixing, from the input/output arrays')
            simpleMixingDensity = mixingParameter*scf_args['outputDensities'][:,SCFindex] + (1-mixingParameter)*scf_args['inputDensities'][:,SCFindex]
            integratedDensity = global_dot( simpleMixingDensity, W, comm )
            rprint(rank,'Integrated simple mixing density: ', integratedDensity)  
            RHO = np.copy(simpleMixingDensity)
          
        elif mixingScheme == 'Anderson':
            if verbosity>0: rprint(rank, 'Using anderson mixing.')
            andersonDensity = densityMixing.computeNewDensity(scf_args['inputDensities'], scf_args['outputDensities'], mixingParameter,W)
#             integratedDensity = np.sum( andersonDensity*W )
            integratedDensity = global_dot( andersonDensity, W, comm )
            if verbosity>0: rprint(rank,'Integrated anderson density: ', integratedDensity)
            RHO = np.copy(andersonDensity)
          
        elif mixingScheme == 'None':
            rprint(rank,'Using no mixing.')
#             RHO += densityResidualVector
            RHO = np.copy( scf_args['outputDensities'][:,SCFindex] )
               
          
          
        else:
            rprint(0, 'Mixing must be set to either Simple, Anderson, or None')
            return
      
                  
       
          
        if Energies['Etotal'] > 0.0:                       # Check that the current guess for energy didn't go positive.  Reset it if it did. 
            rprint(rank, 'Warning, Energy is positive')
            Energies['Etotal'] = -0.5
              
          
        if SCFcount >= 100:
            rprint(0, 'Setting density residual to -1 to exit after the 150th SCF')
            densityResidual = -1
            
            
#         # perform two mesh correction after each SCF iteration
# #         rprint(rank,"Performing two mesh correction after %i SCF iteration." %SCFcount)
# #         if GPUpresent: MOVEDATA.callRemoveVectorFromDevice(orbitals)
# #         if GPUpresent: MOVEDATA.callCopyVectorToDevice(orbitals)
#         twoMeshCorrection, scf_args = twoMeshCorrectionClosure(scf_args)
#         twoMeshCorrection(RHO,scf_args)
        
    
    ## If a pseudopotential calculation, call the two-mesh correction scheme.
    if coreRepresentation=="Pseudopotential":
        rprint(rank,"\n\n\n\n")
        rprint(rank,"\n\n\n\nCalling the Two-Level Mesh Correction.\n\n\n")
        if GPUpresent: MOVEDATA.callRemoveVectorFromDevice(orbitals)
        if GPUpresent: MOVEDATA.callCopyVectorToDevice(orbitals)
        scf_args["TwoMeshStart"]=SCFcount
          
    #     scfFixedPoint, scf_args = scfFixedPointClosure(scf_args)
    #     densityResidualVector = scfFixedPoint(RHO,scf_args,abortAfterInitialHartree)
        
        twoMeshCorrection, scf_args = twoMeshCorrectionClosure(scf_args)
        twoMeshCorrection(RHO,scf_args)
        
        Energies['Etotal']=Energies['Etotal_corrected']
        Energies['Etotal']=Energies['Etotal_corrected']
     
     
              
#     if SCFcount >= 1:
#         rprint(0, 'Setting density residual to -1 to exit after the First SCF just to test treecode or restart')
#         energyResidual = -1
#         densityResidual = -1
          
  
   
  
      
      
    rprint(rank,'\nConvergence to a tolerance of %f took %i iterations' %(SCFtolerance, SCFcount))
#     """
    return Energies, RHO, Times
    
    

    

    
if __name__ == "__main__": 
    #import sys;sys.argv = ['', 'Test.testName']
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    verbosity=0 

    rprint(rank,'='*70) 
    rprint(rank,'='*70) 
    rprint(rank,'='*70,'\n') 
    
    
    
    rprint(rank,'inputFile = ', inputFile)

    rprint(rank,'gradientFree = ', noGradients)
    rprint(rank,'Mixing scheme = ', mixingScheme)
    rprint(rank,'vtk directory = ', vtkDir)
    
    
    # Set domain to be an integrer number of far-field cells
    if verbosity>0: rprint(rank,"original domain size: ", domainSize)
    remainder=(2*domainSize)%maxSideLength
    if verbosity>0: rprint(0, "remainder: ", remainder)
    if remainder>0:
        domainSize+=1/2*(maxSideLength-remainder)
    if verbosity>0: 
        rprint(rank,"Max side length: ",maxSideLength)
        rprint(rank,"Domain length after adjustment: ", domainSize)
        rprint(rank," Far field nx, ny, nz = ", 2*domainSize/maxSideLength)
        
    

    X,Y,Z,W,Xf,Yf,Zf,Wf,pointsPerCell_coarse, pointsPerCell_fine, atoms,PSPs,nPoints,nOrbitals,nElectrons,referenceEigenvalues = buildMeshFromMinimumDepthCells(domainSize,domainSize,domainSize,maxSideLength,coreRepresentation,
                                                                                                     inputFile,outputFile,srcdir,order,fine_order,gaugeShift,
                                                                                                     divideCriterion,divideParameter1,divideParameter2,divideParameter3,divideParameter4)
    
    rprint(rank,"Returned from buildMeshFromMinimumDepthCells()")

    
    
    ## NO LONGER PERFORMING LOAD BALANCING AFTER CREATING MESH POINTS.  IT IS TAKEN CARE OF WHEN CHOPPING THE DOMAIN INTO INITIAL BASE MESH.
#     comm.barrier()
#     xSum = np.sqrt( global_dot(X,X,comm) ) 
#     ySum = np.sqrt( global_dot(Y,Y,comm) ) 
#     zSum = np.sqrt( global_dot(Z,Z,comm) ) 
#     wSum = np.sqrt( global_dot(W,W,comm) ) 
#     
# #     rprint(0, "NOT CALLING LOAD BALANCER.")
#     rprint(0, 'Before load balancing, nPoints and nCells on proc %i: %i, %i' %(rank,len(X),len(X)/(order+1)**3) )
#     rprint(0, "BEFORE BALANCING: rank %i, xmin, xmax, ymin, ymax, zmin, zmax: " %(rank), np.min(X),np.max(X),np.min(Y),np.max(Y),np.min(Z),np.max(Z) )
# 
# #     X,Y,Z,W = scatterArrays(X,Y,Z,W,comm)
# #     rprint(0, 'After scattering, nPoints on proc %i: %i' %(rank,len(X)) )
#     comm.barrier()
#     start=MPI.Wtime()
#     X,Y,Z,W = loadBalance(X,Y,Z,W)
#     comm.barrier()
#     end=MPI.Wtime()
#     rprint(0, "LOAD BALANCING TIME WHEN NOT USING RANDOM FIRST: ", end-start)
#     rprint(0, 'After load balancing, nPoints on proc %i: %i' %(rank,len(X)) )
#     rprint(0, "proc %i: average x, y, z: %f,%f,%f"%(rank, np.mean(X), np.mean(Y), np.mean(Z)))
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
#     rprint(0, "AFTER BALANCING: rank %i, xmin, xmax, ymin, ymax, zmin, zmax: " %(rank), np.min(X),np.max(X),np.min(Y),np.max(Y),np.min(Z),np.max(Z) )
# #     exit(-1)
    
#     eigenvalues = -2*np.ones(nOrbitals)
    RHO, CORECHARGERHO = initializeDensityFromAtomicDataExternally(X,Y,Z,W,atoms,coreRepresentation)
    densityIntegral = global_dot( RHO, W, comm)
    rprint(rank,"Initial density integrates to ", densityIntegral)
    CCdensityIntegral = global_dot( CORECHARGERHO, W, comm)
    rprint(rank,"Initial core charge density integrates to ", CCdensityIntegral)
    nPointsLocal = len(X)
#     assert abs(2-global_dot(RHO,W,comm)) < 1e-12, "Initial density not integrating to 2"
    orbitals = np.zeros((nOrbitals,nPointsLocal))
    
    for atom in atoms:
        atom.orbitalInterpolators(coreRepresentation)
            
            
    if coreRepresentation=="AllElectron":
        orbitals,initialOccupations,initialEnergies = initializeOrbitalsFromAtomicDataExternally(atoms,coreRepresentation,orbitals,nOrbitals,X,Y,Z,W)
    elif coreRepresentation=="Pseudopotential":
        orbitals,initialOccupations,initialEnergies = initializeOrbitalsFromAtomicDataExternally(atoms,coreRepresentation,orbitals,nOrbitals,X,Y,Z,W)
#         orbitals = initializeOrbitalsRandomly(atoms,coreRepresentation,orbitals,nOrbitals,X,Y,Z)
#         initialOccupations=2*np.ones(nOrbitals)
#         initialEnergies=-1*np.ones(nOrbitals)
#     rprint(0, 'nOrbitals: ', nOrbitals)
    comm.barrier()
    

    
    eigenvalues=np.array(initialEnergies)/3
    rprint(rank, "Initial eigenvalues:  ",initialEnergies)
    rprint(rank, "Initial occupations:  ",initialOccupations)
#     input()

    

    
    initialRho = np.copy(RHO)
    finalRho = testGreenIterationsGPU_rootfinding(X,Y,Z,W,Xf,Yf,Zf,Wf,pointsPerCell_coarse, pointsPerCell_fine,RHO, CORECHARGERHO,orbitals,eigenvalues,initialOccupations,atoms,coreRepresentation,nPointsLocal,nOrbitals,nElectrons,referenceEigenvalues)

    if GPUpresent: MOVEDATA.callRemoveVectorFromDevice(orbitals)
    if GPUpresent: MOVEDATA.callRemoveVectorFromDevice(W)
    
    