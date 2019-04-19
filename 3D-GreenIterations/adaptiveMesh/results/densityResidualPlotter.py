import numpy as np
import matplotlib.pyplot as plt
from pyevtk.hl import pointsToVTK


resultsFile='densities.npy'
# resultsDir = '/home/njvaughn/synchronizedDataFiles/krasnyMeshTests/Slice_Testing/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Slice_Testing/H2_LW5_200_SCF_91008_plots/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Slice_Testing/Be_LW5_200_SCF_75776_plots/'

resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/benzeneTesting/Be_LW5_200_SCF_75776_plots/'


def compareTwoIterations(restartDir,iterationA,iterationB):
    orbitals = np.load(wavefunctionFile+'.npy')
    oldOrbitals = np.copy(orbitals)
#     for m in range(nOrbitals): 
#         tree.importPhiOnLeaves(orbitals[:,m], m)
    density = np.load(densityFile+'.npy')
#     tree.importDensityOnLeaves(density)
    
    inputDensities = np.load(inputDensityFile+'.npy')
    outputDensities = np.load(outputDensityFile+'.npy')
    
    V_hartreeNew = np.load(vHartreeFile+'.npy')
#     tree.importVhartreeOnLeaves(V_hartreeNew)
#     tree.updateVxcAndVeffAtQuadpoints()
    
    
    # make and save dictionary
    auxiliaryRestartData = np.load(auxiliaryFile+'.npy').item()
    print('type of aux: ', type(auxiliaryRestartData))
    SCFcount = auxiliaryRestartData['SCFcount']
    tree.totalIterationCount = auxiliaryRestartData['totalIterationCount']
    tree.orbitalEnergies = auxiliaryRestartData['eigenvalues'] 
    Eold = auxiliaryRestartData['Eold']
    
    
    
    pointsToVTK(filename, np.array(x), np.array(y), np.array(z), data = 
        {"rho" : np.array(rho), "V" : np.array(v),  "Phi0" : np.array(phi0), "Phi1" : np.array(phi1),
        "Phi2" : np.array(phi2), "Phi3" : np.array(phi3), "Phi4" : np.array(phi4)  } )
    
 

if __name__=="__main__":
    restartDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/restartFiles_1416000_after25/'
    compareTwoIterations(restartDir,10,11)
    