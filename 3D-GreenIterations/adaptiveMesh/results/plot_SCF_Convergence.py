'''
Created on Mar 29, 2018

@author: nathanvaughn
'''

import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import os
import numpy as np


# # ##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/carbonMonoxide/iterationResults/'
# # ##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/H2IterationResults/'
# # # resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/BerylliumIterationResults/'
# # ##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenIterationResults/'
# # #resultsDir = '/Users/nathanvaughn/Desktop/scratch/O_Gaussian/'
# resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/oxygen_with_anderson/'
# # plotsDir = resultsDir+'plots/'
# # # file='LW5_1000_andersonMixing_p5_1em76_SCF_.csv'
# # # file='LW5_1500_andersonMixing_p5_1em8_SCF_.csv'
# file='LW5_2000_andersonMixing_p5_1em76_SCF_.csv'

# ## Carbon Monoxide
# resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/'
# plotsDir = resultsDir+'plots/'
# # file ='LW5_1500o5_GradientFree_eigRes_looseThenTight_titan_SCF_.csv'
# # file ='LW5o5_1500_SCF_.csv'
# # file='LW5o5_1500_largeDomain_SCF_.csv'
# # file='LW5o5_1000_fixedMesh_only7_looseInit_SCF_.csv'
# # file='LW5o5_1500_fixedAtomicPositions_only7_looseInit_SCF_.csv'
# # file='LW5o4_1000_only7_tightFromStart_GIanderson_afterSCF1_SCF_.csv'
# # file='LW5o5_2000_only7_tightFromStart_GIandersonAfterSCF1_SCF_.csv'
# file='LW5o5_2000_6_orbitals_SCF_.csv'
# # file='LW5o5_2000_7_orbitals_noGIanderson_SCF_.csv'

# ## Oxygen -- Biros
# # resultsDir = '/Users/nathanvaughn/Desktop/meshTests/LWvsBiros/Oxygen/'
# resultsDir = '/Users/nathanvaughn/Desktop/meshTests/LWvsBiros/Oxygen/'
# plotsDir = resultsDir+'plots/'
# # # # file='Biros_o7_1em4_solo_SCF_.csv'
# # # # file='Biros_o7_7em5_alpha_1p5_SCF_.csv'
# # # # file='BirosN_o7_1em3_SCF_.csv'
# # # # file='BirosN_o7_2em4_SCF_.csv'
# # file='BirosG_o7_max15_SCF_.csv'
# file='BirosG_o7_1em5_SCF_.csv'

# resultsDir = '/Users/nathanvaughn/Desktop/meshTests/LWvsBiros/OxygenGaviniRef/'
# file='BirosGN2_o5_1em1_SCF_.csv'


## Krasny refine for oxygen
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Oxygen_densityIntegral4th/'
# file='ds_cellOrder5_maxDepth15_3_3_0.3_0.03_SCF_.csv'

# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Oxygen_psiVextVariation/'
# # file='ds_cellOrder5maxDepth12_2_2_0.2_50000_SCF_.csv'
# file='ds_cellOrder7_maxDepth13_5_5_0.5_100000_SCF_.csv'

# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Oxygen_Hartree/'
# file='ds_krasnyRefine_maxDepth14_5_100_0p05_5000_SCF_.csv'



## Beryllium
#file='LW3_1500_SCF_.csv'

## H2
##file='LW3_2500_SCF_.csv'



## BENZENE TESTS
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/benzeneTests/'  
# # file='tc_gaugeShift0p5_tcOrder8_theta0.8_LW5_500_mixingHistory10_SCF_1485000.csv'
# # file='tc_gaugeShift0p5_mindepth3_tcOrder7_theta0.8_LW5_500_mixingHistory6_SCF_1485000.csv'
# # file='tc_gaugeShift0p5_tcOrder7_theta0.8_LW5_800_mixingHistory6_SCF_2787000.csv'
# # file='tc_gaugeShift0p5_mindepth3_tcOrder7_theta0.8_PCI_1e-2_9random_mixingHistory10_SCF_2172000.csv'
# file='ds_gaugeShift0p25_mindepth3_tcOrder5_theta0.8_LW5_500_1random_mixingHistory10_SCF_1493000.csv'

# ## Carbon Monoxide PCI Testing
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/ParentChildrenIntegral_CO/'

# # file='CO_ds_mixingHistory10_mixingParam0.5_order5_1_1e6_1e6_3e-6_SCF_367625.csv' 
# file='CO_ds_mixingHistory10_mixingParam0.5_order5_1_1e6_1e6_1e-7_SCF_928500.csv'
# file='CO_10orbitals_mixingHistory10_mixingParam0.5_order5_PCI_3e-7_SCF_661625.csv'

# # parent-child integral
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/ParentChildrenIntegral/'
# file='Be_order5_0.000001_SCF_260000.csv'


# ## Symmertric Green's iterations tests
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/symmetricTest/'
# # file='CO_asymm_LW5_SCF_375500.csv'
# file='CO_symm_LW5_SCF_375500.csv'


# # resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/benzeneTests/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/fixedPointImplementation/'
# # file='CO_gaugeShift-0.5_LW5_500_GREEN_375500.csv'
# # file='CO_mixingM5_gaugeShift-0.5_LW5_500_GREEN_1493000.csv'
# # file='scipyAnderson_10initIterations_gaugeShift-0.5_tcOrder8_theta0.7_PCI_1e-06_GREEN_1416000.csv'
# # file='scipyAnderson_looser_gaugeShift-0.5_tcOrder8_theta0.7_PCI_1e-06_GREEN_1416000.csv'
# file='testing_Be_mixingM5_ds_LW5_500_SCF_375500.csv'




# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/noTreeTesting/'
# file='CO_gradual_meshParam_3e-6_mixing_0.2_SCF_370250.csv'

resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/paperData/best-times/'
file='Benzene_init25_mesh_4_3e-6_TC_8_0.6_SCF_2056000.csv'

df = pd.read_csv(resultsDir+file, header=0)   
# df = df.drop(df.index[14]) 

plotsDir = resultsDir+'plots/' 

    
def plotSCFconvergence(df, system = 'H2'):
    
    if system == 'H2':
        dftfeTotalEnergy = -1.1376691191341821e+00
        dftfeExchangeEnergy = -5.5876966592456134e-01
        dftfeCorrelationEnergy = -9.4268448521496129e-02
        dftfeBandEnergy = -7.5499497178953057e-01
    
    if system == "Lithium":
        dftfeTotalEnergy = -7.3340536782581447
        dftfeExchangeEnergy = -1.4916149721121696
        dftfeCorrelationEnergy = -1.5971669832262905e-01
        dftfeBandEnergy = -3.8616389456972078

    if system == "Beryllium":
        dftfeTotalEnergy = -1.4446182766680081e+01
        dftfeExchangeEnergy = -2.2902495359115198e+00
        dftfeCorrelationEnergy = -2.2341044592808737e-01
        dftfeBandEnergy = -8.1239182420318166e+00
        dftfeHartreeEnergy = 7.115165052

    if system == "Oxygen":
#         dftfeTotalEnergy = -7.4469012607372008e+01
#         dftfeExchangeEnergy = -7.2193424444124350e+00
#         dftfeCorrelationEnergy = -5.4455323198374961e-01
#         dftfeBandEnergy = -4.0613156367497737e+01
#         dftfeKineticEnergy = 7.4112265375596365e+01
#         dftfeElectrostaticEnergy = -1.4081739574277032e+02

        # Final converged values
        dftfeTotalEnergy = -7.4469337501098821e+01
        dftfeExchangeEnergy = -7.2193700828939980e+00
        dftfeCorrelationEnergy = -5.4455323568788838e-01
        dftfeBandEnergy = -4.0613397710076626e+01
        dftfeKineticEnergy =  7.4112730191157425e+01
        dftfeElectrostaticEnergy = -1.4081814437367436e+02
        dftfeHartreeEnergy = 36.32506036
        
    if system == "carbonMonoxide":
#         # these taken from mesh size 0.125 run
#         dftfeTotalEnergy = -1.1247167888813128e+02
#         dftfeExchangeEnergy = -1.1997052574614749e+01
#         dftfeCorrelationEnergy = -9.4214501809750550e-01
#         dftfeBandEnergy = -6.2898649220361037e+01
        
        dftfeBandEnergy = -6.2898682441673358e+01 # Band energy 
        dftfeKineticEnergy = 1.1185061770418731e+02 # Kinetic energy 
        dftfeExchangeEnergy = -1.1997011069615391e+01 # Exchange energy 
        dftfeCorrelationEnergy = -9.4214407530225852e-01 # Correlation Energy 
        dftfeElectrostaticEnergy = -2.1138290579726365e+02 # Electrostatic Energy
        dftfeTotalEnergy = -1.1247144323799400e+02 # Total Energy 
        
        dftfeHartreeEnergy=76.1983318
        nAtoms=2
        
    if system == "Benzene":
        
        
#         dftfeBandEnergy = -1.3426386757791246e+02 # Band energy 
#         dftfeExchangeEnergy = -2.9942603449328043e+01 # Exchange energy 
#         dftfeCorrelationEnergy = -2.6706730815123834e+00 # Correlation Energy 
#         dftfeElectrostaticEnergy = -2.1138290579726365e+02 # Electrostatic Energy

        ### SOME OF THESE VALUES COMING FROM BIKASH'S NWChem RUN
#         dftfeBandEnergy = -130.8981224
        dftfeBandEnergy = -130.89648099622937  
        dftfeKineticEnergy = 0 # Kinetic energy 
        dftfeExchangeEnergy = 0 # Exchange energy 
        dftfeCorrelationEnergy = 0 # Correlation Energy 
        dftfeElectrostaticEnergy = 0 # Electrostatic Energy
        dftfeHartreeEnergy = 312.915135214420
        dftfeTotalEnergy = -230.188349460044 # Total Energy
#         dftfeTotalEnergy = -230.18855884130636 
        
        nAtoms=12


    df['bandEnergyError']=abs(df['bandEnergy']-dftfeBandEnergy)
#     df['kineticEnergyError']=abs(df['kineticEnergy']-dftfeKineticEnergy)
#     df['electrostaticEnergyError']=abs(df['electrostaticEnergy']-dftfeElectrostaticEnergy)
    df['hartreeEnergyError']=abs(df['hartreeEnergy']-dftfeHartreeEnergy)
    df['exchangeEnergyError']=abs(df['exchangeEnergy']-dftfeExchangeEnergy)
    df['correlationEnergyError']=abs(df['correlationEnergy']-dftfeCorrelationEnergy)
    df['totalEnergyError']=abs(df['totalEnergy']-dftfeTotalEnergy)
    df['totalEnergyErrorPerAtom']=abs(df['totalEnergy']-dftfeTotalEnergy)/nAtoms
    df['bandEnergyErrorPerAtom']=abs(df['bandEnergy']-dftfeBandEnergy)/nAtoms
    df['hartreeEnergyErrorPerAtom']=abs(df['hartreeEnergy']-dftfeHartreeEnergy)/nAtoms

    print("band energy errors:")
    print(df['bandEnergyError'])
    print("exchange energy errors:")
    print(df['exchangeEnergyError'])
    print("correlation energy errors:")
    print(df['correlationEnergyError'])
    print("total energy errors: \n")
    print(df['totalEnergyErrorPerAtom'])
    

    
# Combined error plot
    f1, ax1 = plt.subplots(1, 1, figsize=(10,6))
    f2, ax2 = plt.subplots(1, 1, figsize=(10,6))
    
#     df.plot(x='Iteration', y='bandEnergyError', logy=True,ax=ax2, style='o')
# #     df.plot(x='Iteration', y='kineticEnergyError', logy=True,ax=ax2, style='o-')
# #     df.plot(x='Iteration', y='electrostaticEnergyError', logy=True,ax=ax2, style='o')
#     df.plot(x='Iteration', y='hartreeEnergyError', logy=True,ax=ax2, style='o')
#     df.plot(x='Iteration', y='exchangeEnergyError', logy=True,ax=ax2, style='o')
#     df.plot(x='Iteration', y='correlationEnergyError',logy=True, ax=ax2, style='o')
# #     df.plot(x='Iteration', y='totalEnergyErrorPerAtom',logy=True, ax=ax2, style='o')
#     df.plot(x='Iteration', y='totalEnergyError',logy=True, ax=ax2, style='o')
    
    df.plot(x='Iteration', y='totalEnergyErrorPerAtom',logy=True, ax=ax2, style='o')
    df.plot(x='Iteration', y='bandEnergyErrorPerAtom',logy=True, ax=ax2, style='o')
    df.plot(x='Iteration', y='hartreeEnergyErrorPerAtom',logy=True, ax=ax2, style='o')

    df.plot(x='Iteration', y='densityResidual', logy=True,ax=ax1, style='o')

    
    ax2.legend(loc='lower left')
##    df.plot(x='Iteration', y='bandEnergyError', logy=True,ax=ax2, style='bo')
##    df.plot(x='Iteration', y='exchangeEnergyError', logy=True,ax=ax2, style='go')
##    df.plot(x='Iteration', y='correlationEnergyError',logy=True, ax=ax2, style='mo')
##    df.plot(x='Iteration', y='totalEnergyError',logy=True, ax=ax2, style='ro')
#     ax0.set_title(system + ': Energy Values During Green Iterations -- Good Initial Guess')
#     ax0.set_title(system + ': Energy Values During Green Iterations -- Bad Initial Guess')
    ax2.set_title(system + ': Energy Errors After Each SCF')
    ax2.set_ylabel('Energy (H)')
    ax2.set_xlabel('SCF Number')
#     plt.savefig(plotsDir+system+'Errors_combined'+'.pdf', bbox_inches='tight',format='pdf')
    
#     ax1.set_title('Oxygen Atom: Density Residual Norm')
    ax1.set_title(system + ': Density Residual Norm')
    ax1.set_ylabel('Density Residual Norm')

    ax1.grid()
    ax2.grid()
#     plt.ylim([5e-7,2e-2])
    plt.show()
    

if __name__=="__main__":
    
#     plotSCFconvergence(df, system="carbonMonoxide")     
#     plotSCFconvergence(df, system="Beryllium")    
#     plotSCFconvergence(df, system="Oxygen")    
    plotSCFconvergence(df, system="Benzene")    




