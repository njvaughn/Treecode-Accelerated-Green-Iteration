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
# # #resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/BerylliumIterationResults/'
# # ##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenIterationResults/'
# # #resultsDir = '/Users/nathanvaughn/Desktop/scratch/O_Gaussian/'
# resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/oxygen_with_anderson/'
# plotsDir = resultsDir+'plots/'
# # file='LW5_1000_andersonMixing_p5_1em76_SCF_.csv'
# # file='LW5_1500_andersonMixing_p5_1em8_SCF_.csv'
# file='LW5_2000_andersonMixing_p5_1em76_SCF_.csv'

# ## Carbon Monoxide
resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/'
plotsDir = resultsDir+'plots/'
# file ='LW5_1500o5_GradientFree_eigRes_looseThenTight_titan_SCF_.csv'
# file ='LW5o5_1500_SCF_.csv'
# file='LW5o5_1500_largeDomain_SCF_.csv'
# file='LW5o5_1000_fixedMesh_only7_looseInit_SCF_.csv'
file='LW5o5_1500_fixedAtomicPositions_only7_looseInit_SCF_.csv'
# file='LW5o4_1000_only7_tightFromStart_GIanderson_afterSCF1_SCF_.csv'
# file='LW5o5_2000_only7_tightFromStart_GIandersonAfterSCF1_SCF_.csv'
# file='LW5o5_2000_6_orbitals_SCF_.csv'
# file='LW5o5_2000_7_orbitals_noGIanderson_SCF_.csv'

# ## Oxygen -- Biros
# resultsDir = '/Users/nathanvaughn/Desktop/meshTests/LWvsBiros/Oxygen/'
# plotsDir = resultsDir+'plots/'
# # file='Biros_o7_1em4_solo_SCF_.csv'
# # file='Biros_o7_7em5_alpha_1p5_SCF_.csv'
# # file='BirosN_o7_1em3_SCF_.csv'
# file='BirosN_o7_2em4_SCF_.csv'

## Beryllium
#file='LW3_1500_SCF_.csv'

## H2
##file='LW3_2500_SCF_.csv'


df = pd.read_csv(resultsDir+file, header=0)
# df = df.drop(df.index[14]) 
    
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


    df['bandEnergyError']=abs(df['bandEnergy']-dftfeBandEnergy)
    df['kineticEnergyError']=abs(df['kineticEnergy']-dftfeKineticEnergy)
    df['electrostaticEnergyError']=abs(df['electrostaticEnergy']-dftfeElectrostaticEnergy)
    df['exchangeEnergyError']=abs(df['exchangeEnergy']-dftfeExchangeEnergy)
    df['correlationEnergyError']=abs(df['correlationEnergy']-dftfeCorrelationEnergy)
    df['totalEnergyErrorPerAtom']=abs(df['totalEnergy']-dftfeTotalEnergy)/1

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
    df.plot(x='Iteration', y='bandEnergyError', logy=True,ax=ax2, style='o')
#     df.plot(x='Iteration', y='kineticEnergyError', logy=True,ax=ax2, style='o-')
    df.plot(x='Iteration', y='electrostaticEnergyError', logy=True,ax=ax2, style='o')
    df.plot(x='Iteration', y='exchangeEnergyError', logy=True,ax=ax2, style='o')
    df.plot(x='Iteration', y='correlationEnergyError',logy=True, ax=ax2, style='o')
    df.plot(x='Iteration', y='totalEnergyErrorPerAtom',logy=True, ax=ax2, style='o')
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
    plt.savefig(plotsDir+system+'Errors_combined'+'.pdf', bbox_inches='tight',format='pdf')
    
    ax1.set_title('Oxygen Atom: Density Residual Norm')
    ax1.set_ylabel('Density Residual Norm')

    ax1.grid()
    ax2.grid()
    
    plt.show()
    

if __name__=="__main__":
    
    plotSCFconvergence(df, system="carbonMonoxide")    
#    plotSCFconvergence(df, system="Beryllium")    
#     plotSCFconvergence(df, system="Oxygen")    




