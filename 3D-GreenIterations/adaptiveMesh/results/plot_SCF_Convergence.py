'''
Created on Mar 29, 2018

@author: nathanvaughn
'''

import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import os
import numpy as np


##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/carbonMonoxide/iterationResults/'
##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/H2IterationResults/'
resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/BerylliumIterationResults/'
##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenIterationResults/'
plotsDir = resultsDir+'plots/'


## Carbon Monoxide
##file='CO_LW3_1200_singSub_fixedMixingBug_SCF_.csv'
##file='CO_LW3_800_simpleMixing_SCF_.csv'
##file='CO_LW3_1200_simpleMixing_SCF_.csv'
##file='CO_LW1_2000_simpleMixing_SCF_.csv'
##file='CO_LW3_2500_simpleMixing_SCF_.csv'
##file='CO_LW3_1600_simpleMixing_SCF_.csv'
##file='CO_LW3_1000_simpleMixing_SCF_.csv'


## Beryllium
file='LW3_1500_SCF_.csv'

## H2
##file='LW3_2500_SCF_.csv'


df = pd.read_csv(resultsDir+file, header=0)
    
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
        dftfeTotalEnergy = -7.4469012607372008e+01
        dftfeExchangeEnergy = -7.2193424444124350e+00
        dftfeCorrelationEnergy = -5.4455323198374961e-01
        dftfeBandEnergy = -4.0613156367497737e+01
        
    if system == "carbonMonoxide":
        # these taken from mesh size 0.125 run
        dftfeTotalEnergy = -1.1247167888813128e+02
        dftfeExchangeEnergy = -1.1997052574614749e+01
        dftfeCorrelationEnergy = -9.4214501809750550e-01
        dftfeBandEnergy = -6.2898649220361037e+01


    df['bandEnergyError']=abs(df['bandEnergy']-dftfeBandEnergy)
    df['exchangeEnergyError']=abs(df['exchangeEnergy']-dftfeExchangeEnergy)
    df['correlationEnergyError']=abs(df['correlationEnergy']-dftfeCorrelationEnergy)
    df['totalEnergyErrorPerAtom']=abs(df['totalEnergy']-dftfeTotalEnergy)/2

    print("band energy errors:")
    print(df['bandEnergyError'])
    print("exchange energy errors:")
    print(df['exchangeEnergyError'])
    print("correlation energy errors:")
    print(df['correlationEnergyError'])
    print("total energy errors: \n")
    print(df['totalEnergyErrorPerAtom'])
    

    
# Combined error plot
    f2, ax2 = plt.subplots(1, 1, figsize=(10,6))
    df.plot(x='Iteration', y='bandEnergyError', logy=True,ax=ax2, style='bo-')
    df.plot(x='Iteration', y='exchangeEnergyError', logy=True,ax=ax2, style='go-')
    df.plot(x='Iteration', y='correlationEnergyError',logy=True, ax=ax2, style='mo-')
    df.plot(x='Iteration', y='totalEnergyErrorPerAtom',logy=True, ax=ax2, style='ro-')
    
    ax2.legend(loc='upper right')
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



    plt.show()
    

if __name__=="__main__":
    
##    plotSCFconvergence(df_CO, system="carbonMonoxide")    
    plotSCFconvergence(df, system="Beryllium")    
##    plotSCFconvergence(df, system="Oxygen")    



