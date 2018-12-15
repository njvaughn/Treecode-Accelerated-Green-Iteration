'''
Created on Mar 29, 2018

@author: nathanvaughn
'''

import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import os
import numpy as np


resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/mixingComparisonBe/'
plotsDir = resultsDir+'plots/'
file_noMixing='LW5_500_noMixing_1em8_SCF_.csv'
file_simpleMixing='LW5_500_simpleMixing_p5_1em8_SCF_.csv'
file_andersonMixing='LW5_500_andersonMixing_p5_1em8_SCF_.csv'




df_noMix = pd.read_csv(resultsDir+file_noMixing, header=0)
df_simple = pd.read_csv(resultsDir+file_simpleMixing, header=0)
df_anderson = pd.read_csv(resultsDir+file_andersonMixing, header=0)

print(df_anderson)
# print(df_simple)
    
def plotSCFconvergence(df_noMix, df_simple, df_anderson, system = 'Beryllium'):
    

    if system == "Oxygen":
        # Final converged values
        dftfeTotalEnergy = -7.4469337501098821e+01
        dftfeExchangeEnergy = -7.2193700828939980e+00
        dftfeCorrelationEnergy = -5.4455323568788838e-01
        dftfeBandEnergy = -4.0613397710076626e+01
        dftfeKineticEnergy =  7.4112730191157425e+01
        dftfeElectrostaticEnergy = -1.4081814437367436e+02
        
    if system == "Beryllium":
        # Final converged values
        dftfeTotalEnergy = -1.4446201118081863e+01
        
        



#     df['bandEnergyError']=abs(df['bandEnergy']-dftfeBandEnergy)
#     df['kineticEnergyError']=abs(df['kineticEnergy']-dftfeKineticEnergy)
#     df['electrostaticEnergyError']=abs(df['electrostaticEnergy']-dftfeElectrostaticEnergy)
#     df['exchangeEnergyError']=abs(df['exchangeEnergy']-dftfeExchangeEnergy)
#     df['correlationEnergyError']=abs(df['correlationEnergy']-dftfeCorrelationEnergy)

    df_noMix = df_noMix.drop(df_noMix.index[0])
    df_simple = df_simple.drop(df_simple.index[0])
    df_anderson = df_anderson.drop(df_anderson.index[0])
#     print(df_anderson)

    df_noMix['totalEnergyError']=abs(df_noMix['totalEnergy']-dftfeTotalEnergy)
    df_simple['totalEnergyError']=abs(df_simple['totalEnergy']-dftfeTotalEnergy)
    df_anderson['totalEnergyError']=abs(df_anderson['totalEnergy']-dftfeTotalEnergy)
    
    print(df_noMix['totalEnergyError'])

  
    
    f0, (ax0, ax1) = plt.subplots(1,2, figsize=(12,6))
    
    df_noMix.plot(x='Iteration', y='densityResidual',logy=True, ax=ax0, style='o-',label='No Mixing')
    df_simple.plot(x='Iteration', y='densityResidual',logy=True, ax=ax0, style='o-', label='Simple')
    df_anderson.plot(x='Iteration', y='densityResidual',logy=True, ax=ax0, style='o-', label='Anderson')
    
    
    df_noMix.plot(x='Iteration', y='totalEnergyError',logy=True, ax=ax1, style='o-',label='No Mixing')
    df_simple.plot(x='Iteration', y='totalEnergyError',logy=True, ax=ax1, style='o-', label='Simple')
    df_anderson.plot(x='Iteration', y='totalEnergyError',logy=True, ax=ax1, style='o-', label='Anderson')
    
  
    
    ax1.set_xlabel('Iteration Number')
    ax1.set_ylabel('Total Energy Error (H)')
    
    ax0.set_xlabel('Iteration Number')
    ax0.set_ylabel('Density Residual L2 Norm')
#     ax0.set_title('Density Residual Norm')
    
    
#     ax0.set_title('Total Energy Error')


    plt.suptitle('SCF Convergence for Beryllium Atom (mixing parameter = 0.5)')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
    

if __name__=="__main__":
    
    plotSCFconvergence(df_noMix, df_simple, df_anderson, system="Beryllium")    




