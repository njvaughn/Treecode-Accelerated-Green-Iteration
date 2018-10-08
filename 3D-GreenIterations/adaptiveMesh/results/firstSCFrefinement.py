'''
Created on Mar 29, 2018

@author: nathanvaughn
'''

import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import os
import numpy as np

file='runComparison.csv'
##file='runComparison_3levels_initial344.csv'
#### Oxygen
##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/Oxygen_SmoothingTests_LW5/'
##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/Oxygen_MeshBuilding/'
##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenFirstSCF_LWtest_singSub2/'
resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenFirstSCF/'
##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenResults_uniformRefinement/'


system="Oxygen"
##if resultsDir == '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenResults_uniformRefinement/':
##if resultsDir == '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenFirstSCF/':
if system == "Oxygen":
    BandEnergy = -4.0613086299670449e+01
    KineticEnergy = 7.4112265375596365e+01
    ExchangeEnergy = -7.2193295915896094e+00
    CorrelationEnergy = -5.4455267145168840e-01
    ElectrostaticEnergy = -1.4081739574277032e+02
    TotalEnergy = -7.4469012630215246e+01

    psi0 = -1.875875893002984895e+01
    psi1 = -8.711960263841991292e-01
    psi2 = -3.382940967105963481e-01
    psi3 = -3.382940967105849128e-01
    psi4 = -3.382940967105836916e-01
    


df = pd.read_csv(resultsDir+file, header=0)
df['BandEnergyError'] = abs( df['BandEnergy'] - BandEnergy)
df['KineticEnergyError'] = abs( df['KineticEnergy'] - KineticEnergy)
df['ExchangeEnergyError'] = abs( df['ExchangeEnergy'] - ExchangeEnergy)
df['CorrelationEnergyError'] = abs( df['CorrelationEnergy'] - CorrelationEnergy)
df['ElectrostaticEnergyError'] = abs( df['ElectrostaticEnergy'] - ElectrostaticEnergy)
df['TotalEnergyError'] = abs( df['TotalEnergy'] - TotalEnergy)

##df2 = df[df['divideCriterion']=='LW2']
##df3 = df[df['divideCriterion']=='LW3']
##df4 = df[df['divideCriterion']=='LW4']
##df5 = df[df['divideCriterion']=='LW5']
##df6 = df[df['divideCriterion']==6]

df3 = df[df['order']==3]
df4 = df[df['order']==4]
df5 = df[df['order']==5]
df6 = df[df['order']==6]




wavfunctionErrors = np.zeros((df.shape[0],5))
for i in range(df.shape[0]):
    wavfunctionErrors[i,:] = np.array(df.orbitalEnergies[i][1:-1].split(),dtype=float)




def energyErrors(dataframe):
    print(dataframe)
    fig, ax = plt.subplots(figsize=(8,6))
    fig.suptitle("Oxygen Atom: Energy Errors")
##    fig.suptitle("Beryllium Atom: Energy Errors")
##    fig.suptitle("Hydrogen Molecule: Energy Errors")
    dataframe.plot(x='numberOfPoints', y='BandEnergyError', style='o', ax=ax, loglog=True)
    dataframe.plot(x='numberOfPoints', y='KineticEnergyError', style='o', ax=ax, loglog=True)
    dataframe.plot(x='numberOfPoints', y='ExchangeEnergyError', style='o', ax=ax, loglog=True)
    dataframe.plot(x='numberOfPoints', y='CorrelationEnergyError', style='o', ax=ax, loglog=True)
    dataframe.plot(x='numberOfPoints', y='ElectrostaticEnergyError', style='o', ax=ax, loglog=True)
    dataframe.plot(x='numberOfPoints', y='TotalEnergyError', style='o', ax=ax, loglog=True)
  
    plt.legend(loc = 'lower left')
    plt.xlabel('Number of Gridpoints')
    plt.ylabel('Energy Error (Hartree)')

    plt.show()

def kineticEnergyErrors(dataframe):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
    ax1.set_title("Number of Gridpoints")


##    grouped = dataframe.groupby('order')
##    for name,group in grouped:
##        group.plot(x='numberOfPoints', y='KineticEnergyError', style='o', ax=ax1, loglog=True, label='%s = %s'%('Kinetic Error: Order',name))
##        group.plot(x='numberOfPoints', y='TotalEnergyError', style='o', ax=ax1, loglog=True, label='%s = %s'%('Total Error: Order',name))
        
    dataframe.plot(x='numberOfPoints', y='KineticEnergyError', style='o', ax=ax1, loglog=True)
    dataframe.plot(x='numberOfPoints', y='TotalEnergyError', style='o', ax=ax1, loglog=True)
    ax1.legend()
    ax1.set_xlabel('Number of Gridpoints')
    ax1.set_ylabel('Energy Error (Hartree)')

    ax1.set_title("Errors Versus N")

##    for name,group in grouped:
##        group.plot(x='maxDepth', y='KineticEnergyError', style='o', ax=ax2, logy=True, label='%s = %s'%('Kinetic Error: Order',name))
##        group.plot(x='maxDepth', y='TotalEnergyError', style='o', ax=ax2, logy=True, label='%s = %s'%('Total Error: Order',name))
    dataframe.plot(x='maxDepth', y='KineticEnergyError', style='o', ax=ax2, logy=True)
    dataframe.plot(x='maxDepth', y='TotalEnergyError', style='o', ax=ax2, logy=True)
    ax2.legend()
    ax2.set_xlabel('Mesh Depth at Nucleus')
    ax2.set_ylabel('Energy Error (Hartree)')
    ax2.set_title("Errors Versus Max Depth")

##    plt.suptitle('Oxygen Atom Kinetic Energy Errors')
    plt.tight_layout(pad=2.0)
    plt.show()
    


if __name__=="__main__":
##    pass
    energyErrors(df)
##    kineticEnergyErrors(df4)
