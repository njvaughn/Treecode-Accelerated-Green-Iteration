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
##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenSmoothingPreSCF/'
resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenPreSCF_GaussianSS/'

##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/Oxygen_MeshBuilding/'
##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenFirstSCF_LWtest_singSub2/'
##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenFirstSCF/'
##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/StaggeredGridTests/'
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

df_smoothing = df

df5 = df[df['order']==5]

##df_double = df.loc[0:6]
##
resultsDir_baseline = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/BaselineNoStaggerNoSmooth/'
df_single = pd.read_csv(resultsDir_baseline+file, header=0)
df_single['BandEnergyError'] = abs( df_single['BandEnergy'] - BandEnergy)
df_single['KineticEnergyError'] = abs( df_single['KineticEnergy'] - KineticEnergy)
df_single['ExchangeEnergyError'] = abs( df_single['ExchangeEnergy'] - ExchangeEnergy)
df_single['CorrelationEnergyError'] = abs( df_single['CorrelationEnergy'] - CorrelationEnergy)
df_single['ElectrostaticEnergyError'] = abs( df_single['ElectrostaticEnergy'] - ElectrostaticEnergy)
df_single['TotalEnergyError'] = abs( df_single['TotalEnergy'] - TotalEnergy)

resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/StaggeredGridTests/'
df2 = pd.read_csv(resultsDir+file, header=0)
df2['BandEnergyError'] = abs( df2['BandEnergy'] - BandEnergy)
df2['KineticEnergyError'] = abs( df2['KineticEnergy'] - KineticEnergy)
df2['ExchangeEnergyError'] = abs( df2['ExchangeEnergy'] - ExchangeEnergy)
df2['CorrelationEnergyError'] = abs( df2['CorrelationEnergy'] - CorrelationEnergy)
df2['ElectrostaticEnergyError'] = abs( df2['ElectrostaticEnergy'] - ElectrostaticEnergy)
df2['TotalEnergyError'] = abs( df2['TotalEnergy'] - TotalEnergy)
df_double = df2.loc[0:6]


wavfunctionErrors = np.zeros((df.shape[0],5))
for i in range(df.shape[0]):
    wavfunctionErrors[i,:] = np.array(df.orbitalEnergies[i][1:-1].split(),dtype=float)




def energyErrors(dataframe):
    print(dataframe)
    fig, ax = plt.subplots(figsize=(8,6))
    fig.suptitle("Oxygen Atom: Energy Errors")
##    fig.suptitle("Beryllium Atom: Energy Errors")
##    fig.suptitle("Hydrogen Molecule: Energy Errors")
##    dataframe.plot(x='numberOfPoints', y='BandEnergyError', style='o', ax=ax, loglog=True)
    dataframe.plot(x='numberOfPoints', y='KineticEnergyError', style='o', ax=ax, loglog=True)
    dataframe.plot(x='numberOfPoints', y='ExchangeEnergyError', style='o', ax=ax, loglog=True)
    dataframe.plot(x='numberOfPoints', y='CorrelationEnergyError', style='o', ax=ax, loglog=True)
    dataframe.plot(x='numberOfPoints', y='ElectrostaticEnergyError', style='o', ax=ax, loglog=True)
    dataframe.plot(x='numberOfPoints', y='TotalEnergyError', style='^', ax=ax, loglog=True)
  
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


def compareSingleToDoubleMesh(df_single, df_double):
    fig, ax = plt.subplots(figsize=(8,6))
    fig.suptitle("Oxygen Atom: Energy Errors")
    df_single.plot(x='numberOfPoints', y='ElectrostaticEnergyError', label='Electrostatic: Single Mesh', style='o', ax=ax, loglog=True)
    df_single.plot(x='numberOfPoints', y='TotalEnergyError', style='^', label='Total Error: Single Mesh', ax=ax, loglog=True)

    df_double.plot(x='numberOfPoints', y='ElectrostaticEnergyError', label='Electrostatic: Double Mesh', style='o', ax=ax, loglog=True)
    df_double.plot(x='numberOfPoints', y='TotalEnergyError', style='^', label='Total Error: Double Mesh', ax=ax, loglog=True)
  
    plt.legend(loc = 'lower left')
    plt.xlabel('Number of Gridpoints')
    plt.ylabel('Energy Error (Hartree)')

    plt.ylim([1e-3, 1e-1])
    plt.show()

def compareSingleToDoubleMeshToSmoothing(df_single, df_double, df_smoothing):
    fig, ax = plt.subplots(figsize=(8,6))
    fig.suptitle("Oxygen Atom: Energy Errors")
    df_single.plot(x='numberOfPoints', y='ElectrostaticEnergyError', label='Electrostatic: Single Mesh', style='o', ax=ax, loglog=True)

    df_double.plot(x='numberOfPoints', y='ElectrostaticEnergyError', label='Electrostatic: Double Mesh', style='o', ax=ax, loglog=True)

    df_smoothing.plot(x='numberOfPoints', y='ElectrostaticEnergyError', label='Electrostatic: Regularized', style='o', ax=ax, loglog=True)
  
    plt.legend(loc = 'lower left')
    plt.xlabel('Number of Gridpoints')
    plt.ylabel('Energy Error (Hartree)')

    plt.ylim([1e-3, 1e-1])
    plt.show()



if __name__=="__main__":
##    pass
    energyErrors(df5)
##    kineticEnergyErrors(df4)
##    compareSingleToDoubleMesh(df_single,df_double)
##    compareSingleToDoubleMesh(df_single,df_smoothing)
##    compareSingleToDoubleMeshToSmoothing(df_single, df_double, df_smoothing)
