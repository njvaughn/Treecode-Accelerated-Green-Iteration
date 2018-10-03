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

#### Oxygen
resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenFirstSCF/'

df = pd.read_csv(resultsDir+file, header=0)


if resultsDir == '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenFirstSCF/':
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
    




df['BandEnergyError'] = abs( df['BandEnergy'] - BandEnergy)
df['KineticEnergyError'] = abs( df['KineticEnergy'] - KineticEnergy)
df['ExchangeEnergyError'] = abs( df['ExchangeEnergy'] - ExchangeEnergy)
df['CorrelationEnergyError'] = abs( df['CorrelationEnergy'] - CorrelationEnergy)
df['ElectrostaticEnergyError'] = abs( df['ElectrostaticEnergy'] - ElectrostaticEnergy)
df['TotalEnergyError'] = abs( df['TotalEnergy'] - TotalEnergy)

wavfunctionErrors = np.zeros((df.shape[0],5))
for i in range(df.shape[0]):
    wavfunctionErrors[i,:] = np.array(df.orbitalEnergies[i][1:-1].split(),dtype=float)




def energyErrors():
    fig, ax = plt.subplots(figsize=(8,6))
    fig.suptitle("Oxygen Atom: Energy Errors")
##    fig.suptitle("Beryllium Atom: Energy Errors")
##    fig.suptitle("Hydrogen Molecule: Energy Errors")
    df.plot(x='numberOfPoints', y='BandEnergyError', style='o', ax=ax, loglog=True)
    df.plot(x='numberOfPoints', y='KineticEnergyError', style='o', ax=ax, loglog=True)
    df.plot(x='numberOfPoints', y='ExchangeEnergyError', style='o', ax=ax, loglog=True)
    df.plot(x='numberOfPoints', y='CorrelationEnergyError', style='o', ax=ax, loglog=True)
    df.plot(x='numberOfPoints', y='ElectrostaticEnergyError', style='.', ax=ax, loglog=True)
    df.plot(x='numberOfPoints', y='TotalEnergyError', style='o', ax=ax, loglog=True)
  
    plt.legend(loc = 'best')
    plt.xlabel('Number of Gridpoints')
    plt.ylabel('Energy Error (Hartree)')

    plt.show()


if __name__=="__main__":
    energyErrors()
