'''
Created on Mar 29, 2018

@author: nathanvaughn
'''

import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import os
import numpy as np


resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/carbonMonoxide/iterationResults/'
file = 'CO_refinement_results.csv'

# df = pd.read_csv(resultsDir+file, header=0)
df = pd.read_csv(file, header=0,comment="#")

print(df.shape)

df['EnergyError1'] = -1.1247200574178584e+02 - df['TotalEnergy']
df['EnergyError2'] = -1.1248147249764637e+02 - df['TotalEnergy']
df['EnergyError3'] = -1.1247635297749687e+02 - df['TotalEnergy']

df['TotalEnergyErrorPerAtom'] = abs( (df['TotalEnergy']+1.1247167888813128e+02)/2 )
df['BandEnergyError'] = abs( df['BandEnergy']+6.2898649220361037e+01 )
df['CorrelationEnergyError'] = abs( df['CorrelationEnergy']+9.4214501809750550e-01 )
df['ExchangeEnergyError'] = abs( df['ExchangeEnergy']+1.1997052574614749e+01 )



def AversusB(df,A,B,save=False):
    fig, ax = plt.subplots(figsize=(8,6))
    fig.suptitle('%s versus %s' %(A,B))
    df.plot(x=B, y=A, style='o',ax=ax)

    if A=='TotalEnergy':
        dftfeEnergy1 = -1.1247200574178584e+02
        dftfeEnergy2 = -1.1248147249764637e+02
        dftfeEnergy3 = -1.1247635297749687e+02
        plt.axhline(y=dftfeEnergy1,color='r',label='dft-fe1')
        plt.axhline(y=dftfeEnergy2,color='g',label='dft-fe2')
        plt.axhline(y=dftfeEnergy3,color='b',label='dft-fe3')
        plt.legend()
        plt.ylim([-113,-112.3])

    if save == True:
        saveID = A+'Vs'+B
        plt.savefig(plotsDir+saveID+'.pdf', bbox_inches='tight',format='pdf')
    plt.show()

def logAversusLogB(df,A,B,save=False):
    fig, ax = plt.subplots(figsize=(8,6))
    fig.suptitle('%s versus %s' %(A,B))
    df.plot(x=B, y=A, style='o', ax=ax, loglog=True)

        
    plt.legend(loc = 'best')
    plt.xlabel(B)
    plt.ylabel(A)

    if save == True:
        saveID = 'log'+A+'VsLog'+B
        plt.savefig(plotsDir+saveID+'.pdf', bbox_inches='tight',format='pdf')
    plt.show()

def energyErrors():
    fig, ax = plt.subplots(figsize=(8,6))
    fig.suptitle("Energy Errors w.r.t. DFT-FE")
    df.plot(x='NumberOfGridpoints', y='BandEnergyError', style='o', ax=ax, loglog=True)
    df.plot(x='NumberOfGridpoints', y='ExchangeEnergyError', style='o', ax=ax, loglog=True)
    df.plot(x='NumberOfGridpoints', y='CorrelationEnergyError', style='o', ax=ax, loglog=True)
    df.plot(x='NumberOfGridpoints', y='TotalEnergyErrorPerAtom', style='o', ax=ax, loglog=True)
##
        
##    df.plot(x='NumberOfGridpoints', y='BandEnergyError', ax=ax, loglog=True)
##    df.plot(x='NumberOfGridpoints', y='ExchangeEnergyError', ax=ax, loglog=True)
##    df.plot(x='NumberOfGridpoints', y='CorrelationEnergyError', ax=ax, loglog=True)
##    df.plot(x='NumberOfGridpoints', y='TotalEnergyErrorPerAtom', ax=ax, loglog=True)


    plt.legend(loc = 'best')
    plt.xlabel('Number of Gridpoints')
    plt.ylabel('Energy Error (Hartree)')

    plt.show()


def energyValues():
    fig, ax = plt.subplots(figsize=(8,6))
    fig.suptitle("Energy Eigenvalues")
    df.plot(x='NumberOfGridpoints', y='Orbital0Energy', style='o', ax=ax)
    df.plot(x='NumberOfGridpoints', y='Orbital1Energy', style='o', ax=ax)
    df.plot(x='NumberOfGridpoints', y='Orbital2Energy', style='o', ax=ax)
    df.plot(x='NumberOfGridpoints', y='Orbital3Energy', style='o', ax=ax)
    df.plot(x='NumberOfGridpoints', y='Orbital4Energy', style='o', ax=ax)
    df.plot(x='NumberOfGridpoints', y='Orbital5Energy', style='o', ax=ax)
    df.plot(x='NumberOfGridpoints', y='Orbital6Energy', style='o', ax=ax)

        
    plt.legend(loc = 'best')
    plt.xlabel('Number of Gridpoints')
    plt.ylabel('Energy Eigenvalues')

    plt.show()


    

if __name__=="__main__":
#     plotFirstSCF(df)
##    AversusB(df,'TotalEnergy','NumberOfGridpoints')
##    logAversusLogB(df,'EnergyError2','NumberOfGridpoints')
    energyErrors()

##    energyValues()    

