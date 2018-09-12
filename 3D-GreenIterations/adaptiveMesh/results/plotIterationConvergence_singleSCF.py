'''
Created on Mar 29, 2018

@author: nathanvaughn
'''

import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import os
import numpy as np


resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/singleSCF/'
file='CO_LW3_400_SCF1_skipping_minD4.csv'
plotsDir = resultsDir+'plots/'

##df = pd.read_csv(resultsDir+'/CO_LW3_800_SCF1_skipping.csv', header=0)
##df = pd.read_csv(resultsDir+'/CO_LW3_800_SCF1_subtracting.csv', header=0)
##df = pd.read_csv(resultsDir+'/CO_LW3_800_SCF1_skipping_minD3.csv', header=0)
df = pd.read_csv(resultsDir+file, header=0)

print(df.shape)

residualsMatrix = np.zeros((df.shape[0],7))
errorsMatrix = np.zeros((df.shape[0],7))
for i in range(df.shape[0]):
##    print('i=%i'%i)
##    print(np.array(df.orbitalResiduals[i][1:-1].split('  '),dtype=float))
    residualsMatrix[i,:] = np.array(df.orbitalResiduals[i][1:-1].split(),dtype=float)
    errorsMatrix[i,:] = np.array( df.energyErrors[i][1:-1].split(),dtype=float) 

df['residual0'] = residualsMatrix[:,0]
df['residual1'] = residualsMatrix[:,1]
df['residual2'] = residualsMatrix[:,2]
df['residual3'] = residualsMatrix[:,3]
df['residual4'] = residualsMatrix[:,4]
df['residual5'] = residualsMatrix[:,5]
df['residual6'] = residualsMatrix[:,6]

df['errors0'] = np.abs(errorsMatrix[:,0])
df['errors1'] = np.abs(errorsMatrix[:,1])
df['errors2'] = np.abs(errorsMatrix[:,2])
df['errors3'] = np.abs(errorsMatrix[:,3])
df['errors4'] = np.abs(errorsMatrix[:,4])
df['errors5'] = np.abs(errorsMatrix[:,5])
df['errors6'] = np.abs(errorsMatrix[:,6])


def plotFirstSCF():
    f0, (ax0, ax1) = plt.subplots(1,2, figsize=(12,6))
    df.plot(y='residual0',ax=ax0,logy=True,label='Phi0')
    df.plot(y='residual1',ax=ax0,logy=True,label='Phi1')
    df.plot(y='residual2',ax=ax0,logy=True,label='Phi2')
    df.plot(y='residual3',ax=ax0,logy=True,label='Phi3')
    df.plot(y='residual4',ax=ax0,logy=True,label='Phi4')
    df.plot(y='residual5',ax=ax0,logy=True,label='Phi5')
    df.plot(y='residual6',ax=ax0,logy=True,label='Phi6')
    ax0.set_xlabel('Iteration Number')
    ax0.set_ylabel('Residual L2 Norm')
    ax0.set_title('Orbital Residuals')


    df.plot(y='errors0',ax=ax1,logy=True,label='Phi0')
    df.plot(y='errors1',ax=ax1,logy=True,label='Phi1')
    df.plot(y='errors2',ax=ax1,logy=True,label='Phi2')
    df.plot(y='errors3',ax=ax1,logy=True,label='Phi3')
    df.plot(y='errors4',ax=ax1,logy=True,label='Phi4')
    df.plot(y='errors5',ax=ax1,logy=True,label='Phi5')
    df.plot(y='errors6',ax=ax1,logy=True,label='Phi6')
    ax1.set_xlabel('Iteration Number')
    ax1.set_ylabel('Energy Error (Hartree)')
    ax1.set_title('Orbital Errors')

##    plt.suptitle('Using Singularity Skipping, LW3-800')
    plt.suptitle('Using Singularity Subtraction, LW3-800, minDepth 3')
    plt.suptitle(file)
    plt.show()

    

if __name__=="__main__":
    plotFirstSCF()
    


