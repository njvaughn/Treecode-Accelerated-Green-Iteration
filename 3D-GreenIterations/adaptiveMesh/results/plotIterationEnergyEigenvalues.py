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

#### H2
##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/H2IterationResults/'
##file='LW3_2500_GREEN_.csv'
##
#### Lithium
##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/LithiumIterationResults/'
##file='LW3_1000_GREEN_.csv'

#### Beryllium
##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/LithiumIterationResults/'
##file='LW3_1000_GREEN_.csv'

## Oxygen
##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenIterationResults/'
##file='LW3_1500_GREEN_.csv'
resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/Oxygen_SmoothingTests_LW5/'
file = 'LW5_4000_N3_EPSp25_GREEN_.csv'
##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenOrder5/'
##file='LW3_1000_GREEN_.csv'

if resultsDir == '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/LithiumIterationResults/':
    TotalEnergy = -7.3340536782581447
    ExchangeEnergy = -1.4916149721121696
    CorrelationEnergy = -0.15971669832262905
    BandEnergy = -3.8616389456972078


if resultsDir == '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/BerylliumIterationResults/':
    TotalEnergy = -1.4446182766680081e+01
    ExchangeEnergy = -2.2902495359115198e+00
    CorrelationEnergy = -2.2341044592808737e-01
    BandEnergy = -8.1239182420318166e+00

    df = df.drop([6,7,8,9,10])  #because I accidentally wrote to the same file again



##if resultsDir == '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenIterationResults/':
##if resultsDir == '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenOrder5/':
BandEnergy = -4.0613161867355522e+01
KineticEnergy = 7.4112356154762352e+01
ExchangeEnergy = -7.2193342430668448e+00
CorrelationEnergy = -5.4455279159404091e-01
ElectrostaticEnergy = -1.4081748175072624e+02
TotalEnergy = -7.4469012630624775e+01

psi0 = -1.875879493052850933e+01
psi1 = -8.711989839502347621e-01
psi2 = -3.382966064098608672e-01
psi3 = -3.382966063773331644e-01
psi4 = -3.382966063244193800e-01


if resultsDir == '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/H2IterationResults/':
    TotalEnergy = -1.1376691191341821e+00
    ExchangeEnergy = -5.5876966592456134e-01
    CorrelationEnergy = -9.4268448521496129e-02
    BandEnergy = -7.5499497178953057e-01



plotsDir = resultsDir+'plots/'

df = pd.read_csv(resultsDir+file, header=0)

print(df.shape)

##referenceEnergies = np.array([-1.871923237485756886e+01,
##                              -9.907091923705507952e+00,
##                              -1.075296339177760352e+00,
##                              -5.215175505938050016e-01,
##                              -4.455359698088234843e-01,
##                              -4.455359698088187659e-01 ,
##                              -3.351144822320292205e-01])

## OXYGEN
referenceEnergies = np.array([psi0,psi1,psi2,psi3,psi4])

## H2
##referenceEnergies = np.array([-0.3774974859])


residualsMatrix = np.zeros((df.shape[0],5))
errorsMatrix = np.zeros((df.shape[0],5))
errorsMatrix1st = np.zeros((df.shape[0],5))
for i in range(df.shape[0]):
    residualsMatrix[i,:] = np.array(df.orbitalResiduals[i][1:-1].split(),dtype=float)
    errorsMatrix[i,:] = abs( np.array( df.energyEigenvalues[i][1:-1].split(),dtype=float) - referenceEnergies )
##    errorsMatrix[i,:] = np.array( df.energyEigenvalues[i][1:-1].split(),dtype=float)
    try:
        errorsMatrix1st[i,:] = np.array( df.energyErrorsWRTfirstSCF[i][1:-1].split(),dtype=float) 
    except AttributeError:
        pass

df['residual0'] = residualsMatrix[:,0]
df['residual1'] = residualsMatrix[:,1]
df['residual2'] = residualsMatrix[:,2]
df['residual3'] = residualsMatrix[:,3]
df['residual4'] = residualsMatrix[:,4]
##df['residual5'] = residualsMatrix[:,5]
##df['residual6'] = residualsMatrix[:,6]
##df['residual7'] = residualsMatrix[:,7]
##df['residual8'] = residualsMatrix[:,8]
##df['residual9'] = residualsMatrix[:,9]

##df['errors0'] = np.copy(df['energyEigenvalues'])
df['errors0'] = np.abs(errorsMatrix[:,0])
df['errors1'] = np.abs(errorsMatrix[:,1])
df['errors2'] = np.abs(errorsMatrix[:,2])
df['errors3'] = np.abs(errorsMatrix[:,3])
df['errors4'] = np.abs(errorsMatrix[:,4])
##df['errors5'] = np.abs(errorsMatrix[:,5])
##df['errors6'] = np.abs(errorsMatrix[:,6])


##try:
##    df['1stSCFerrors0'] = np.abs(errorsMatrix1st[:,0])
##    df['1stSCFerrors1'] = np.abs(errorsMatrix1st[:,1])
##    df['1stSCFerrors2'] = np.abs(errorsMatrix1st[:,2])
##    df['1stSCFerrors3'] = np.abs(errorsMatrix1st[:,3])
##    df['1stSCFerrors4'] = np.abs(errorsMatrix1st[:,4])
##    df['1stSCFerrors5'] = np.abs(errorsMatrix1st[:,5])
##    df['1stSCFerrors6'] = np.abs(errorsMatrix1st[:,6])
##except AttributeError:
##    pass

def plotFirstSCF(df):

    
    f0, (ax0, ax1) = plt.subplots(1,2, figsize=(12,6))
    df.plot(y='residual0',ax=ax0,logy=True,label='Phi0')
    df.plot(y='residual1',ax=ax0,logy=True,label='Phi1')
    df.plot(y='residual2',ax=ax0,logy=True,label='Phi2')
    df.plot(y='residual3',ax=ax0,logy=True,label='Phi3')
    df.plot(y='residual4',ax=ax0,logy=True,label='Phi4')
##    df.plot(y='residual5',ax=ax0,logy=True,label='Phi5')
##    df.plot(y='residual6',ax=ax0,logy=True,label='Phi6')
##    df.plot(y='residual7',ax=ax0,logy=True,label='Phi7')
##    df.plot(y='residual8',ax=ax0,logy=True,label='Phi8')
##    df.plot(y='residual9',ax=ax0,logy=True,label='Phi9')
    ax0.set_xlabel('Iteration Number')
    ax0.set_ylabel('Residual L2 Norm')
    ax0.set_title('Orbital Residuals')


    df.plot(y='errors0',ax=ax1,logy=True,label='Phi0')
    df.plot(y='errors1',ax=ax1,logy=True,label='Phi1')
    df.plot(y='errors2',ax=ax1,logy=True,label='Phi2')
    df.plot(y='errors3',ax=ax1,logy=True,label='Phi3')
    df.plot(y='errors4',ax=ax1,logy=True,label='Phi4')
##    df.plot(y='errors5',ax=ax1,logy=True,label='Phi5')
##    df.plot(y='errors6',ax=ax1,logy=True,label='Phi6')
##    df.plot(y='errors7',ax=ax1,logy=True,label='Phi7')
##    df.plot(y='errors8',ax=ax1,logy=True,label='Phi8')
##    df.plot(y='errors9',ax=ax1,logy=True,label='Phi9')
    ax1.set_xlabel('Iteration Number')
    ax1.set_ylabel('Energy Error (Hartree)')
##    ax1.set_ylim([1e-4,2e-2])
    ax1.set_title('Orbital Errors')

##    plt.suptitle('Using Singularity Skipping, LW3-800')
##    plt.suptitle('Using Singularity Subtraction, LW3-800, minDepth 3')
##    plt.suptitle(file)
##    plt.suptitle('Convergence of Green Iterations for Oxygen -- Coarse')
    plt.suptitle('Convergence of Green Iterations for Oxygen')

##    try:
##        f1, (ax2,ax3) = plt.subplots(1,2, figsize=(12,6))
##        df.plot(y='residual0',ax=ax2,logy=True,label='Phi0')
##        df.plot(y='residual1',ax=ax2,logy=True,label='Phi1')
##        df.plot(y='residual2',ax=ax2,logy=True,label='Phi2')
##        df.plot(y='residual3',ax=ax2,logy=True,label='Phi3')
##        df.plot(y='residual4',ax=ax2,logy=True,label='Phi4')
##        df.plot(y='residual5',ax=ax2,logy=True,label='Phi5')
##        df.plot(y='residual6',ax=ax2,logy=True,label='Phi6')
##        plt.suptitle(file+'Errors w.r.t. first SCF energies')
##        df.plot(y='1stSCFerrors0',ax=ax3,logy=True,label='Phi0')
##        df.plot(y='1stSCFerrors1',ax=ax3,logy=True,label='Phi1')
##        df.plot(y='1stSCFerrors2',ax=ax3,logy=True,label='Phi2')
##        df.plot(y='1stSCFerrors3',ax=ax3,logy=True,label='Phi3')
##        df.plot(y='1stSCFerrors4',ax=ax3,logy=True,label='Phi4')
##        df.plot(y='1stSCFerrors5',ax=ax3,logy=True,label='Phi5')
##        df.plot(y='1stSCFerrors6',ax=ax3,logy=True,label='Phi6')
##    except AttributeError:
##        pass
    
    plt.show()



    

if __name__=="__main__":
    plotFirstSCF(df)
    


