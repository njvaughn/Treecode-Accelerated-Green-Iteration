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

##file='CO_LW3_1200_singSub_fixedMixingBug_GREEN_.csv'

##file='CO_LW3_800_simpleMixing_GREEN_.csv'
##file='CO_LW3_1200_simpleMixing_GREEN_.csv'
##file='CO_LW3_1000_simpleMixing_GREEN_.csv'
##file='CO_LW3_1600_simpleMixing_GREEN_.csv'
##file='CO_LW1_2000_simpleMixing_GREEN_.csv'
file='CO_LW3_2500_simpleMixing_GREEN_.csv'



plotsDir = resultsDir+'plots/'

df = pd.read_csv(resultsDir+file, header=0)

print(df.shape)

referenceEnergies = np.array([-1.871923237485756886e+01,
                              -9.907091923705507952e+00,
                              -1.075296339177760352e+00,
                              -5.215175505938050016e-01,
                              -4.455359698088234843e-01,
                              -4.455359698088187659e-01 ,
                              -3.351144822320292205e-01])

residualsMatrix = np.zeros((df.shape[0],7))
errorsMatrix = np.zeros((df.shape[0],7))
errorsMatrix1st = np.zeros((df.shape[0],7))
for i in range(df.shape[0]):
    residualsMatrix[i,:] = np.array(df.orbitalResiduals[i][1:-1].split(),dtype=float)
    errorsMatrix[i,:] = np.array( df.energyEigenvalues[i][1:-1].split(),dtype=float) - referenceEnergies
    try:
        errorsMatrix1st[i,:] = np.array( df.energyErrorsWRTfirstSCF[i][1:-1].split(),dtype=float) 
    except AttributeError:
        pass

df['residual0'] = residualsMatrix[:,0]
df['residual1'] = residualsMatrix[:,1]
df['residual2'] = residualsMatrix[:,2]
df['residual3'] = residualsMatrix[:,3]
df['residual4'] = residualsMatrix[:,4]
df['residual5'] = residualsMatrix[:,5]
df['residual6'] = residualsMatrix[:,6]
##df['residual7'] = residualsMatrix[:,7]
##df['residual8'] = residualsMatrix[:,8]
##df['residual9'] = residualsMatrix[:,9]

df['errors0'] = np.abs(errorsMatrix[:,0])
df['errors1'] = np.abs(errorsMatrix[:,1])
df['errors2'] = np.abs(errorsMatrix[:,2])
df['errors3'] = np.abs(errorsMatrix[:,3])
df['errors4'] = np.abs(errorsMatrix[:,4])
df['errors5'] = np.abs(errorsMatrix[:,5])
df['errors6'] = np.abs(errorsMatrix[:,6])


try:
    df['1stSCFerrors0'] = np.abs(errorsMatrix1st[:,0])
    df['1stSCFerrors1'] = np.abs(errorsMatrix1st[:,1])
    df['1stSCFerrors2'] = np.abs(errorsMatrix1st[:,2])
    df['1stSCFerrors3'] = np.abs(errorsMatrix1st[:,3])
    df['1stSCFerrors4'] = np.abs(errorsMatrix1st[:,4])
    df['1stSCFerrors5'] = np.abs(errorsMatrix1st[:,5])
    df['1stSCFerrors6'] = np.abs(errorsMatrix1st[:,6])
except AttributeError:
    pass

def plotFirstSCF(df):

    
    f0, (ax0, ax1) = plt.subplots(1,2, figsize=(12,6))
    df.plot(y='residual0',ax=ax0,logy=True,label='Phi0')
    df.plot(y='residual1',ax=ax0,logy=True,label='Phi1')
    df.plot(y='residual2',ax=ax0,logy=True,label='Phi2')
    df.plot(y='residual3',ax=ax0,logy=True,label='Phi3')
    df.plot(y='residual4',ax=ax0,logy=True,label='Phi4')
    df.plot(y='residual5',ax=ax0,logy=True,label='Phi5')
    df.plot(y='residual6',ax=ax0,logy=True,label='Phi6')
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
    df.plot(y='errors5',ax=ax1,logy=True,label='Phi5')
    df.plot(y='errors6',ax=ax1,logy=True,label='Phi6')
##    df.plot(y='errors7',ax=ax1,logy=True,label='Phi7')
##    df.plot(y='errors8',ax=ax1,logy=True,label='Phi8')
##    df.plot(y='errors9',ax=ax1,logy=True,label='Phi9')
    ax1.set_xlabel('Iteration Number')
    ax1.set_ylabel('Energy Error (Hartree)')
    ax1.set_title('Orbital Errors')

##    plt.suptitle('Using Singularity Skipping, LW3-800')
    plt.suptitle('Using Singularity Subtraction, LW3-800, minDepth 3')
    plt.suptitle(file)

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


def plotFirstSCF_2(df):
    ### Alternative where the x axis is iterations.  Need to use only data where iteration != 0
##    backup = np.copy(df)
    
    f0, (ax0, ax1) = plt.subplots(1,2, figsize=(12,6))
##    df6 = df.loc[df.residual6!=1.0]
##    df6.plot(x='Iteration',y='residual6',ax=ax0,logy=True,label='Phi6')
##    df6.plot(x='Iteration',y='errors6',ax=ax1,logy=True,label='Phi6')
##    
##    df5 = df.loc[df.residual5!=1.0]
##    df5.plot(x='Iteration',y='residual5',ax=ax0,logy=True,label='Phi5')
##    df5.plot(x='Iteration',y='errors5',ax=ax1,logy=True,label='Phi5')
##
##    df4 = df.loc[df.residual4!=1.0]
##    df4.plot(x='Iteration',y='residual4',ax=ax0,logy=True,label='Phi4')
##    df4.plot(x='Iteration',y='errors4',ax=ax1,logy=True,label='Phi4')
##
##    df3 = df.loc[df.residual3!=1.0]
##    df3.plot(x='Iteration',y='residual3',ax=ax0,logy=True,label='Phi3')
##    df3.plot(x='Iteration',y='errors3',ax=ax1,logy=True,label='Phi3')

    df2 = df.loc[df.residual2!=1.0]
    df2.plot(x='Iteration',y='residual2',ax=ax0,logy=True,label='Phi2')
    df2.plot(x='Iteration',y='errors2',ax=ax1,logy=True,label='Phi2')

    df1 = df.loc[df.residual1!=1.0]
    df1.plot(x='Iteration',y='residual1',ax=ax0,logy=True,label='Phi1')
    df1.plot(x='Iteration',y='errors1',ax=ax1,logy=True,label='Phi1')


    df0 = df.loc[df.residual0!=1.0].loc[df.residual1!=1.0].loc[df.residual2!=1.0]
    df0.plot(x='Iteration',y='residual0',ax=ax0,logy=True,label='Phi0')
    df0.plot(x='Iteration',y='errors0',ax=ax1,logy=True,label='Phi0')


    ax0.set_xlabel('Iteration Number')
    ax0.set_ylabel('Residual L2 Norm')
    ax0.set_title('Orbital Residuals')


    ax1.set_xlabel('Iteration Number')
    ax1.set_ylabel('Energy Error (Hartree)')
    ax1.set_title('Orbital Errors')

    plt.show()

    

if __name__=="__main__":
    plotFirstSCF(df)
    


