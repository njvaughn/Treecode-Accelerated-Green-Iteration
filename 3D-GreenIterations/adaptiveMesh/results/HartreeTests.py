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
resultsDir = '/Users/nathanvaughn/Desktop/HartreeTests/GaussianSingularitySubtraction/GaussianDensity/'


df = pd.read_csv(resultsDir+file, header=0)
df4 = df[df['order']==4]
df5 = df[df['order']==5]

def hartreeEnergyErrors(dataframe):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))    #fig, ax = plt.subplots(figsize=(8,6))
    f.suptitle("Gaussian Density: Gaussian Singularity Subtraction")
    
    dataframe.plot(x='numberOfPoints', y='EnergyErrorFromAnalytic', style='o', ax=ax1, loglog=True)
    dataframe.plot(x='numberOfPoints', y='EnergyErrorFromNumerical', style='o', ax=ax1, loglog=True)
    dataframe.plot(x='numberOfPoints', y='L2Error', style='o', ax=ax2, loglog=True)
    dataframe.plot(x='numberOfPoints', y='LinfError', style='o', ax=ax2, loglog=True)

    ax1.set_title('Energy Errors')
    ax1.legend(loc = 'lower left')
    ax1.set_xlabel('Number of Gridpoints')
    ax1.set_ylabel('Relative Energy Errors')

    ax2.set_title('Potential Errors')
    ax2.legend(loc = 'lower left')
    ax2.set_xlabel('Number of Gridpoints')
    ax2.set_ylabel('Relative Potential Errors')

    plt.show()

def hartreeEnergyErrors_colorBy(dataframe,B='order'):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))    #fig, ax = plt.subplots(figsize=(8,6))
    f.suptitle("Gaussian Density: Gaussian Singularity Subtraction")

    grouped = dataframe.groupby(B)
    for name,group in grouped:
        group.plot(x='numberOfPoints', y='EnergyErrorFromAnalytic', style='o', ax=ax1, loglog=True, label='From Analytic: %s = %s'%(B,name))
        group.plot(x='numberOfPoints', y='EnergyErrorFromNumerical', style='o', ax=ax1, loglog=True, label='From Numerical: %s = %s'%(B,name))
        group.plot(x='numberOfPoints', y='L2Error', style='o', ax=ax2, loglog=True, label='L2 Error: %s = %s'%(B,name))
        group.plot(x='numberOfPoints', y='LinfError', style='o', ax=ax2, loglog=True, label='Linf Error: %s = %s'%(B,name))
##        group.plot(x='numberOfPoints', y='KineticEnergyError', style='o', ax=ax1, loglog=True, label='%s = %s'%('Kinetic Error: Order',name))
##        group.plot(x='numberOfPoints', y='TotalEnergyError', style='o', ax=ax1, loglog=True, label='%s = %s'%('Total Error: Order',name))
        


    ax1.set_title('Energy Errors')
    ax1.legend(loc = 'lower left')
    ax1.set_xlabel('Number of Gridpoints')
    ax1.set_ylabel('Relative Energy Errors')

    ax2.set_title('Potential Errors')
    ax2.legend(loc = 'lower left')
    ax2.set_xlabel('Number of Gridpoints')
    ax2.set_ylabel('Relative Potential Errors')

    plt.show()




if __name__=="__main__":
##    pass
##    hartreeEnergyErrors(df)
    hartreeEnergyErrors_colorBy(df,B='order')

