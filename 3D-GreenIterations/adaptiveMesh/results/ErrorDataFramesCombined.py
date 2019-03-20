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



# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Be_combined/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Ox_combined/'
resultsDir = '/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/ParentIntegral_vs_LW5/'


df1 = pd.read_csv(resultsDir+'Oxygen_PC.csv', header=0)
df2 = pd.read_csv(resultsDir+'Oxygen_Biros_and_LW.csv', header=0)

# df1 = pd.read_csv(resultsDir+'Be_PC.csv', header=0)
# df2 = pd.read_csv(resultsDir+'Be_LW.csv', header=0)


df2 = df2[df2['divideCriterion'].isin(['LW5'])]



 
# # Beryllium
# TotalEnergy = -1.4446201118081863e+01
# ExchangeEnergy = -2.2903921833555341e+00
# CorrelationEnergy = -2.2343205529440757e-01
# BandEnergy = -8.1232305760491457e+00
# KineticEnergy =  1.4309060170370618e+01 
# ElectrostaticEnergy = -8.1232305760491457e+00
# HartreeEnergy = 7.115165052  


# Oxygen
TotalEnergy = -7.4469337501098821e+01  # Oxygen Atom
ExchangeEnergy = -7.2193700828939980e+00
CorrelationEnergy = -5.4455323568788838e-01
BandEnergy = -4.0613397710076626e+01
KineticEnergy =  7.4112730191157425e+01
ElectrostaticEnergy = -1.4081814437367436e+02
HartreeEnergy = 36.32506036 





df1['absBandEnergyError'] = abs( df1['BandEnergy'] - BandEnergy)
df1['absExchangeEnergyError'] = abs( df1['ExchangeEnergy'] - ExchangeEnergy)
try: 
    df1['absHartreeEnergyError'] = abs( df1['HartreeEnergy'] - HartreeEnergy)
except Exception as e:
    print(e, ' not present.')
df1['absCorrelationEnergyError'] = abs( df1['CorrelationEnergy'] - CorrelationEnergy)
df1['absTotalEnergyError'] = abs( df1['TotalEnergy'] - TotalEnergy)
df1['absKineticEnergyError'] = abs( df1['KineticEnergy'] - KineticEnergy)
try: 
    df1['absElectrostaticEnergyError'] = abs( df1['ElectrostaticEnergy'] - ElectrostaticEnergy)
except Exception as e:
    print(e, ' not present.')
    
df1['BandEnergyError'] = ( df1['BandEnergy'] - BandEnergy)
df1['ExchangeEnergyError'] = ( df1['ExchangeEnergy'] - ExchangeEnergy)
try: 
    df1['HartreeEnergyError'] = ( df1['HartreeEnergy'] - HartreeEnergy)
except Exception as e:
    print(e, ' not present.')
df1['CorrelationEnergyError'] = ( df1['CorrelationEnergy'] - CorrelationEnergy)
df1['TotalEnergyError'] = ( df1['TotalEnergy'] - TotalEnergy)
df1['KineticEnergyError'] = ( df1['KineticEnergy'] - KineticEnergy)
try: 
    df1['ElectrostaticEnergyError'] = ( df1['ElectrostaticEnergy'] - ElectrostaticEnergy)
except Exception as e:
    print(e, ' not present.')
    



df2['absBandEnergyError'] = abs( df2['BandEnergy'] - BandEnergy)
df2['absExchangeEnergyError'] = abs( df2['ExchangeEnergy'] - ExchangeEnergy)
try: 
    df2['absHartreeEnergyError'] = abs( df2['HartreeEnergy'] - HartreeEnergy)
except Exception as e:
    print(e, ' not present.')
df2['absCorrelationEnergyError'] = abs( df2['CorrelationEnergy'] - CorrelationEnergy)
df2['absTotalEnergyError'] = abs( df2['TotalEnergy'] - TotalEnergy)
df2['absKineticEnergyError'] = abs( df2['KineticEnergy'] - KineticEnergy)
try: 
    df2['absElectrostaticEnergyError'] = abs( df2['ElectrostaticEnergy'] - ElectrostaticEnergy)
except Exception as e:
    print(e, ' not present.')
    
df2['BandEnergyError'] = ( df2['BandEnergy'] - BandEnergy)
df2['ExchangeEnergyError'] = ( df2['ExchangeEnergy'] - ExchangeEnergy)
try: 
    df2['HartreeEnergyError'] = ( df2['HartreeEnergy'] - HartreeEnergy)
except Exception as e:
    print(e, ' not present.')
df2['CorrelationEnergyError'] = ( df2['CorrelationEnergy'] - CorrelationEnergy)
df2['TotalEnergyError'] = ( df2['TotalEnergy'] - TotalEnergy)
df2['KineticEnergyError'] = ( df2['KineticEnergy'] - KineticEnergy)
try: 
    df2['ElectrostaticEnergyError'] = ( df2['ElectrostaticEnergy'] - ElectrostaticEnergy)
except Exception as e:
    print(e, ' not present.')

# print(df)

    
    


def AversusBcolorbyC(df1,df2,A,B,C,save=False):
    fig, ax = plt.subplots(figsize=(8,6))
    fig.suptitle('%s versus %s colored by %s' %(A,B,C))
    grouped1 = df1.groupby(C)
    grouped2 = df2.groupby(C)
    for name,group in grouped1:
        group.plot(x=B, y=A, style='o', ax=ax, label='%s = %s'%(C,name))
    for name,group in grouped2:
        group.plot(x=B, y=A, style='o', ax=ax, label='%s = %s'%(C,name))
    plt.legend(loc = 'best')

    if save == True:
        saveID = A+'Vs'+B+'ColoredBy'+C
        plt.savefig(plotsDir+saveID+'.pdf', bbox_inches='tight',format='pdf')
    plt.show()

def logAversusBcolorbyC(df1,df2A,B,C,save=False):
    fig, ax = plt.subplots(figsize=(8,6))
    fig.suptitle('Log %s versus %s colored by %s' %(A,B,C))
    grouped1 = df1.groupby(C)
    grouped2 = df2.groupby(C)
    for name,group in grouped1:
#         group['logA'] = np.log10(np.abs(group[A]))
        group.plot(x=B, y=A, logy=True, style='o', ax=ax, label='%s = %.2f'%(C,name))
##        group.plot(x=B, y='logA', style='o', ax=ax, label='%s = %s'%(C,name))
    plt.legend(loc = 'best')
    plt.xlabel(B)
    plt.ylabel(A)
#     plt.ylim([1e-3,1e-2])
    plt.grid()

    if save == True:
        saveID = 'log'+A+'Vs'+B+'ColoredBy'+C
        plt.savefig(plotsDir+saveID+'.pdf', bbox_inches='tight',format='pdf')
    plt.show()

def logAversusLogBcolorbyC(df1,df2, A,B,C,save=False):
    fig, ax = plt.subplots(figsize=(8,6))
    fig.suptitle('%s versus %s colored by %s' %(A,B,C))
    grouped1 = df1.groupby(C)
    grouped2 = df2.groupby(C)
    for name,group in grouped1:
##        group['logA'] = np.log10(np.abs(group[A]))
##        group['logB'] = np.log10(np.abs(group[B]))
        if isinstance(name,str):
##            group.plot(x='logB', y='logA', style='o', ax=ax, label='%s = %s'%(C,name))
            group.plot(x=B, y=A, style='o', ax=ax, loglog=True,label='%s = %s'%(C,name))
        elif isinstance(name,float):
            group.plot(x=B, y=A, style='o', ax=ax, loglog=True,label='%s = %f'%(C,name))
        elif isinstance(name,int):
            group.plot(x=B, y=A, style='o', ax=ax, loglog=True,label='%s = %i'%(C,name))
            
    for name,group in grouped2:
##        group['logA'] = np.log10(np.abs(group[A]))
##        group['logB'] = np.log10(np.abs(group[B]))
        if isinstance(name,str):
##            group.plot(x='logB', y='logA', style='o', ax=ax, label='%s = %s'%(C,name))
            group.plot(x=B, y=A, style='o', ax=ax, loglog=True,label='%s = %s'%(C,name))
        elif isinstance(name,float):
            group.plot(x=B, y=A, style='o', ax=ax, loglog=True,label='%s = %f'%(C,name))
        elif isinstance(name,int):
            group.plot(x=B, y=A, style='o', ax=ax, loglog=True,label='%s = %i'%(C,name))
        
    plt.legend(loc = 'best')
    plt.xlabel(B)
    plt.ylabel(A)
#     plt.xlim([1e5,2e6])
#     plt.ylim([1e-5,1e-2])
    plt.grid()
    
    if save == True:
        saveID = 'log'+A+'VsLog'+B+'ColoredBy'+C
        plt.savefig(plotsDir+saveID+'.pdf', bbox_inches='tight',format='pdf')
    plt.show()




 
  
if __name__=="__main__":   
    
#     df = df.loc[df['additionalDepthAtAtoms']==3]
    df1 = df1.loc[df1['order']==5]
    df2 = df2.loc[df2['order']==5]
    df2 = df2.loc[df2['gradientFree']==True]
    
    logAversusLogBcolorbyC(df1,df2,'absTotalEnergyError', 'numberOfPoints', 'divideCriterion')
    logAversusLogBcolorbyC(df1,df2,'absBandEnergyError', 'numberOfPoints', 'divideCriterion')
#     logAversusLogBcolorbyC(df1,df2,'absHartreeEnergyError', 'numberOfPoints', 'divideCriterion')
#     
#     AversusBcolorbyC(df1,df2,'TotalEnergyError', 'numberOfPoints', 'divideCriterion')
#     AversusBcolorbyC(df1,df2,'BandEnergyError', 'numberOfPoints', 'divideCriterion') 
#     AversusBcolorbyC(df1,df2,'HartreeEnergyError', 'numberOfPoints', 'divideCriterion')
