'''
Created on Mar 29, 2018

@author: nathanvaughn
'''

import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import os
import numpy as np

plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})        # use LaTeX

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams.update({'font.size': 18})


file='runComparison.csv'


## MICDE SYMPOSIUM DATA
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/MICDE_Data_2019/berylliumData/'
resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/MICDE_Data_2019/CarbonMonoxideData/'


## Temorary plot-testing data
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/CO_treecode_testing/'


# # Oxygen
# TotalEnergy = -7.4469337501098821e+01  # Oxygen Atom
# ExchangeEnergy = -7.2193700828939980e+00
# CorrelationEnergy = -5.4455323568788838e-01
# BandEnergy = -4.0613397710076626e+01
# KineticEnergy =  7.4112730191157425e+01
# ElectrostaticEnergy = -1.4081814437367436e+02
# HartreeEnergy = 36.32506036  


# # Beryllium
TotalEnergy = -1.4446201118081863e+01
ExchangeEnergy = -2.2903921833555341e+00
CorrelationEnergy = -2.2343205529440757e-01
BandEnergy = -8.1232305760491457e+00
KineticEnergy =  1.4309060170370618e+01
ElectrostaticEnergy = -8.1232305760491457e+00
HartreeEnergy = 7.115165052  


# # Carbon Monoxide
# TotalEnergy = -112.47144323799400
# ExchangeEnergy = -1.1997011069615391e+01
# CorrelationEnergy = -9.4214407530225852e-01
# BandEnergy = -6.2898682441673358e+01
# KineticEnergy =  0.0
# ElectrostaticEnergy = 0.0
# HartreeEnergy = 76.1983318 




### For treecode order 9 theta 0.7, 
# Orbital Energies:  [-19.36814367 -10.50463297  -1.68559763  -1.10713083  -1.02023553
#   -1.02023553  -0.89612904]
# Updated V_x:                           -16.0262855269 Hartree
# Updated V_c:                           -1.0514399128 Hartree
# Updated Band Energy:                   -64.2042104245 H, -1.3055279828e+00 H
# Updated E_Hartree:                      76.3638212814 H, 1.6548948140e-01 H
# Updated E_x:                           -12.0197141452 H, -2.2703075574e-02 H
# Updated E_c:                           -0.9422194712 H, -7.5395869255e-05 H
# Total Energy:                          -113.9340565920 H, -1.4626133540e+00 H
# Energy Residual:                        1.039e+02
# Density Residual:                       2.575e-01


## AFTER FIRST SCF, WITH PCI-3e-7 MESH DIRECT SUM 
## Direct sum, mixing parameter 0.5
# Orbital Energies:  [-19.36814634 -10.50463033  -1.68559589  -1.10712967  -1.02023537
#   -1.02023522  -0.89612752]
# Updated V_x:                           -16.0262781591 Hartree
# Updated V_c:                           -1.0514394177 Hartree
# Updated Band Energy:                   -64.2042006799 H, -1.3055182382e+00 H
# Updated E_Hartree:                      76.3637679069 H, 1.6543610687e-01 H
# Updated E_x:                           -12.0197086193 H, -2.2697549701e-02 H
# Updated E_c:                           -0.9422190237 H, -7.4948414415e-05 H
# Total Energy:                          -113.9339953625 H, -1.4625521245e+00 H
# Energy Residual:                        1.039e+02
# Density Residual:                       2.575e-01


TotalEnergy = -113.9339953625
BandEnergy = -64.2042006799
HartreeEnergy = 76.3637679069

df = pd.read_csv(resultsDir+file, header=0)
# print(df)




df['absBandEnergyError'] = abs( df['BandEnergy'] - BandEnergy)
df['absExchangeEnergyError'] = abs( df['ExchangeEnergy'] - ExchangeEnergy)
try: 
    df['absHartreeEnergyError'] = abs( df['HartreeEnergy'] - HartreeEnergy)
except Exception as e:
    print(e, ' not present.')
df['absCorrelationEnergyError'] = abs( df['CorrelationEnergy'] - CorrelationEnergy)
df['absTotalEnergyError'] = abs( df['TotalEnergy'] - TotalEnergy)
df['absKineticEnergyError'] = abs( df['KineticEnergy'] - KineticEnergy)
try: 
    df['absElectrostaticEnergyError'] = abs( df['ElectrostaticEnergy'] - ElectrostaticEnergy)
except Exception as e:
    print(e, ' not present.')
    
df['BandEnergyError'] = ( df['BandEnergy'] - BandEnergy)
df['ExchangeEnergyError'] = ( df['ExchangeEnergy'] - ExchangeEnergy)
try: 
    df['HartreeEnergyError'] = ( df['HartreeEnergy'] - HartreeEnergy)
except Exception as e:
    print(e, ' not present.')
df['CorrelationEnergyError'] = ( df['CorrelationEnergy'] - CorrelationEnergy)
df['TotalEnergyError'] = ( df['TotalEnergy'] - TotalEnergy)
df['KineticEnergyError'] = ( df['KineticEnergy'] - KineticEnergy)
try: 
    df['ElectrostaticEnergyError'] = ( df['ElectrostaticEnergy'] - ElectrostaticEnergy)
except Exception as e:
    print(e, ' not present.')



def AversusB(df,A,B,save=False):
    fig, ax = plt.subplots(figsize=(8,6))
    fig.suptitle('%s versus %s' %(A,B))
    df.plot(x=B, y=A, style='o',ax=ax)

    dftfeEnergy = -1.1376237062839634e+00
    NWchemEnergy = -1.1372499
    plt.axhline(y=dftfeEnergy,color='r')
    plt.axhline(y=NWchemEnergy,color='g')
##    plt.plot(dftfeEnergy*np.ones(100),'r-')
##    plt.plot(NWchemEnergy*np.ones(100),'g-')
    if save == True:
        saveID = A+'Vs'+B
        plt.savefig(plotsDir+saveID+'.pdf', bbox_inches='tight',format='pdf')
    plt.show()

def AversusBcolorbyC(df,A,B,C,save=False):
    fig, ax = plt.subplots(figsize=(8,6))
    fig.suptitle('%s versus %s colored by %s' %(A,B,C))
    grouped = df.groupby(C)
    for name,group in grouped:
##        group.plot(x=B, y=A, style='o', ax=ax, label='%s = %.2f'%(C,name))
        group.plot(x=B, y=A, style='o', ax=ax, label='%s = %s'%(C,name))
    plt.legend(loc = 'best')

    if save == True:
        saveID = A+'Vs'+B+'ColoredBy'+C
        plt.savefig(plotsDir+saveID+'.pdf', bbox_inches='tight',format='pdf')
    plt.show()

def logAversusBcolorbyC(df,A,B,C,save=False):
    fig, ax = plt.subplots(figsize=(8,6))
    fig.suptitle('Log %s versus %s colored by %s' %(A,B,C))
    grouped = df.groupby(C)
    for name,group in grouped:
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

def berylliumMeshRefinement(df,A,B,C,save=False):
    df = df.drop(df.index[14])

#     df.loc[df['divideParameter3'].isin(6e-8)]
    
#     df2 = df.loc[df['divideParameter3']==6e-8]
#     df2 = df2.loc[df2['energyTolerance']==5e-7]
#     df = df.loc[df['divideParameter3'] != 6e-8]
    fig, ax = plt.subplots(figsize=(8,6))
    df.plot(x=B, y='absTotalEnergyError', style='bo', markerSize=8, ax=ax, loglog=True, label='Total Energy Error')
    df.plot(x=B, y='absBandEnergyError', style='ro', markerSize=7, ax=ax, loglog=True, label='Band Energy Error')
    df.plot(x=B, y='absHartreeEnergyError', style='go', markerSize=8, ax=ax, loglog=True, label='Hartree Energy Error')
    
    
#     fig.suptitle('%s versus %s colored by %s' %(A,B,C))
#     grouped = df.groupby(C)
#     for name,group in grouped:
#         group.plot(x=B, y=A, style='bo', markerSize=12, ax=ax, loglog=True, legend=False)
#         if isinstance(name,str):
#             group.plot(x=B, y=A, style='o', ax=ax, loglog=True,label='%s = %s'%(C,name))
#         elif isinstance(name,float):
#             group.plot(x=B, y=A, style='o', ax=ax, loglog=True,label='%s = %1.3e'%(C,name))
#         elif isinstance(name,int):
#             group.plot(x=B, y=A, style='o', ax=ax, loglog=True,label='%s = %i'%(C,name))
    plt.legend() 
    
    
#     df2.plot(x=B, y='absTotalEnergyError', style='bo', markerSize=8, ax=ax, loglog=True)
#     df2.plot(x=B, y='absBandEnergyError', style='ro', markerSize=6, ax=ax, loglog=True,legend=False)
#     df2.plot(x=B, y='absHartreeEnergyError', style='go', markerSize=8, ax=ax, loglog=True,legend=False)
    plt.xlabel('Quadrature Points')
    plt.ylabel('Energy Error (Hartree)')
    
#     plt.xlim([3e5,2e6])
#     plt.ylim([5e-5,1e-2])
#     plt.grid()
    
    if save == True:
        saveID = 'log'+A+'VsLog'+B+'ColoredBy'+C
        plt.savefig(plotsDir+saveID+'.pdf', bbox_inches='tight',format='pdf')
    plt.show()
    
def treecodeAfterFirstSCF(df,A,B,C,save=False):
    

    fig, ax = plt.subplots(figsize=(8,6))

    grouped = df.groupby(C)
    for name,group in grouped:
        if isinstance(name,str):
            group.plot(x=B, y=A, style='o',markerSize=12, ax=ax, logy=True,label='%s = %s'%(C,name))
        elif isinstance(name,float):
            group.plot(x=B, y=A, style='o',markerSize=12, ax=ax, logy=True,label='%s = %1.3e'%(C,name))
        elif isinstance(name,int):
            group.plot(x=B, y=A, style='o',markerSize=12, ax=ax, logy=True,label='Order %i'%(name))
    plt.legend() 
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    plt.xlabel(r'$\theta$')
#     plt.xlabel('Polynomial Order')
    plt.ylabel('Energy Error (Hartree)') 
    
    plt.xlim([0.45, 0.75])
    plt.ylim([1e-6,1e-3])
#     plt.grid()
    
    plt.show()



def logAandBversusLogCcolorbyD(df,A,B,C,D,save=False):

    ## EXAMPLE ##
    '''
    logAandBversusLogCcolorbyD(df,'energyErrorGS','energyErrorGS_analyticPsi','numberOfGridpoints','divideCriterion')
    '''
    fig, ax = plt.subplots(figsize=(8,6))
    fig.suptitle('Log %s and Log %s versus Log %s colored by %s' %(A,B,C,D))
##    fig.suptitle('Ground State Energy Errors: epsilon = 1.0*volume**(1/3)')
    grouped = df.groupby(D)
    counter=0
    for name,group in grouped:
        group['logA'] = np.log10(np.abs(group[A]))
        group['logB'] = np.log10(np.abs(group[B]))
        group['logC'] = np.log10(np.abs(group[C]))
        group['absA'] = np.abs(group[A])
        group['absB'] = np.abs(group[B])
        if counter == 0:
            sty1 = 'bo'
            sty2 = 'b^'
        elif counter == 1:
            sty1 = 'go'
            sty2 = 'g^'
        elif counter == 2:
            sty1 = 'ro'
            sty2 = 'r^'
        elif counter == 3:
            sty1 = 'mo'
            sty2 = 'm^'
        if isinstance(name,str):
##            group.plot(x='logC', y='logA', style=sty1, ax=ax, label='%s: %s = %s'%(A,D,name))
##            group.plot(x='logC', y='logB', style=sty2, ax=ax, label='%s: %s = %s'%(B,D,name))
##            group.plot(x='logC', y='logA', style=sty1, ax=ax, label='GreenIteration: %s'%(name))
##            group.plot(x='logC', y='logB', style=sty2, ax=ax, label='AnalyticPsi:      %s'%(name))
            group.plot(x=C, y='absA', style=sty1, ax=ax, loglog=True,label='GreenIteration: %s'%(name))
            group.plot(x=C, y='absB', style=sty2, ax=ax, loglog=True, label='AnalyticPsi:      %s'%(name))
        elif isinstance(name,float):
            group.plot(x='logC', y='logA', style=sty1, ax=ax, label='%s: %s = %f'%(A,D,name))
            group.plot(x='logC', y='logB', style=sty2, ax=ax, label='%s: %s = %f'%(B,D,name))
        elif isinstance(name,int):
            group.plot(x='logC', y='logA', style=sty1, ax=ax, label='%s: %s = %i'%(A,D,name))
            group.plot(x='logC', y='logB', style=sty2, ax=ax, label='%s: %s = %i'%(B,D,name))
        counter+=1
        
    plt.legend(loc = 'best')
##    plt.xlabel('number of gridpoints')
##    plt.ylabel('energy error (H)')
##    plt.yticks([1e-4,5e-4,1e-3,5e-3,1e-2],['1e-4','5e-4','1e-3','5e-3','1e-2'])
##    plt.xticks([5e4, 1e5,2e5, 5e5],['5e4', '1e5','2e5', '5e5'])

    if save == True:
        saveID = 'log'+A+'andLog'+B+'VsLog'+C+'ColoredBy'+C
        plt.savefig(plotsDir+saveID+'.pdf', bbox_inches='tight',format='pdf')
    plt.show()


if __name__=="__main__":
    print(df)
    df = df.loc[df['gradientFree']==True]  
    df = df.loc[df['order']==5]  
    print(df)
#     df = df[df['divideCriterion'].isin(['LW5'])] 
     
#     df = df.loc[df['treecodeOrder'].isin([1,8])]
#     df = df.loc[df['theta'].isin([0.7,1.0])]

#     logAversusLogBcolorbyC(df,'absTotalEnergyError', 'theta', 'treecodeOrder')
#     logAversusLogBcolorbyC(df,'absBandEnergyError', 'theta', 'treecodeOrder') 
#     logAversusLogBcolorbyC(df,'absHartreeEnergyError', 'theta', 'treecodeOrder')
#     logAversusLogBcolorbyC(df,'absExchangeEnergyError', 'theta', 'treecodeOrder')
#     logAversusLogBcolorbyC(df,'absCorrelationEnergyError', 'theta', 'treecodeOrder')
    
#     df = df.loc[df['additionalDepthAtAtoms']==3]
#     df = df.loc[df['maxDepth']>=13]
#     df = df.loc[df['order']==5]

#     berylliumMeshRefinement(df,'absTotalEnergyError', 'numberOfPoints', 'divideParameter3')

    df = df.loc[df['treecodeOrder']!=1]
    df = df.loc[df['treecodeOrder']!=8]
#     df['timePerIteration'] = df['totalTime']/df['totalIterationCount']
    treecodeAfterFirstSCF(df,'absTotalEnergyError', 'theta', 'treecodeOrder')
#     treecodeAfterFirstSCF(df,'absBandEnergyError', 'theta', 'treecodeOrder')
#     treecodeAfterFirstSCF(df,'absHartreeEnergyError', 'theta', 'treecodeOrder')
#     treecodeAfterFirstSCF(df,'absTotalEnergyError', 'treecodeOrder','theta')
#     treecodeAfterFirstSCF(df,'absTotalEnergyError', 'totalIterationCount', 'theta')
