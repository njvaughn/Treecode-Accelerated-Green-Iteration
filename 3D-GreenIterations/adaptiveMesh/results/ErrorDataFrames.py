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


#### Lithium
##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/LithiumIterationResults/'

#### Beryllium
# resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/BerylliumIterationResults/'
# resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/Be_gradientFree/Be_gradientFree/'

#### H2
##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/H2IterationResults/'


#### Oxygen
##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenIterationResults/'
# resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenResults/'
# resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/O_firstSCF_gradientFree/'
# resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/oxygen_with_anderson/'

# resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/O_gradientFree/'


#### Biros Meshes
# resultsDir = '/Users/nathanvaughn/Desktop/meshTests/LWvsBiros/Beryllium/'
# resultsDir = '/Users/nathanvaughn/Desktop/meshTests/LWvsBiros/Oxygen/'
# resultsDir = '/Users/nathanvaughn/Desktop/meshTests/LWvsBiros/OxygenDepthTest/'
# resultsDir = '/Users/nathanvaughn/Desktop/meshTests/LWvsBiros/OxygenGaussianAlphaTest/'
# resultsDir = '/Users/nathanvaughn/Desktop/meshTests/LWvsBiros/OxygenGaviniRef/'
# resultsDir = '/Users/nathanvaughn/Desktop/meshTests/LWvsBiros/mergedOxygen/'

#### Krasny Mesh
# resultsDir='/Users/nathanvaughn/Desktop/krasnyMeshTest/Oxygen/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Oxygen_Hartree/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Oxygen_4param_beforeSCF_smoothedVextHarrison/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Oxygen_4param_beforeSCF/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Oxygen_LW5_smoothedVext/'

# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Oxygen_LW5_beforeSCF_smoothingEpsTests/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Oxygen_4param_beforeSCF_smoothingEpsTests/'

# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Oxygen_psiVextVariation_preSCF/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Oxygen_psiVextVariation/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Oxygen_Hartree/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Oxygen_noSphHarmonics_preSCF/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Oxygen_densityIntegral4th/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Oxygen_integralSqrtDensity/' 


# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Oxygen_VextVariation/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Oxygen_VextRegularized/'


# Beryllium Mesh Tests
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Beryllium_VextVariation/'

# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Beryllium_baseScaling_VextVariation/'
# resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/Be_gradientFree/Be_gradientFree/'
resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Beryllium_LW5/'


# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Beryllium_extrapolation_VextRegularization/'


#  Meshes designed with distribution plots
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Beryllium_density_roottwothirds/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Beryllium_root_45/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Beryllium_root_45_additional3/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Beryllium_nathan/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Oxygen_nathan2/'

# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Be_krasny_density3/'

# Meshes generated with mesh density functions
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Be_krasny_density3/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Be_krasny_density4/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Oxygen_krasny_density3/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Ox_krasny_density4/'


# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Be_combined/'

## Treecode accuracy tests
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/CO_treecode_testing/'
# resultsDir='/Users/nathanvaughn/Desktop/TreecodeTests/KohnShamOxygen/Oxygen/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/CO_treecode_testing/'



## Parent-Child integral tests
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/ParentChildrenIntegral/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/ParentChildrenIntegral_onlyThird/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/ParentChildrenIntegral_onlyThird_oxygen/'

## Benzene
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/benzeneTests/'


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
# TotalEnergy = -1.4446201118081863e+01
# ExchangeEnergy = -2.2903921833555341e+00
# CorrelationEnergy = -2.2343205529440757e-01
# BandEnergy = -8.1232305760491457e+00
# KineticEnergy =  1.4309060170370618e+01
# ElectrostaticEnergy = -8.1232305760491457e+00
# HartreeEnergy = 7.115165052 



df = pd.read_csv(resultsDir+file, header=0)
# print(df)


if resultsDir == '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/LithiumIterationResults/':
    TotalEnergy = -7.3340536782581447
    ExchangeEnergy = -1.4916149721121696
    CorrelationEnergy = -0.15971669832262905
    BandEnergy = -3.8616389456972078


# if resultsDir == '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/BerylliumIterationResults/':
# if resultsDir == '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/Be_gradientFree/Be_gradientFree/':
if resultsDir == '/Users/nathanvaughn/Desktop/meshTests/LWvsBiros/Beryllium/':
    TotalEnergy = -1.4446201118081863e+01
    ExchangeEnergy = -2.2903921833555341e+00
    CorrelationEnergy = -2.2343205529440757e-01
    BandEnergy = -8.1232305760491457e+00
    KineticEnergy =   1.4309060170370618e+01
    ElectrostaticEnergy = -2.6241437049802535e+01
    
#     df.drop(df.index[24], inplace=True)  # 24th row is bad in Beryllium gradient free data




##if resultsDir == '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenIterationResults/':
# if resultsDir == '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenResults/':
# if resultsDir == '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/oxygen_with_anderson/':
# if resultsDir == '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/O_gradientFree/':
if ( (resultsDir == '/Users/nathanvaughn/Desktop/meshTests/LWvsBiros/Oxygen/') or 
     (resultsDir == '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/O_gradientFree/' ) or
     (resultsDir == '/Users/nathanvaughn/Desktop/meshTests/LWvsBiros/mergedOxygen/') or
     (resultsDir == '/Users/nathanvaughn/Desktop/meshTests/LWvsBiros/OxygenDepthTest/')or
     (resultsDir == '/Users/nathanvaughn/Desktop/meshTests/LWvsBiros/OxygenGaviniRef/')or
     (resultsDir == '/Users/nathanvaughn/Desktop/meshTests/LWvsBiros/OxygenGaussianAlphaTest/') ):
# if resultsDir == '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/O_firstSCF_gradientFree/':

    TotalEnergy = -7.4469337501098821e+01
    ExchangeEnergy = -7.2193700828939980e+00
    CorrelationEnergy = -5.4455323568788838e-01
    BandEnergy = -4.0613397710076626e+01
    KineticEnergy =  7.4112730191157425e+01
    ElectrostaticEnergy = -1.4081814437367436e+02


if resultsDir == '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/H2IterationResults/':
    TotalEnergy = -1.1376691191341821e+00
    ExchangeEnergy = -5.5876966592456134e-01
    CorrelationEnergy = -9.4268448521496129e-02
    BandEnergy = -7.5499497178953057e-01


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

# print(df)

def energyAndHOMO():
    nwchemEnergy  = -1.1372499
    nwchemHOMO = -0.378649

##    dftfeEnergy   = -1.1376237062839634 # at T=500
    dftfeEnergy =  -1.1394876804557477 # at T=1e-3
##    dftfeBandgap  = -0.75485764369701525

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6))


    grouped = df.groupby('divideCriterion')
    for name,group in grouped:
##        group.plot(x=B, y=A, style='o', ax=ax, label='%s = %s'%(C,name))
        group.plot(x='numberOfCells', y='computedE', style='o', ax=ax1,label='%s'%name)
        group.plot(x='numberOfCells', y='computedHOMO', style='o', ax=ax2,label='%s'%name)


##    df.plot(x='numberOfCells', y='computedE', style='o', ax=ax1)
    ax1.axhline(y=dftfeEnergy,color='r',label='dft-fe')
    ax1.axhline(y=nwchemEnergy,color='g',label='nwchem')
    ax1.set_title('Total Energy')
    ax1.set_ylabel('Energy (H)')
    ax1.set_ylim([-1.15,-1.135])
    ax1.legend()

##    df.plot(x='numberOfCells', y='Bandgap', style='o', ax=ax2)
##    ax2.axhline(y=dftfeBandgap,color='r',label='dft-fe')
    ax2.axhline(y=nwchemHOMO,color='g',label='nwchem')
    ax2.set_title('HOMO Energy')
    ax2.set_ylabel('Energy (H)')
    ax2.legend()

##    plt.suptitle('Green Iterations results compared to DFT-FE and NWCHEM')
    plt.tight_layout(pad=1.0)
    plt.show()
    
    

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

def logAversusLogBcolorbyC(df,A,B,C,save=False):
    fig, ax = plt.subplots(figsize=(8,6))
    fig.suptitle('%s versus %s colored by %s' %(A,B,C))
    grouped = df.groupby(C)
    for name,group in grouped:
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
    plt.xlim([1e5,2e6])
    plt.ylim([1e-5,1e-2])
    plt.grid()
    
    if save == True:
        saveID = 'log'+A+'VsLog'+B+'ColoredBy'+C
        plt.savefig(plotsDir+saveID+'.pdf', bbox_inches='tight',format='pdf')
    plt.show()

def loglogEversusNcolorbyOrder(df,A,B,C,title=None,save=False):
    fig, ax = plt.subplots(figsize=(8,6))
##    fig.suptitle('Log %s versus Log %s colored by %s' %(A,B,C))
    if title==None:
        fig.suptitle('Ground State Energy Errors: Clenshaw-Curtis vs. Midpoint')
    else:
        fig.suptitle(title)
    
##    fig.suptitle('Ground State Energy Errors: Levine-Wilkins-1 refinement')
    grouped = df.groupby(C)
    for name,group in grouped:
##        group['logA'] = np.log10(np.abs(group[A]))
##        group['logB'] = np.log10(np.abs(group[B]))
        group['absA'] = np.abs(group[A])
        group['absB'] = np.abs(group[B])
    
        if isinstance(name,str):
##            group.plot(x='logB', y='logA', style='o', ax=ax, label='%s = %s'%(C,name))
            group.plot(x=B, y=A, style='o', ax=ax, loglog=True,label='%s = %s'%(C,name))
        elif isinstance(name,float):
            group.plot(x='absB', y='absA', style='o', loglog=True, ax=ax, label='%s = %f'%(C,name))
        elif isinstance(name,int):
            if name==0:
                group.plot(x='absB', y='absA', style='o', loglog=True, ax=ax, label='Midpoint')
            else:
               group.plot(x='absB', y='absA', style='o', loglog=True, ax=ax, label='CC %s %i'%(C,name))
        
    plt.legend(loc = 'best')
    plt.xlabel('Number of Cells')
    plt.ylabel('Energy Error (Hartree)')
##    plt.yticks([1e-6,1e-5,1e-4,1e-3,1e-2],['1e-6','1e-5','1e-4','1e-3','1e-2'])
##    plt.xticks([2e4, 1e5,2e5, 3e6],['2e4', '1e5','2e5', '3e6'])

    if save != False:
##        saveID = A+'Vs'+B+'ColoredBy'+C
        saveID = save
        plt.savefig(plotsDir+saveID+'.pdf', bbox_inches='tight',format='pdf')
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

def energyErrors():
    fig, ax = plt.subplots(figsize=(8,6))
#    fig.suptitle("Oxygen Atom: Energy Errors")
#     fig.suptitle("Beryllium Atom: Energy Errors")
##    fig.suptitle("Hydrogen Molecule: Energy Errors")
#     df.plot(x='numberOfPoints', y='BandEnergyError', style='o', ax=ax, loglog=True)
#     df.plot(x='numberOfPoints', y='ExchangeEnergyError', style='o', ax=ax, loglog=True)
#     df.plot(x='numberOfPoints', y='CorrelationEnergyError', style='o', ax=ax, loglog=True)
#     df.plot(x='numberOfPoints', y='ElectrostaticEnergyError', style='o', ax=ax, loglog=True)
    df.plot(x='numberOfPoints', y='TotalEnergyError', style='o', ax=ax, loglog=True)
  
    plt.legend(loc = 'best')
    plt.xlabel('Number of Gridpoints')
    plt.ylabel('Energy Error (Hartree)')
    plt.title('Oxygen Atom Energy Errors')

    plt.show()
    
    
def energyErrors_splitByGradientHandling(order=None):
    if order==None:
        df_gradient = df.loc[df['gradientFree']==False]
        df_free = df.loc[df['gradientFree']==True]

    else:
        df5 = df.loc[df['order']==5]
        df_gradient = df5.loc[df5['gradientFree']==False]
        df_free = df5.loc[df5['gradientFree']==True]
    
    print('df_gradient: ', df_gradient.head(5))
    print('df_free: ', df_free.head(5))
    fig, ax1 = plt.subplots(figsize=(8,6))
    fig, ax2 = plt.subplots(figsize=(8,6))
#    fig.suptitle("Oxygen Atom: Energy Errors")
#     fig.suptitle("Beryllium Atom: Energy Errors")
##    fig.suptitle("Hydrogen Molecule: Energy Errors")
    df_gradient.plot(x='numberOfPoints', y='BandEnergyError', style='o', ax=ax1, loglog=True)
    df_gradient.plot(x='numberOfPoints', y='KineticEnergyError', style='o', ax=ax1, loglog=True)
    df_gradient.plot(x='numberOfPoints', y='ExchangeEnergyError', style='o', ax=ax1, loglog=True)
    df_gradient.plot(x='numberOfPoints', y='CorrelationEnergyError', style='o', ax=ax1, loglog=True)
    df_gradient.plot(x='numberOfPoints', y='ElectrostaticEnergyError', style='o', ax=ax1, loglog=True)
    df_gradient.plot(x='numberOfPoints', y='TotalEnergyError', style='o', ax=ax1, loglog=True)
    
    
    df_free.plot(x='numberOfPoints', y='BandEnergyError', style='o', ax=ax2, loglog=True)
    df_free.plot(x='numberOfPoints', y='KineticEnergyError', style='o', ax=ax2, loglog=True)
    df_free.plot(x='numberOfPoints', y='ExchangeEnergyError', style='o', ax=ax2, loglog=True)
    df_free.plot(x='numberOfPoints', y='CorrelationEnergyError', style='o', ax=ax2, loglog=True)
    df_free.plot(x='numberOfPoints', y='ElectrostaticEnergyError', style='o', ax=ax2, loglog=True)
    df_free.plot(x='numberOfPoints', y='TotalEnergyError', style='o', ax=ax2, loglog=True)
  
    ax1.legend(loc = 'best')
    ax1.set_xlabel('Number of Gridpoints')
    ax1.set_ylabel('Energy Error (Hartree)')
    ax2.legend(loc = 'best')
    ax2.set_xlabel('Number of Gridpoints')
    ax2.set_ylabel('Energy Error (Hartree)')
    
    ax1.set_title('Beryllium Errors: With Gradients')
    ax2.set_title('Beryllium Errors: Gradient-Free')

    plt.show()
    
def totalEnergyErrors_splitByGradientHandling():
    
    anderson = False
    if anderson==True:
        resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/oxygen_with_anderson/'
        df_anderson = pd.read_csv(resultsDir+file, header=0)
        df_anderson['TotalEnergyError'] = abs( df_anderson['TotalEnergy'] - TotalEnergy)
    
    df6 = df.loc[df['order']==6]
    df5 = df.loc[df['order']==5]
    df4 = df.loc[df['order']==4]
#     df4_gradient = df4.loc[df4['gradientFree']==False]
#     df4_free = df4.loc[df4['gradientFree']==True]
#     
#     df6_gradient = df6.loc[df6['gradientFree']==False]
#     df6_free = df6.loc[df6['gradientFree']==True]
    
    
    df5_gradient = df5.loc[df5['gradientFree']==False]
    df5_free = df5.loc[df5['gradientFree']==True]
    
    
    fig, ax1 = plt.subplots(figsize=(8,6))
#     fig, ax2 = plt.subplots(figsize=(8,6))
#    fig.suptitle("Oxygen Atom: Energy Errors")
#     fig.suptitle("Beryllium Atom: Energy Errors")
##    fig.suptitle("Hydrogen Molecule: Energy Errors")

#     df4_gradient.plot(x='numberOfPoints', y='TotalEnergyError', style='ro', ax=ax1, loglog=True, label='Order 4')
#     df4_free.plot(x='numberOfPoints', y='TotalEnergyError', style='rx', ax=ax1, loglog=True, label='Order 4, Gradient Free')
    
    df5_gradient.plot(x='numberOfPoints', y='TotalEnergyError', style='bo', ax=ax1, loglog=True, label='Order 5')
    if anderson==True:
        df_anderson.plot(x='numberOfPoints', y='TotalEnergyError', style='go', ax=ax1, loglog=True, label='Order 5, Anderson Mixing')
    df5_free.plot(x='numberOfPoints', y='TotalEnergyError', style='bx', ax=ax1, loglog=True, label='Order 5, Gradient Free')
    
#     df6_gradient.plot(x='numberOfPoints', y='TotalEnergyError', style='go', ax=ax1, loglog=True, label='Order 6')
#     df6_free.plot(x='numberOfPoints', y='TotalEnergyError', style='gx', ax=ax1, loglog=True, label='Order 6, Gradient Free')
    
    

  
    ax1.legend(loc = 'best')
    ax1.set_xlabel('Number of Gridpoints')
    ax1.set_ylabel('Energy Error (Hartree)')
    ax1.set_title('Oxygen Atom: Total Energy Error')
    
# #     df4_gradient.plot(x='numberOfCells', y='TotalEnergyError', style='ro', ax=ax2, loglog=True, label='Order 4')
# #     df4_free.plot(x='numberOfCells', y='TotalEnergyError', style='rx', ax=ax2, loglog=True, label='Order 4, Gradient Free')
#     df5_gradient.plot(x='numberOfCells', y='TotalEnergyError', style='bo', ax=ax2, loglog=True, label='Order 5')
#     df5_free.plot(x='numberOfCells', y='TotalEnergyError', style='bx', ax=ax2, loglog=True, label='Order 5, Gradient Free')
# #     df6_gradient.plot(x='numberOfCells', y='TotalEnergyError', style='go', ax=ax2, loglog=True, label='Order 6')
# #     df6_free.plot(x='numberOfCells', y='TotalEnergyError', style='gx', ax=ax2, loglog=True, label='Order 6, Gradient Free')
# 
#     ax2.legend(loc = 'best')
#     ax2.set_xlabel('Number of Cells')
#     ax2.set_ylabel('Energy Error (Hartree)')
#     ax2.set_title('Oxygen Atom: Total Energy Error') 
    plt.show()
    
    
def extrapolate_Vext_regularization(df,X,Y,degree,plot=True):
    q=np.polyfit(df[X],df[Y],degree)
#     print(q)
    p = np.poly1d(q)
    
    print(p)
    
    print('Extrapolated value to epsilon = 0: ', p(0.0))
    if Y=='TotalEnergy':
        print('Error in extrapolated value:       ', p(0.0)-TotalEnergy)
    elif Y=='BandEnergy':
        print('Error in extrapolated value:       ', p(0.0)-BandEnergy)
    elif Y=='HartreeEnergy':
        print('Error in extrapolated value:       ', p(0.0)-HartreeEnergy)
    
    
    if plot==True:
        fig, ax = plt.subplots(figsize=(8,6))
        fig.suptitle('%s versus %s with Extrapolated Value' %(Y,X))
        df.plot(x=X, y=Y, logy=False, style='o', ax=ax)
        plt.legend(loc = 'best')
        plt.xlabel(X)
        plt.ylabel(Y)
        plt.grid()
        
        plt.plot(0.0, p(0.0), 'ro')
        if Y=='TotalEnergy': plt.axhline(y=TotalEnergy,color='k')
        if Y=='BandEnergy': plt.axhline(y=BandEnergy,color='k')
        if Y=='HartreeEnergy': plt.axhline(y=HartreeEnergy,color='k')
    
        plt.show()
    
    return
 
  
if __name__=="__main__":
    df = df.loc[df['gradientFree']==True]  
    df = df.loc[df['order']==5]  
    
    
#     df = df.loc[df['treecodeOrder'].isin([1,8])]
#     df = df.loc[df['theta'].isin([0.7,1.0])]
#     AversusBcolorbyC(df,'TotalEnergy', 'theta', 'treecodeOrder')
#     AversusBcolorbyC(df,'BandEnergyError', 'theta', 'treecodeOrder') 
#     AversusBcolorbyC(df,'HartreeEnergyError', 'theta', 'treecodeOrder')
#     AversusBcolorbyC(df,'ExchangeEnergyError', 'theta', 'treecodeOrder')
#     AversusBcolorbyC(df,'CorrelationEnergyError', 'theta', 'treecodeOrder')
    
#     df = df.loc[df['additionalDepthAtAtoms']==3]
#     df = df.loc[df['maxDepth']>=13]
#     df = df.loc[df['order']==5]
    logAversusLogBcolorbyC(df,'absTotalEnergyError', 'numberOfPoints', 'maxDepth')
    logAversusLogBcolorbyC(df,'absBandEnergyError', 'numberOfPoints', 'maxDepth')
    logAversusLogBcolorbyC(df,'absHartreeEnergyError', 'numberOfPoints', 'maxDepth')
#     AversusBcolorbyC(df,'TotalEnergyError', 'numberOfPoints', 'maxDepth')
#     AversusBcolorbyC(df,'BandEnergyError', 'numberOfPoints', 'maxDepth') 
#     AversusBcolorbyC(df,'HartreeEnergyError', 'numberOfPoints', 'maxDepth')
#     AversusBcolorbyC(df,'ExchangeEnergyError', 'numberOfPoints', 'maxDepth')
#     AversusBcolorbyC(df,'CorrelationEnergyError', 'numberOfPoints', 'maxDepth')


    
#     df = df.loc[df['divideParameter4']==20000000]   
# #     df = df.loc[df['divideParameter4']==200]  
#     df_extrap = df.loc[df['VextSmoothingEpsilon']>=1e-4] 
# #     df_extrap = df_extrap.loc[df_extrap['VextSmoothingEpsilon']>=2e-3] 
# #     extrapolate_Vext_regularization(df_extrap, 'HartreeEnergy', 'VextSmoothingEpsilon',len(df_extrap.index)-1)
# #     extrapolate_Vext_regularization(df_extrap, 'BandEnergy', 'VextSmoothingEpsilon',len(df_extrap.index)-1)
#     extrapolate_Vext_regularization(df_extrap, 'VextSmoothingEpsilon', 'TotalEnergy', len(df_extrap.index)-1)
# #     logAversusLogBcolorbyC(df_extrap,'absTotalEnergyError', 'VextSmoothingEpsilon', 'maxDepth')
    
    
    
#     logAversusLogBcolorbyC(df,'absElectrostaticEnergyError', 'numberOfPoints', 'divideParameter')
#     AversusBcolorbyC(df,'ElectrostaticEnergyError', 'numberOfPoints', 'divideParameter')
    
#     df = df.loc[df['VextSmoothingEpsilon']==0.0] 
#     df = df.loc[df['maxDepth']==11] 
#     logAversusLogBcolorbyC(df,'absBandEnergyError', 'numberOfPoints', 'divideParameter1')
#     AversusBcolorbyC(df,'BandEnergyError', 'numberOfPoints', 'maxDepth')
#     logAversusLogBcolorbyC(df,'absHartreeEnergyError', 'numberOfPoints', 'divideParameter1')
#     AversusBcolorbyC(df,'HartreeEnergyError', 'numberOfPoints', 'maxDepth')
#     logAversusLogBcolorbyC(df,'absTotalEnergyError', 'numberOfPoints', 'order')
#     AversusBcolorbyC(df,'TotalEnergyError', 'numberOfPoints', 'maxDepth')


    ## Vext Investigations
     
#     df = df.loc[df['divideParameter1']==2.0]  
#     df = df.loc[df['order']==5]  
#     df = df.loc[df['maxDepth']==12]  
#     df = df.loc[df['divideParameter3']<=0.01]  
#     logAversusLogBcolorbyC(df,'TotalEnergyError', 'numberOfPoints', 'divideParameter1')
#     logAversusLogBcolorbyC(df,'TotalEnergyError', 'numberOfPoints', 'order')
#     logAversusLogBcolorbyC(df,'TotalEnergyError', 'numberOfPoints', 'maxDepth')
#     logAversusLogBcolorbyC(df,'TotalEnergyError', 'VextSmoothingEpsilon', 'divideParameter1')

#     df_extrap = df.loc[df['VextSmoothingEpsilon']>=1e-4] 
#     extrapolate_Vext_regularization(df_extrap, 'HartreeEnergy', 'VextSmoothingEpsilon',2)
#     extrapolate_Vext_regularization(df_extrap, 'BandEnergy', 'VextSmoothingEpsilon',2)
#     extrapolate_Vext_regularization(df_extrap, 'TotalEnergy', 'VextSmoothingEpsilon',2)
#     logAversusLogBcolorbyC(df,'absBandEnergyError', 'VextSmoothingEpsilon', 'divideParameter1')
    
    
#     logAversusLogBcolorbyC(df,'absTotalEnergyError', 'VextSmoothingEpsilon', 'divideParameter1')
#     logAversusLogBcolorbyC(df,'absBandEnergyError', 'VextSmoothingEpsilon', 'divideParameter1')
#     logAversusLogBcolorbyC(df,'absHartreeEnergyError', 'VextSmoothingEpsilon', 'divideParameter1')
 

    
#     logAversusBcolorbyC(df,'TotalEnergyError', 'numberOfPoints', 'divideParameter')
#     logAversusBcolorbyC(df,'TotalEnergyError', 'VextSmoothingEpsilon', 'divideParameter1')

#     df = df.loc[df['order']==7]  
#     df = df.loc[df['depthAtAtoms']==2]  
#     df = df.loc[df['divideParameter1']==2]  
#     df = df.loc[df['divideParameter3']==0.1]  
#     df = df.loc[df['divideParameter2']==3]  
#     df = df.loc[df['divideParameter4']==100000]  
#     df = df.loc[df['maxDepth']>=14] 
#     logAversusBcolorbyC(df,'BandEnergyError', 'maxDepth', 'divideParameter1') 

#     logAversusLogBcolorbyC(df,'TotalEnergyError', 'numberOfPoints','maxDepth') 
#     logAversusLogBcolorbyC(df,'TotalEnergyError', 'numberOfPoints', 'order') 
#     logAversusLogBcolorbyC(df,'HartreeEnergyError', 'numberOfPoints', 'divideParameter4') 
#     logAversusLogBcolorbyC(df,'BandEnergyError', 'numberOfPoints', 'maxDepth') 
#     logAversusLogBcolorbyC(df,'CorrelationEnergyError', 'numberOfPoints', 'order') 
#     logAversusLogBcolorbyC(df,'ExchangeEnergyError', 'numberOfPoints', 'order') 
    
#     df = df.loc[df['depthAtAtoms']==2]   
#     print(df['HartreeEnergyError'])   
#     df = df.loc[df['Treecode']==False]    
#     logAversusLogBcolorbyC(df,'TotalEnergyError', 'VextSmoothingEpsilon', 'divideParameter1')
#     logAversusLogBcolorbyC(df,'BandEnergyError', 'VextSmoothingEpsilon', 'depthAtAtoms')
    
#     logAversusLogBcolorbyC(df,'BandEnergyError', 'numberOfPoints', 'depthAtAtoms')
#     logAversusLogBcolorbyC(df,'BandEnergyError', 'numberOfPoints', 'maxDepth')
#     logAversusLogBcolorbyC(df,'HartreeEnergyError', 'numberOfPoints', 'depthAtAtoms')
#     logAversusLogBcolorbyC(df,'ExchangeEnergyError', 'numberOfPoints', 'depthAtAtoms')
#     logAversusLogBcolorbyC(df,'BandEnergyError', 'numberOfPoints', 'depthAtAtoms')
#     logAversusLogBcolorbyC(df,'TotalEnergyError', 'numberOfPoints', 'order')
#     logAversusLogBcolorbyC(df,'TotalEnergyError', 'numberOfPoints', 'depthAtAtoms')   

 
#     df = df.loc[df['gradientFree']==1]
#     df = df.loc[df['maxDepth']>11]
#     logAversusLogBcolorbyC(df,'TotalEnergyError', 'numberOfPoints', 'divideCriterion')
    
    
# #     print( df['numberOfPoints'] )
#     df = df.loc[df['numberOfPoints']==666000]
#     df = df.loc[df['treecodeOrder']>0.0]
#  
#     df['TreecodeError'] = np.abs( df['TotalEnergy'] + 74.4706852 )
# #     print(df['TreecodeError'])
#     logAversusBcolorbyC(df,'TreecodeError', 'theta', 'treecodeOrder')


#     logAversusLogBcolorbyC(df,'TotalEnergyError', 'theta', 'Treecode')
#     logAversusBcolorbyC(df,'TotalEnergyError', 'theta', 'treecodeOrder')

  
#     energyErrors()
#     df = df.loc[df['gradientFree']==1]
#     df = df.loc[df['maxDepth']>11]
#     df = df.loc[df['order'].isin([7,5])]
#     logAversusLogBcolorbyC(df,'TotalEnergyError', 'numberOfPoints', 'divideCriterion')

    
#     df_LW_and_BirosN = df[df['divideCriterion'].isin(['LW5', 'BirosN','BirosG'])]
#     df_LW_and_BirosN = df[df['divideCriterion'].isin(['BirosN','BirosG'])]
#     df_LW_and_BirosN = df[df['divideCriterion'].isin(['LW5', 'BirosN', 'BirosG', 'BirosGN', 'BirosGN2'])]
#     df_LW_and_BirosN = df[df['divideCriterion'].isin(['LW5'])]
#     df_LW_and_BirosN = df[df['divideCriterion'].isin(['BirosGN2'])]
#     df_LW_and_BirosN = df[df['divideCriterion'].isin(['LW5','BirosN', 'BirosGN'])]
#     print(df_LW_and_BirosN)
#     logAversusLogBcolorbyC(df.loc[df['order']==7],'TotalEnergyError', 'numberOfPoints', 'divideParameter')
#     logAversusLogBcolorbyC(df.loc[df['order']==7],'TotalEnergyError', 'numberOfPoints', 'maxDepth')
#     logAversusLogBcolorbyC(df,'TotalEnergyError', 'numberOfPoints', 'divideParameter')
#     logAversusLogBcolorbyC(df.loc[df['gradientFree']==1],'TotalEnergyError', 'numberOfPoints', 'divideCriterion')
#     logAversusLogBcolorbyC(df_LW_and_BirosN,'TotalEnergyError', 'numberOfPoints', 'divideCriterion')
#     logAversusLogBcolorbyC(df_LW_and_BirosN,'TotalEnergyError', 'numberOfPoints', 'maxDepth')
#     logAversusLogBcolorbyC(df_LW_and_BirosN,'TotalEnergyError', 'numberOfPoints', 'order')
#     logAversusLogBcolorbyC(df_LW_and_BirosN,'TotalEnergyError', 'numberOfPoints', 'gaussianAlpha')
#     logAversusBcolorbyC(df_LW_and_BirosN,'TotalEnergyError', 'gaussianAlpha', 'maxDepth')
#     logAversusLogBcolorbyC(df,'TotalEnergyError', 'numberOfPoints', 'gradientFree')
#     logAversusLogBcolorbyC(df_LW_and_BirosN,'TotalEnergyError', 'numberOfPoints', 'divideParameter')

    

#     totalEnergyErrors_splitByGradientHandling()
#     energyErrors_splitByGradientHandling(order=5)
    
    
    ### Plot effect of gradient-free approach for oxygen atom.
    # resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/O_gradientFree/'
#     df = df.loc[df['order']==5]    
#     logAversusLogBcolorbyC(df,'TotalEnergyError', 'numberOfPoints', 'gradientFree')
#     logAversusLogBcolorbyC(df,'BandEnergyError', 'numberOfPoints', 'gradientFree')

