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
resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/BerylliumIterationResults/'

#### H2
##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/H2IterationResults/'


#### Oxygen
##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenIterationResults/'

df = pd.read_csv(resultsDir+file, header=0)


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



if resultsDir == '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenIterationResults/':
    TotalEnergy = -7.4469012607372008e+01
    ExchangeEnergy = -7.2193424444124350e+00
    CorrelationEnergy = -5.4455323198374961e-01
    BandEnergy = -4.0613156367497737e+01


if resultsDir == '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/H2IterationResults/':
    TotalEnergy = -1.1376691191341821e+00
    ExchangeEnergy = -5.5876966592456134e-01
    CorrelationEnergy = -9.4268448521496129e-02
    BandEnergy = -7.5499497178953057e-01


df['BandEnergyError'] = abs( df['BandEnergy'] - BandEnergy)
df['ExchangeEnergyError'] = abs( df['ExchangeEnergy'] - ExchangeEnergy)
df['CorrelationEnergyError'] = abs( df['CorrelationEnergy'] - CorrelationEnergy)
df['TotalEnergyError'] = abs( df['TotalEnergy'] - TotalEnergy)




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
        group['logA'] = np.log10(np.abs(group[A]))
        group.plot(x=B, y='logA', style='o', ax=ax, label='%s = %.2f'%(C,name))
##        group.plot(x=B, y='logA', style='o', ax=ax, label='%s = %s'%(C,name))
    plt.legend(loc = 'best')

    if save == True:
        saveID = 'log'+A+'Vs'+B+'ColoredBy'+C
        plt.savefig(plotsDir+saveID+'.pdf', bbox_inches='tight',format='pdf')
    plt.show()

def logAversusLogBcolorbyC(df,A,B,C,save=False):
    fig, ax = plt.subplots(figsize=(8,6))
    fig.suptitle('Log %s versus Log %s colored by %s' %(A,B,C))
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
##    fig.suptitle("Oxygen Atom: Energy Errors")
    fig.suptitle("Beryllium Atom: Energy Errors")
##    fig.suptitle("Hydrogen Molecule: Energy Errors")
    df.plot(x='numberOfPoints', y='BandEnergyError', style='o', ax=ax, loglog=True)
    df.plot(x='numberOfPoints', y='ExchangeEnergyError', style='o', ax=ax, loglog=True)
    df.plot(x='numberOfPoints', y='CorrelationEnergyError', style='o', ax=ax, loglog=True)
    df.plot(x='numberOfPoints', y='TotalEnergyError', style='o', ax=ax, loglog=True)
  
    plt.legend(loc = 'best')
    plt.xlabel('Number of Gridpoints')
    plt.ylabel('Energy Error (Hartree)')

    plt.show()


if __name__=="__main__":
    energyErrors()
