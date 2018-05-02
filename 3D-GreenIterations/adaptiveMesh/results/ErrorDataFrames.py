'''
Created on Mar 29, 2018

@author: nathanvaughn
'''

import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import os
import numpy as np

resultsDir = '/Users/nathanvaughn/Desktop/LevineWilkins'
##df = pd.read_csv(resultsDir+'/accuracyResults_GI_and_energyComp.csv', 
##                 names=['domainSize', 'minDepth', 'maxDepth', 
##                        'numCells', 'numGridpoints', 'LW-Order', 'N_elements',
##                        'residualTolerance', 'energyErrorGS_analyticPsi',
##                        'energyErrorGS','psiL2ErrorGS','psiLinfErrorGS',
##                        'energyErrorFES_analyticPsi','energyErrorFES','psiL2ErrorFES','psiLinfErrorFES'])
##
##df.astype({'domainSize':float})
##df.astype({'minDepth':int})
##df.astype({'numCells':int})
##df.astype({'numGridpoints':int})
##df.astype({'LW-Order':str})
##df.astype({'N_elements':int})
##df.astype({'residualTolerance':float})
##df.astype({'energyErrorGS_analyticPsi':float})
##df.astype({'energyErrorGS':float})
##df.astype({'psiL2ErrorGS':float})
##df.astype({'psiLinfErrorGS':float})
##df.astype({'energyErrorFES_analyticPsi':float})
##df.astype({'energyErrorFES':float})
##df.astype({'psiL2ErrorFES':float})
##df.astype({'psiLinfErrorFES':float})

##resultsDir = '/home/njvaughn/results'
##currentDir = os.getcwd()
plotsDir = resultsDir+'/plots/'
## accuracyResults_psiGSonly
## accuracyResults_psiVpsi
##df = pd.read_csv(resultsDir+'/accuracyResults_variety.csv', 
##                 names=['domainSize', 'minDepth', 'maxDepth', 
##                        'N', 'testFunction1', 'refinementTol1',
##                        'testFunction2', 'refinementTol2', 'residualTolerance',
##                        'energyErrorGS','psiL2ErrorGS','psiLinfErrorGS',
##                        'energyErrorFES','psiL2ErrorFES','psiLinfErrorFES'])

##df = pd.read_csv(resultsDir+'/accuracyResults_variety.csv', 
##                 names=['domainSize', 'minDepth', 'maxDepth', 
##                        'numberOfGridpoints', 'testFunction', 'refinementTolerance', 'residualTolerance',
##                        'energyErrorGS','psiL2ErrorGS','psiLinfErrorGS',
##                        'energyErrorFES','psiL2ErrorFES','psiLinfErrorFES'])

df = pd.read_csv(resultsDir+'/LW12withSmoothing.csv', 
                 header=0)

df = df.drop(12) # 12th entry is redudant, with an error in one value, so drop it.
df.set_value(11,'energyErrorGS_analyticPsi',-0.000445341) # energyErrorGS_analyticPsi is incorect for entry 11 due to typo.  It should be -0.000445341


##df.astype({'domainSize':float})
##df.astype({'minDepth':int})
##df.astype({'numberOfGridpoints':int})
##df.astype({'testFunction':str})
##df.astype({'refinementTolerance':float})
####df.astype({'testFunction2':str})
####df.astype({'refinementTol2':float})
##df.astype({'residualTolerance':float})
##df.astype({'energyErrorGS':float})
##df.astype({'psiL2ErrorGS':float})
##df.astype({'psiLinfErrorGS':float})
##df.astype({'energyErrorFES':float})
##df.astype({'psiL2ErrorFES':float})
##df.astype({'psiLinfErrorFES':float})

##df.sort_values(by='numberOfGridpoints')
##print(df.sort_values(by='numberOfGridpoints'))



def AversusB(df,A,B,save=False):
    fig, ax = plt.subplots(figsize=(8,6))
    fig.suptitle('%s versus %s' %(A,B))
    df.plot(x=B, y=A, style='o',ax=ax)
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

def logAversusLogBcolorbyC(df,A,B,C,trendline=False,save=False):
    fig, ax = plt.subplots(figsize=(8,6))
    fig.suptitle('Log %s versus Log %s colored by %s' %(A,B,C))
    grouped = df.groupby(C)
    for name,group in grouped:
        group['logA'] = np.log10(np.abs(group[A]))
        group['logB'] = np.log10(np.abs(group[B]))
        if trendline==True:
            z = np.polyfit(x=group['logB'], y=group['logA'], deg=1)
            p = np.poly1d(z)
            group['trendline'] = p(group['logB'])
            group['trendline'].plot(ax=ax)
        if isinstance(name,str):
##            group.plot(x='logB', y='logA', style='o', ax=ax, label='%s = %s'%(C,name))
            group.plot(x=B, y=A, style='o', ax=ax, loglog=True,label='%s = %s'%(C,name))
        elif isinstance(name,float):
            group.plot(x='logB', y='logA', style='o', ax=ax, label='%s = %f'%(C,name))
        elif isinstance(name,int):
            group.plot(x='logB', y='logA', style='o', ax=ax, label='%s = %i'%(C,name))
        
    plt.legend(loc = 'best')

    if save == True:
        saveID = 'log'+A+'VsLog'+B+'ColoredBy'+C
        plt.savefig(plotsDir+saveID+'.pdf', bbox_inches='tight',format='pdf')
    plt.show()

def logAandBversusLogCcolorbyD(df,A,B,C,D,save=False):

    ## EXAMPLE ##
    '''
    logAandBversusLogCcolorbyD(df,'energyErrorGS','energyErrorGS_analyticPsi','numberOfGridpoints','divideCriterion')
    '''
    fig, ax = plt.subplots(figsize=(8,6))
##    fig.suptitle('Log %s and Log %s versus Log %s colored by %s' %(A,B,C,D))
    fig.suptitle('Ground State Energy Errors: epsilon = 1.0*volume**(1/3)')
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
    plt.xlabel('number of gridpoints')
    plt.ylabel('energy error (H)')
    plt.yticks([1e-4,5e-4,1e-3,5e-3,1e-2],['1e-4','5e-4','1e-3','5e-3','1e-2'])
    plt.xticks([5e4, 1e5,2e5, 5e5],['5e4', '1e5','2e5', '5e5'])

    if save == True:
        saveID = 'log'+A+'andLog'+B+'VsLog'+C+'ColoredBy'+C
        plt.savefig(plotsDir+saveID+'.pdf', bbox_inches='tight',format='pdf')
    plt.show()




    
##def AversusBwithCequalD(df,A,B,C,D):
    

##print('\nBest ground state energy: ')
##print(df.loc[df.energyErrorGS.abs().idxmin()])
##
##
##print('\nBest excited state energy: ')
##print(df.loc[df.energyErrorFES.abs().idxmin()])

##groupedByMinDepth = df.groupby('minDepth')
##print()
##for name,group in groupedByMinDepth:
##    print('='*70)
##    print('minDepth = ', name)
##    print(group)
##    print('='*70)
##    print()

##groupedByRefinementTolerance = df.groupby('refinementTolerance')
##print()
##for name,group in groupedByRefinementTolerance:
##    print('='*70)
##    print('RefinementTolerance = ', name)
##    print(group.sort_values(by='numberOfGridpoints'))
##    print('='*70)
##    print()
##    # noteworthy -- row 14 is much better than row 5 despite using many fewer gridpoints
