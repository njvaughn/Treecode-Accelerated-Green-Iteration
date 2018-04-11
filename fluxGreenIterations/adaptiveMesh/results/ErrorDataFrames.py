'''
Created on Mar 29, 2018

@author: nathanvaughn
'''

import pandas as pd
import matplotlib.pyplot as plt
import os

currentDir = os.getcwd()
plotsDir = currentDir+'/plots/'

df = pd.read_csv('accuracyResults.csv', 
                 names=['domainSize', 'minDepth', 'maxDepth', 
                        'numberOfGridpoints', 'testFunction', 'refinementTolerance', 'residualTolerance',
                        'energyErrorGS','psiL2ErrorGS','psiLinfErrorGS',
                        'energyErrorFES','psiL2ErrorFES','psiLinfErrorFES'])

print(df.tail(5))

df.astype({'domainSize':float})
df.astype({'minDepth':int})
df.astype({'numberOfGridpoints':int})
df.astype({'testFunction':str})
df.astype({'refinementTolerance':float})
df.astype({'residualTolerance':float})
df.astype({'energyErrorGS':float})
df.astype({'psiL2ErrorGS':float})
df.astype({'psiLinfErrorGS':float})
df.astype({'energyErrorFES':float})
df.astype({'psiL2ErrorFES':float})
df.astype({'psiLinfErrorFES':float})

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
