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
# resultsDir = '/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/KohnShamTests'
plotsDir = resultsDir+'plots/'


##file='CO_LW3_1200_singSub_fixedMixingBug_SCF_.csv'
##file='CO_LW3_800_simpleMixing_SCF_.csv'
##file='CO_LW3_1200_simpleMixing_SCF_.csv'
##file='CO_LW1_2000_simpleMixing_SCF_.csv'
file='CO_LW3_2500_simpleMixing_SCF_.csv'
##file='CO_LW3_1600_simpleMixing_SCF_.csv'
##file='CO_LW3_1000_simpleMixing_SCF_.csv'


df_CO = pd.read_csv(resultsDir+file, header=0)
    
def plotSCFconvergence(df, system = 'H2'):
    
    if system == 'H2':
        dftfeTotalEnergy = -1.1376691191341821e+00
        dftfeExchangeEnergy = -5.5876966592456134e-01
        dftfeCorrelationEnergy = -9.4268448521496129e-02
        dftfeBandEnergy = -7.5499497178953057e-01
    
    if system == "Lithium":
        dftfeTotalEnergy = -7.3340536782581447
        dftfeExchangeEnergy = -1.4916149721121696
        dftfeCorrelationEnergy = -1.5971669832262905e-01
        dftfeBandEnergy = -3.8616389456972078
        
    if system == "carbonMonoxide":
        # these taken from mesh size 0.125 run
        dftfeTotalEnergy = -1.1247167888813128e+02
        dftfeExchangeEnergy = -1.1997052574614749e+01
        dftfeCorrelationEnergy = -9.4214501809750550e-01
        dftfeBandEnergy = -6.2898649220361037e+01


    df['bandEnergyError']=abs(df['bandEnergy']-dftfeBandEnergy)
    df['exchangeEnergyError']=abs(df['exchangeEnergy']-dftfeExchangeEnergy)
    df['correlationEnergyError']=abs(df['correlationEnergy']-dftfeCorrelationEnergy)
    df['totalEnergyErrorPerAtom']=abs(df['totalEnergy']-dftfeTotalEnergy)/2

    print("band energy errors:")
    print(df['bandEnergyError'])
    print("exchange energy errors:")
    print(df['exchangeEnergyError'])
    print("correlation energy errors:")
    print(df['correlationEnergyError'])
    print("total energy errors: \n")
    print(df['totalEnergyErrorPerAtom'])
    

    
# Combined error plot
    f2, ax2 = plt.subplots(1, 1, figsize=(10,6))
    df.plot(x='Iteration', y='bandEnergyError', logy=True,ax=ax2, style='bo-')
    df.plot(x='Iteration', y='exchangeEnergyError', logy=True,ax=ax2, style='go-')
    df.plot(x='Iteration', y='correlationEnergyError',logy=True, ax=ax2, style='mo-')
    df.plot(x='Iteration', y='totalEnergyErrorPerAtom',logy=True, ax=ax2, style='ro-')
    
    ax2.legend(loc='upper right')
##    df.plot(x='Iteration', y='bandEnergyError', logy=True,ax=ax2, style='bo')
##    df.plot(x='Iteration', y='exchangeEnergyError', logy=True,ax=ax2, style='go')
##    df.plot(x='Iteration', y='correlationEnergyError',logy=True, ax=ax2, style='mo')
##    df.plot(x='Iteration', y='totalEnergyError',logy=True, ax=ax2, style='ro')
#     ax0.set_title(system + ': Energy Values During Green Iterations -- Good Initial Guess')
#     ax0.set_title(system + ': Energy Values During Green Iterations -- Bad Initial Guess')
    ax2.set_title(system + ': Energy Errors After Each SCF')
    ax2.set_ylabel('Energy (H)')
    ax2.set_xlabel('SCF Number')
    plt.savefig(plotsDir+system+'Errors_combined'+'.pdf', bbox_inches='tight',format='pdf')



    plt.show()
    

if __name__=="__main__":
#     plotBeIterationConvergence(system = 'Lithium')    
#     plotSCFconvergence(df_Li, system="Lithium")    
    plotSCFconvergence(df_CO, system="carbonMonoxide")    


#     grouped = df_good.groupby('divideCriterion')
#     for name,group in grouped:
# ##        group.plot(x=B, y=A, style='o', ax=ax, label='%s = %s'%(C,name))
#         group.plot(x='numberOfCells', y='computedE', style='o', ax=ax1,label='%s'%name)
#         group.plot(x='numberOfCells', y='computedHOMO', style='o', ax=ax2,label='%s'%name)
# 
# 
# ##    df_good.plot(x='numberOfCells', y='computedE', style='o', ax=ax1)
#     ax1.axhline(y=dftfeEnergy,color='r',label='dft-fe')
#     ax1.axhline(y=nwchemEnergy,color='g',label='nwchem')
#     ax1.set_title('Total Energy')
#     ax1.set_ylabel('Energy (H)')
#     ax1.set_ylim([-1.15,-1.135])
#     ax1.legend()
# 
# ##    df_good.plot(x='numberOfCells', y='Bandgap', style='o', ax=ax2)
# ##    ax2.axhline(y=dftfeBandgap,color='r',label='dft-fe')
#     ax2.axhline(y=nwchemHOMO,color='g',label='nwchem')
#     ax2.set_title('HOMO Energy')
#     ax2.set_ylabel('Energy (H)')
#     ax2.legend()
# 
# ##    plt.suptitle('Green Iterations results compared to DFT-FE and NWCHEM')
#     plt.tight_layout(pad=1.0)
#     plt.show()
    
    
# 
# def AversusB(df_good,A,B,save=False):
#     fig, ax = plt.subplots(figsize=(8,6))
#     fig.suptitle('%s versus %s' %(A,B))
#     df_good.plot(x=B, y=A, style='o',ax=ax)
# 
#     dftfeEnergy = -1.1376237062839634e+00
#     NWchemEnergy = -1.1372499
#     plt.axhline(y=dftfeEnergy,color='r')
#     plt.axhline(y=NWchemEnergy,color='g')
# ##    plt.plot(dftfeEnergy*np.ones(100),'r-')
# ##    plt.plot(NWchemEnergy*np.ones(100),'g-')
#     if save == True:
#         saveID = A+'Vs'+B
#         plt.savefig(plotsDir+saveID+'.pdf', bbox_inches='tight',format='pdf')
#     plt.show()
# 
# def AversusBcolorbyC(df_good,A,B,C,save=False):
#     fig, ax = plt.subplots(figsize=(8,6))
#     fig.suptitle('%s versus %s colored by %s' %(A,B,C))
#     grouped = df_good.groupby(C)
#     for name,group in grouped:
# ##        group.plot(x=B, y=A, style='o', ax=ax, label='%s = %.2f'%(C,name))
#         group.plot(x=B, y=A, style='o', ax=ax, label='%s = %s'%(C,name))
#     plt.legend(loc = 'best')
# 
#     if save == True:
#         saveID = A+'Vs'+B+'ColoredBy'+C
#         plt.savefig(plotsDir+saveID+'.pdf', bbox_inches='tight',format='pdf')
#     plt.show()
# 
# def logAversusBcolorbyC(df_good,A,B,C,save=False):
#     fig, ax = plt.subplots(figsize=(8,6))
#     fig.suptitle('Log %s versus %s colored by %s' %(A,B,C))
#     grouped = df_good.groupby(C)
#     for name,group in grouped:
#         group['logA'] = np.log10(np.abs(group[A]))
#         group.plot(x=B, y='logA', style='o', ax=ax, label='%s = %.2f'%(C,name))
# ##        group.plot(x=B, y='logA', style='o', ax=ax, label='%s = %s'%(C,name))
#     plt.legend(loc = 'best')
# 
#     if save == True:
#         saveID = 'log'+A+'Vs'+B+'ColoredBy'+C
#         plt.savefig(plotsDir+saveID+'.pdf', bbox_inches='tight',format='pdf')
#     plt.show()
# 
# def logAversusLogBcolorbyC(df_good,A,B,C,save=False):
#     fig, ax = plt.subplots(figsize=(8,6))
#     fig.suptitle('Log %s versus Log %s colored by %s' %(A,B,C))
#     grouped = df_good.groupby(C)
#     for name,group in grouped:
# ##        group['logA'] = np.log10(np.abs(group[A]))
# ##        group['logB'] = np.log10(np.abs(group[B]))
#         if isinstance(name,str):
# ##            group.plot(x='logB', y='logA', style='o', ax=ax, label='%s = %s'%(C,name))
#             group.plot(x=B, y=A, style='o', ax=ax, loglog=True,label='%s = %s'%(C,name))
#         elif isinstance(name,float):
#             group.plot(x=B, y=A, style='o', ax=ax, loglog=True,label='%s = %f'%(C,name))
#         elif isinstance(name,int):
#             group.plot(x=B, y=A, style='o', ax=ax, loglog=True,label='%s = %i'%(C,name))
#         
#     plt.legend(loc = 'best')
#     plt.xlabel(B)
#     plt.ylabel(A)
# 
#     if save == True:
#         saveID = 'log'+A+'VsLog'+B+'ColoredBy'+C
#         plt.savefig(plotsDir+saveID+'.pdf', bbox_inches='tight',format='pdf')
#     plt.show()
# 
# def loglogEversusNcolorbyOrder(df_good,A,B,C,title=None,save=False):
#     fig, ax = plt.subplots(figsize=(8,6))
# ##    fig.suptitle('Log %s versus Log %s colored by %s' %(A,B,C))
#     if title==None:
#         fig.suptitle('Ground State Energy Errors: Clenshaw-Curtis vs. Midpoint')
#     else:
#         fig.suptitle(title)
#     
# ##    fig.suptitle('Ground State Energy Errors: Levine-Wilkins-1 refinement')
#     grouped = df_good.groupby(C)
#     for name,group in grouped:
# ##        group['logA'] = np.log10(np.abs(group[A]))
# ##        group['logB'] = np.log10(np.abs(group[B]))
#         group['absA'] = np.abs(group[A])
#         group['absB'] = np.abs(group[B])
#     
#         if isinstance(name,str):
# ##            group.plot(x='logB', y='logA', style='o', ax=ax, label='%s = %s'%(C,name))
#             group.plot(x=B, y=A, style='o', ax=ax, loglog=True,label='%s = %s'%(C,name))
#         elif isinstance(name,float):
#             group.plot(x='absB', y='absA', style='o', loglog=True, ax=ax, label='%s = %f'%(C,name))
#         elif isinstance(name,int):
#             if name==0:
#                 group.plot(x='absB', y='absA', style='o', loglog=True, ax=ax, label='Midpoint')
#             else:
#                group.plot(x='absB', y='absA', style='o', loglog=True, ax=ax, label='CC %s %i'%(C,name))
#         
#     plt.legend(loc = 'best')
#     plt.xlabel('Number of Cells')
#     plt.ylabel('Energy Error (Hartree)')
# ##    plt.yticks([1e-6,1e-5,1e-4,1e-3,1e-2],['1e-6','1e-5','1e-4','1e-3','1e-2'])
# ##    plt.xticks([2e4, 1e5,2e5, 3e6],['2e4', '1e5','2e5', '3e6'])
# 
#     if save != False:
# ##        saveID = A+'Vs'+B+'ColoredBy'+C
#         saveID = save
#         plt.savefig(plotsDir+saveID+'.pdf', bbox_inches='tight',format='pdf')
#     plt.show()


