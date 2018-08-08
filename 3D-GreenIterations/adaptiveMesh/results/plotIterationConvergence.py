'''
Created on Mar 29, 2018

@author: nathanvaughn
'''

import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import os
import numpy as np


resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations'
# resultsDir = '/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/KohnShamTests'
plotsDir = resultsDir+'/plots/'
# df_H2 = pd.read_csv(resultsDir+'/iterationConvergenceHe_LW3_1200.csv', header=0)
# df_H2 = pd.read_csv(resultsDir+'/iterationConvergenceLi_LW3_1200.csv', header=0)
# df_bad = pd.read_csv(resultsDir+'/iterationConvergenceH2_LW3_800_perturbed.csv', header=0)
# df_good = pd.read_csv(resultsDir+'/iterationConvergenceBe_LW3_1200.csv', header=0)
# df_bad = pd.read_csv(resultsDir+'/iterationConvergenceBe_LW3_1200_perturbed.csv', header=0)


# Lithium
df_good = pd.read_csv(resultsDir+'/iterationConvergenceLi_LW3_1200_ssForPhi0.csv', header=0)
df_bad = pd.read_csv(resultsDir+'/iterationConvergenceLi_LW3_1200.csv', header=0)


def plotBeIterationConvergence(system="Beryllium"):
   
    if system == "Beryllium": 
        dftfeTotalEnergy = -1.4446182766680081e+01
        dftfeExchangeEnergy = -2.2902495359115198e+00
        dftfeCorrelationEnergy = -2.2341044592808737e-01
        dftfeBandEnergy = -8.1239182420318166e+00
    
    if system == "Li":
        dftfeTotalEnergy = -7.3340536782581447
        dftfeExchangeEnergy = -1.4916149721121696
        dftfeCorrelationEnergy = -1.5971669832262905e-01
        dftfeBandEnergy = -3.8616389456972078

    # Combined plot
    f0, ax0 = plt.subplots(1, 1, figsize=(8,6))
    df_good.plot(x='Iteration', y='exchangeEnergy', ax=ax0, style='bo', label='Good Initial Guess')
    df_good.plot(x='Iteration', y='correlationEnergy', ax=ax0, style='go', label='Good Initial Guess')
    df_good.plot(x='Iteration', y='totalEnergy', ax=ax0, style='ro', label='Good Initial Guess')
    
    df_bad.plot(x='Iteration', y='exchangeEnergy', ax=ax0, style='bx', label='Bad Initial Guess')
    df_bad.plot(x='Iteration', y='correlationEnergy', ax=ax0, style='gx', label='Bad Initial Guess')
    df_bad.plot(x='Iteration', y='totalEnergy', ax=ax0, style='rx', label='Bad Initial Guess')
    
    ax0.axhline(y=dftfeExchangeEnergy,color='b',label='dft-fe')
    ax0.axhline(y=dftfeCorrelationEnergy,color='g',label='dft-fe')
    ax0.axhline(y=dftfeTotalEnergy,color='r',label='dft-fe')
    
    ax0.legend()
#     ax0.set_title(system + ': Energy Values During Green Iterations -- Good Initial Guess')
#     ax0.set_title(system + ': Energy Values During Green Iterations -- Bad Initial Guess')
    ax0.set_title(system + ': Energy Values During Green Iterations')
    ax0.set_ylabel('Energy (H)')
    ax0.set_xlabel('Iteration Number')
    plt.savefig(plotsDir+system+'Energies'+'.pdf', bbox_inches='tight',format='pdf')

    

    # Individual error plots
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,8), sharex=True)
    
    df_good['exchangeEnergyError -- Good Initial']=abs(df_good['exchangeEnergy']-dftfeExchangeEnergy)
    df_good['correlationEnergyError -- Good Initial']=abs(df_good['correlationEnergy']-dftfeCorrelationEnergy)
    df_good['totalEnergyError -- Good Initial']=abs(df_good['totalEnergy']-dftfeTotalEnergy)
    
    df_bad['exchangeEnergyError -- Bad Initial']=abs(df_bad['exchangeEnergy']-dftfeExchangeEnergy)
    df_bad['correlationEnergyError -- Bad Initial']=abs(df_bad['correlationEnergy']-dftfeCorrelationEnergy)
    df_bad['totalEnergyError -- Bad Initial']=abs(df_bad['totalEnergy']-dftfeTotalEnergy)
    
    df_good.plot(x='Iteration', y='exchangeEnergyError -- Good Initial', logy=True, ax=ax1, style='bo')
    df_good.plot(x='Iteration', y='correlationEnergyError -- Good Initial', logy=True, ax=ax2, style='go')
    df_good.plot(x='Iteration', y='totalEnergyError -- Good Initial', logy=True, ax=ax3, style='ro')
    
    df_bad.plot(x='Iteration', y='exchangeEnergyError -- Bad Initial', logy=True, ax=ax1, style='bx')
    df_bad.plot(x='Iteration', y='correlationEnergyError -- Bad Initial', logy=True, ax=ax2, style='gx')
    df_bad.plot(x='Iteration', y='totalEnergyError -- Bad Initial', logy=True, ax=ax3, style='rx')

#     ax1.axhline(y=dftfeExchangeEnergy,color='b',label='dft-fe')
#     ax2.axhline(y=dftfeCorrelationEnergy,color='g',label='dft-fe')
#     ax3.axhline(y=dftfeTotalEnergy,color='r',label='dft-fe')
    
    ax1.legend()
    ax2.legend()
    ax3.legend()
    
    ax1.set_ylabel('Energy (H)')
    ax2.set_ylabel('Energy (H)')
    ax3.set_ylabel('Energy (H)')
    ax3.set_xlabel('Iteration Number')
    
#     plt.suptitle(system + ': Absolute Errors During Iterations -- Good Initial Guess')
#     plt.suptitle(system + ': Absolute Errors During Iterations -- Bad Initial Guess')
    plt.suptitle(system + ': Absolute Errors During Iterations')
    plt.savefig(plotsDir+system+'Errors'+'.pdf', bbox_inches='tight',format='pdf')
    
    plt.show()
    
def plotH2IterationConvergence(system = 'H2'):
    
    if system == 'H2':
        dftfeTotalEnergy = -1.1376691191341821e+00
        dftfeExchangeEnergy = -5.5876966592456134e-01
        dftfeCorrelationEnergy = -9.4268448521496129e-02
        dftfeBandEnergy = -7.5499497178953057e-01
    
    if system == "Li":
        dftfeTotalEnergy = -7.3340536782581447
        dftfeExchangeEnergy = -1.4916149721121696
        dftfeCorrelationEnergy = -1.5971669832262905e-01
        dftfeBandEnergy = -3.8616389456972078
    
    # Combined plot
    f0, ax0 = plt.subplots(1, 1, figsize=(8,6))
    df_H2.plot(x='Iteration', y='exchangeEnergy', ax=ax0, style='bo')
    df_H2.plot(x='Iteration', y='correlationEnergy', ax=ax0, style='go')
#     df_H2.plot(x='Iteration', y='bandEnergy', ax=ax0, style='mo')
    df_H2.plot(x='Iteration', y='totalEnergy', ax=ax0, style='ro')
    

    ax0.axhline(y=dftfeExchangeEnergy,color='b',label='dft-fe')
    ax0.axhline(y=dftfeCorrelationEnergy,color='g',label='dft-fe')
    ax0.axhline(y=dftfeBandEnergy,color='m',label='dft-fe')
    ax0.axhline(y=dftfeTotalEnergy,color='r',label='dft-fe')
    
    ax0.legend()
#     ax0.set_title(system + ': Energy Values During Green Iterations -- Good Initial Guess')
#     ax0.set_title(system + ': Energy Values During Green Iterations -- Bad Initial Guess')
    ax0.set_title(system + ': Energy Values During Green Iterations')
    ax0.set_ylabel('Energy (H)')
    ax0.set_xlabel('Iteration Number')
    plt.savefig(plotsDir+system+'Energies'+'.pdf', bbox_inches='tight',format='pdf')

    

    # Individual error plots
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,8), sharex=True)
    
    df_H2['exchangeEnergyError']=abs(df_H2['exchangeEnergy']-dftfeExchangeEnergy)
    df_H2['correlationEnergyError']=abs(df_H2['correlationEnergy']-dftfeCorrelationEnergy)
    df_H2['totalEnergyError']=abs(df_H2['totalEnergy']-dftfeTotalEnergy)

    
    df_H2.plot(x='Iteration', y='exchangeEnergyError', logy=True, ax=ax1, style='bo')
    df_H2.plot(x='Iteration', y='correlationEnergyError', logy=True, ax=ax2, style='go')
    df_H2.plot(x='Iteration', y='totalEnergyError', logy=True, ax=ax3, style='ro')
    


#     ax1.axhline(y=dftfeExchangeEnergy,color='b',label='dft-fe')
#     ax2.axhline(y=dftfeCorrelationEnergy,color='g',label='dft-fe')
#     ax3.axhline(y=dftfeTotalEnergy,color='r',label='dft-fe')
    
    ax1.legend()
    ax2.legend()
    ax3.legend()
    
    ax1.set_ylabel('Energy (H)')
    ax2.set_ylabel('Energy (H)')
    ax3.set_ylabel('Energy (H)')
    ax3.set_xlabel('Iteration Number')
    
#     plt.suptitle(system + ': Absolute Errors During Iterations -- Good Initial Guess')
#     plt.suptitle(system + ': Absolute Errors During Iterations -- Bad Initial Guess')
    plt.suptitle(system + ': Absolute Errors During Iterations')
    plt.savefig(plotsDir+system+'Errors'+'.pdf', bbox_inches='tight',format='pdf')
    
    plt.show()
    

if __name__=="__main__":
    plotBeIterationConvergence(system = 'Li')    
#     plotH2IterationConvergence(system="Li")    


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


