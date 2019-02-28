'''
Created on Mar 29, 2018

@author: nathanvaughn
'''

import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import os
import numpy as np


##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/carbonMonoxide/iterationResults/'

#### H2
##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/H2IterationResults/'
##file='LW3_2500_GREEN_.csv'
##
#### Lithium
##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/LithiumIterationResults/'
##file='LW3_1000_GREEN_.csv'

#### Beryllium
##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/LithiumIterationResults/'
##file='LW3_1000_GREEN_.csv'
# resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/Be_gradientFree/Be_gradientFree/'
# file = 'LW5_3000_order5_gradient_GREEN_.csv'


## Oxygen
##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenIterationResults/'
##file='LW3_1500_GREEN_.csv'
# resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/'
# file = 'HarrisonGradientFree_LW5_3000_order5_GREEN_.csv'
##resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenOrder5/'
##file='LW3_1000_GREEN_.csv'
# resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/O_gradientFree/'

# file = 'LW5_2000_order5_gradient_GREEN_.csv'
# file = 'LW5_2000_order6_gradientFree_GREEN_.csv'



# resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/oxygen_with_anderson/'
# file='LW5_1500_andersonMixing_p5_1em8_GREEN_.csv'


# resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/CO_with_anderson/'
# file='LW5_1000o5_gradientFree_eigRes_GREEN_.csv'
# resultsDir = '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/'
# file ='LW5_1500o5_GradientFree_eigRes_looseThenTight_titan_GREEN_.csv'
# file='LW5o5_1500_GREEN_.csv'
# file='LW5o5_1500_largeDomain_GREEN_.csv'
# file='LW5o5_1500_tight_GREEN_.csv'
# file='LW5o4_500_fixedMesh_GREEN_.csv'
# file='LW5o5_1000_fixedMesh_GREEN_.csv'
# file='LW5o4_2000_fixedMesh_randInit_noTitan_GREEN_.csv'
# file='LW5o5_1000_fixedMesh_only7_looseInit_GREEN_.csv'
# file='LW5o5_1500_fixedAtomicPositions_only7_looseInit_GREEN_.csv'
# file='LW5o4_1000_only7_tightFromStart_GIanderson_afterSCF1_GREEN_.csv'
# file='LW5o5_2000_only7_tightFromStart_GIandersonAfterSCF1_GREEN_.csv'
# file='LW5o5_2000_6_orbitals_GREEN_.csv'
# file='LW5o5_2000_7_orbitals_noGIanderson_GREEN_.csv'

# ## Oxygen -- Biros
# # # resultsDir = '/Users/nathanvaughn/Desktop/meshTests/LWvsBiros/Oxygen/'
# resultsDir = '/Users/nathanvaughn/Desktop/meshTests/LWvsBiros/Oxygen/'
# plotsDir = resultsDir+'plots/'
# # # # file='Biros_o7_1em4_solo_SCF_.csv'
# # # # file='Biros_o7_7em5_alpha_1p5_SCF_.csv'
# # # # file='BirosN_o7_1em3_SCF_.csv'
# # # # file='BirosN_o7_2em4_SCF_.csv'
# # file='BirosG_o7_max15_SCF_.csv'
# # file='BirosG_o7_1em5_GREEN_.csv'
# file='BirosGN_o7_1em2_GREEN_.csv'


### Oxygen 4 parameter
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Oxygen_densityIntegral4th/'
# file='ds_cellOrder5_maxDepth15_3_3_0.3_0.03_GREEN_.csv'

# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Oxygen_psiVextVariation/'
# file='ds_cellOrder5maxDepth12_2_2_0.1_50000_GREEN_.csv'

resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Oxygen_integralSqrtDensity/'
file='ds_cellOrder5maxDepth15_2_3_0.3_0.05_GREEN_.csv'


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
    
if ( (resultsDir == '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/O_gradientFree/') or 
     (resultsDir == '/Users/nathanvaughn/Desktop/meshTests/LWvsBiros/Oxygen/') ):

    TotalEnergy = -7.4469337501098821e+01
    ExchangeEnergy = -7.2193700828939980e+00
    CorrelationEnergy = -5.4455323568788838e-01
    BandEnergy = -4.0613397710076626e+01
    KineticEnergy =  7.4112730191157425e+01
    ElectrostaticEnergy = -1.4081814437367436e+02




##if resultsDir == '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenIterationResults/':
##if resultsDir == '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/OxygenOrder5/':
BandEnergy = -4.0613161867355522e+01
KineticEnergy = 7.4112356154762352e+01
ExchangeEnergy = -7.2193342430668448e+00
CorrelationEnergy = -5.4455279159404091e-01
ElectrostaticEnergy = -1.4081748175072624e+02
TotalEnergy = -7.4469012630624775e+01

psi0 = -1.875879493052850933e+01
psi1 = -8.711989839502347621e-01
psi2 = -3.382966064098608672e-01
psi3 = -3.382966063773331644e-01
psi4 = -3.382966063244193800e-01


if resultsDir == '/Users/nathanvaughn/Desktop/ClenshawCurtisGreenIterations/H2IterationResults/':
    TotalEnergy = -1.1376691191341821e+00
    ExchangeEnergy = -5.5876966592456134e-01
    CorrelationEnergy = -9.4268448521496129e-02
    BandEnergy = -7.5499497178953057e-01



plotsDir = resultsDir+'plots/'

df = pd.read_csv(resultsDir+file, header=0)

print(df.shape)

##referenceEnergies = np.array([-1.871923237485756886e+01,
##                              -9.907091923705507952e+00,
##                              -1.075296339177760352e+00,
##                              -5.215175505938050016e-01,
##                              -4.455359698088234843e-01,
##                              -4.455359698088187659e-01 ,
##                              -3.351144822320292205e-01])


""" AFTER FIRST SCF """
# referenceEnergies = np.array( [-1.886761702508549021e+01, -1.000441073298974892e+01,
#                                             -1.185545917003633321e+00, -6.070872074377245964e-01,
#                                             -5.201973981507257427e-01, -5.201973981507234113e-01,
#                                             -3.960960368603070325e-01,-1.338775668379516559e-02,
#                                             -7.325760563979200057e-02, 1.721054880813185223e-02,] )

# referenceEnergies = np.array( [-1.886761702508549021e+01, -1.000441073298974892e+01,
#                                             -1.185545917003633321e+00, -6.070872074377245964e-01,
#                                             -5.201973981507257427e-01, -5.201973981507234113e-01,
#                                             -3.960960368603070325e-01] )

""" FINAL EIGENVALUES """
# referenceEnergies = np.array( [   -1.871953147002199813e+01, -9.907188115343084078e+00,
#                                   -1.075324514852165958e+00, -5.215419985881135645e-01,
#                                   -4.455527567163568570e-01, -4.455527560478895199e-01,
#                                   -3.351419327004790394e-01, -8.275071966753577701e-02,
#                                   -8.273399296312561324e-02,  7.959071929649078059e-03] )

# referenceEnergies = np.array( [   -1.871953147002199813e+01, -9.907188115343084078e+00,
#                                   -1.075324514852165958e+00, -5.215419985881135645e-01,
#                                   -4.455527567163568570e-01, -4.455527560478895199e-01,
#                                   -3.351419327004790394e-01] )

## OXYGEN
referenceEnergies = np.array([psi0,psi1,psi2,psi3,psi4])

## H2
##referenceEnergies = np.array([-0.3774974859])


# residualsMatrix = np.zeros((df.shape[0],10))
# errorsMatrix = np.zeros((df.shape[0],10))
# eigenvaluesMatrix = np.zeros((df.shape[0],10))
# errorsMatrix1st = np.zeros((df.shape[0],10))

residualsMatrix = np.zeros((df.shape[0],5))
errorsMatrix = np.zeros((df.shape[0],5))
eigenvaluesMatrix = np.zeros((df.shape[0],5))
errorsMatrix1st = np.zeros((df.shape[0],5))
for i in range(df.shape[0]):
    residualsMatrix[i,:] = np.array(df.orbitalResiduals[i][1:-1].split(),dtype=float)
    errorsMatrix[i,:] = abs( np.array( df.energyEigenvalues[i][1:-1].split(),dtype=float) - referenceEnergies )
    eigenvaluesMatrix[i,:] =  np.array( df.energyEigenvalues[i][1:-1].split(),dtype=float) 
##    errorsMatrix[i,:] = np.array( df.energyEigenvalues[i][1:-1].split(),dtype=float)
    try:
        errorsMatrix1st[i,:] = np.array( df.energyErrorsWRTfirstSCF[i][1:-1].split(),dtype=float) 
    except AttributeError:
        pass

df['residual0'] = residualsMatrix[:,0]
df['residual1'] = residualsMatrix[:,1]
df['residual2'] = residualsMatrix[:,2]
df['residual3'] = residualsMatrix[:,3]
df['residual4'] = residualsMatrix[:,4]
# df['residual5'] = residualsMatrix[:,5]
# df['residual6'] = residualsMatrix[:,6]
# df['residual7'] = residualsMatrix[:,7]
# df['residual8'] = residualsMatrix[:,8]
# df['residual9'] = residualsMatrix[:,9]

##df['errors0'] = np.copy(df['energyEigenvalues'])
df['errors0'] = np.abs(errorsMatrix[:,0])
df['errors1'] = np.abs(errorsMatrix[:,1])
df['errors2'] = np.abs(errorsMatrix[:,2])
df['errors3'] = np.abs(errorsMatrix[:,3])
df['errors4'] = np.abs(errorsMatrix[:,4])
# df['errors5'] = np.abs(errorsMatrix[:,5])
# df['errors6'] = np.abs(errorsMatrix[:,6])
# df['errors7'] = np.abs(errorsMatrix[:,7])
# df['errors8'] = np.abs(errorsMatrix[:,8])
# df['errors9'] = np.abs(errorsMatrix[:,9])

df['eigenvalue0'] = eigenvaluesMatrix[:,0]
df['eigenvalue1'] = eigenvaluesMatrix[:,1]
df['eigenvalue2'] = eigenvaluesMatrix[:,2]
df['eigenvalue3'] = eigenvaluesMatrix[:,3]
df['eigenvalue4'] = eigenvaluesMatrix[:,4]
# df['eigenvalue5'] = eigenvaluesMatrix[:,5]
# df['eigenvalue6'] = eigenvaluesMatrix[:,6]
# df['eigenvalue7'] = eigenvaluesMatrix[:,7]
# df['eigenvalue8'] = eigenvaluesMatrix[:,8]
# df['eigenvalue9'] = eigenvaluesMatrix[:,9]

##try:
##    df['1stSCFerrors0'] = np.abs(errorsMatrix1st[:,0])
##    df['1stSCFerrors1'] = np.abs(errorsMatrix1st[:,1])
##    df['1stSCFerrors2'] = np.abs(errorsMatrix1st[:,2])
##    df['1stSCFerrors3'] = np.abs(errorsMatrix1st[:,3])
##    df['1stSCFerrors4'] = np.abs(errorsMatrix1st[:,4])
##    df['1stSCFerrors5'] = np.abs(errorsMatrix1st[:,5])
##    df['1stSCFerrors6'] = np.abs(errorsMatrix1st[:,6])
##except AttributeError:
##    pass

def plotFirstSCF(df):
    
    
#     f, ax = plt.subplots(1,1, figsize=(12,6))
#     df.plot(y='eigenvalue0',ax=ax,label='Phi0')
#     df.plot(y='eigenvalue1',ax=ax,label='Phi1')
#     df.plot(y='eigenvalue2',ax=ax,label='Phi2')
#     df.plot(y='eigenvalue3',ax=ax,label='Phi3')
#     df.plot(y='eigenvalue4',ax=ax,label='Phi4')
# #     df.plot(y='eigenvalue5',ax=ax,label='Phi5')
# #     df.plot(y='eigenvalue6',ax=ax,label='Phi6')
# #     df.plot(y='eigenvalue7',ax=ax,label='Phi7')
# #     df.plot(y='eigenvalue8',ax=ax,label='Phi8')
# #     df.plot(y='eigenvalue9',ax=ax,label='Phi9')
#     
#     for i in range(len(referenceEnergies)):
#         ax.axhline(y=referenceEnergies[i],color='k',linestyle='--',linewidth=0.5)
#     ax.set_xlabel('Iteration Number')
#     ax.set_ylabel('Eigenvalues')
#     ax.set_title('Eigenvalues During First SCF Iteration')
    
#     plt.show()
#     return

    
    f0, (ax0, ax1) = plt.subplots(1,2, figsize=(12,6))
    df.plot(y='residual0',ax=ax0,logy=True,label='Phi0')
    df.plot(y='residual1',ax=ax0,logy=True,label='Phi1')
    df.plot(y='residual2',ax=ax0,logy=True,label='Phi2')
    df.plot(y='residual3',ax=ax0,logy=True,label='Phi3')
    df.plot(y='residual4',ax=ax0,logy=True,label='Phi4')
#     df.plot(y='residual5',ax=ax0,logy=True,label='Phi5')
#     df.plot(y='residual6',ax=ax0,logy=True,label='Phi6')
#     df.plot(y='residual7',ax=ax0,logy=True,label='Phi7')
#     df.plot(y='residual8',ax=ax0,logy=True,label='Phi8')
#     df.plot(y='residual9',ax=ax0,logy=True,label='Phi9')
    ax0.set_xlabel('Iteration Number')
    ax0.set_ylabel('Residual L2 Norm')
    ax0.set_title('Orbital Residuals')


    df.plot(y='errors0',ax=ax1,logy=True,label='Phi0')
    df.plot(y='errors1',ax=ax1,logy=True,label='Phi1')
    df.plot(y='errors2',ax=ax1,logy=True,label='Phi2')
    df.plot(y='errors3',ax=ax1,logy=True,label='Phi3')
    df.plot(y='errors4',ax=ax1,logy=True,label='Phi4')
#     df.plot(y='errors5',ax=ax1,logy=True,label='Phi5')
#     df.plot(y='errors6',ax=ax1,logy=True,label='Phi6')
#     df.plot(y='errors7',ax=ax1,logy=True,label='Phi7')
#     df.plot(y='errors8',ax=ax1,logy=True,label='Phi8')
#     df.plot(y='errors9',ax=ax1,logy=True,label='Phi9')
    ax1.set_xlabel('Iteration Number')
    ax1.set_ylabel('Energy Error (Hartree)')
##    ax1.set_ylim([1e-4,2e-2])
    ax1.set_title('Orbital Errors')

##    plt.suptitle('Using Singularity Skipping, LW3-800')
##    plt.suptitle('Using Singularity Subtraction, LW3-800, minDepth 3')
##    plt.suptitle(file)
##    plt.suptitle('Convergence of Green Iterations for Oxygen -- Coarse')
    plt.suptitle('Convergence of Green Iterations for Carbon Monoxide')
#     plt.suptitle('Convergence of Green Iterations for Oxygen')

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
    
def plot_eigenvalues(eigenvaluesMatrix,referenceEnergies):

    (n,m) = np.shape(eigenvaluesMatrix)
    
    # fix some bad initial values from before phi5 was being updated
    for i in range(n):
        if eigenvaluesMatrix[i,5] < -10.0:
            eigenvaluesMatrix[i,5] = 0
    
    
    f, ax = plt.subplots(1,1, figsize=(12,8))
#     for i in range(m):
    for i in range(0,1):
        eigs = []
#         for j in range(315):
        for j in range(n-1):
            if eigenvaluesMatrix[j,i] != eigenvaluesMatrix[j+1,i]:
                eigs.append(eigenvaluesMatrix[j,i])
        plt.plot(eigs,label='phi%i'%i)
        if i==0:
            plt.axhline(y=referenceEnergies[i],color='k',linestyle='--',linewidth=0.5,label='Reference')
        else:
            plt.axhline(y=referenceEnergies[i],color='k',linestyle='--',linewidth=0.5)
        
        
#     for i in range(len(referenceEnergies)):
#         if i==0:
#             plt.axhline(y=referenceEnergies[i],color='k',linestyle='--',linewidth=0.5,label='Reference')
#         else:
#             plt.axhline(y=referenceEnergies[i],color='k',linestyle='--',linewidth=0.5)
        
    plt.legend()
#     plt.ylim([-20,0])
    plt.title('Orbital Energies During First SCF Iteration')
    plt.xlabel('Green Iteration Count')
    plt.ylabel('Energy (H)')
    plt.show()
    
    return



    

if __name__=="__main__":
#     print(df.head())
    plotFirstSCF(df)
#     plot_eigenvalues(eigenvaluesMatrix,referenceEnergies)


