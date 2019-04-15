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
# 
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Oxygen_integralSqrtDensity/'
# file='ds_cellOrder5maxDepth15_2_3_0.3_0.05_GREEN_.csv'


# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/benzeneTests/'  
# # file='tc_gaugeShift-0.5_tcOrder8_theta0.7_PCI_1e-06_GREEN_1493000.csv'
# # file='tc_gaugeShift0p5_tcOrder8_theta0.8_LW5_500_mixingHistory10_GREEN_1485000.csv' 
# # file='tc_gaugeShift0p5_tcOrder7_theta0.8_LW5_800_mixingHistory6_GREEN_2787000.csv'
# # file='tc_gaugeShift0p5_mindepth3_tcOrder7_theta0.8_LW5_1000_mixingHistory6_GREEN_2948000.csv'
# # file='tc_gaugeShift0p5_mindepth3_tcOrder7_theta0.8_LW5_500_mixingHistory6_GREEN_1485000.csv'
# # file='tc_gaugeShift0p5_mindepth3_tcOrder7_theta0.8_PCI_1e-2_9random_mixingHistory10_GREEN_2172000.csv'
# # file='tc_gaugeShift-1.0_tcOrder8_theta0.7_PCI_1e-06_GREEN_1416000.csv' 
# file='tc_exp_init_gaugeShift-1.0_tcOrder8_theta0.7_LW5_500_GREEN_1416000.csv'
  
# file='tc_gaugeShift0p5_mindepth3_tcOrder7_theta0.8_PCI_1e-2_9random_mixingHistory6_GREEN_2172000.csv'
  
# file='ds_gaugeShift0p25_mindepth3_tcOrder7_theta0.8_LW5_500_1random_mixingHistory15_GREEN_1493000.csv'
# file='ds_gaugeShift0p25_mindepth3_tcOrder5_theta0.8_LW5_500_1random_mixingHistory10_GREEN_1493000.csv'
# file='tc_gaugeShift-0.5_mindepth3_tcOrder8_theta0.7_LW5_500_GREEN_1493000.csv'


# ## Symmertric Green's iterations tests
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/symmetricTest/'
# # file='CO_asymm_LW5_GREEN_375500.csv' 
# file='CO_symm_LW5_GREEN_375500.csv' 
# # file='CO_aitken_after_3_LW5_GREEN_375500.csv'
# # file='CO_aitken_after_12_LW5_GREEN_375500.csv'
 
# Testing simultaneous GI

# Testing ScipyAnderson
resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/benzeneTests/'
# file='CO_gaugeShift-0.5_LW5_500_GREEN_375500.csv'
# file='CO_mixingM5_gaugeShift-0.5_LW5_500_GREEN_1493000.csv'
# file='scipyAnderson_10initIterations_gaugeShift-0.5_tcOrder8_theta0.7_PCI_1e-06_GREEN_1416000.csv'
# file='scipyAnderson_looser_gaugeShift-0.5_tcOrder8_theta0.7_PCI_1e-06_GREEN_1416000.csv'
file='scipyAnderson_init1Sonly_gaugeShift-0.5_tcOrder8_theta0.7_PCI_1e-06_GREEN_1416000.csv'  

# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/gaugeShiftTests/'
# file='CO_swap_gaugeShift-1.0_LW5_500_GREEN_375500.csv'

# file='O2_degeneracyTest_gaugeShift-0.5_LW5_500_GREEN_375500.csv'
# file='CO_shiftedIteration_simultaneous_gaugeShift-0.0_LW5_500_GREEN_375500.csv'
# file = 'CO_TaylorInverseIteration_simultaneous_gaugeShift-0.0_LW5_500_GREEN_375500.csv'


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

## Carbon Monoxide
# nWavefunctions=7
# referenceEnergies = np.array( [   -1.871953147002199813e+01, -9.907188115343084078e+00,
#                                   -1.075324514852165958e+00, -5.215419985881135645e-01,
#                                   -4.455527567163568570e-01, -4.455527560478895199e-01,
#                                   -3.351419327004790394e-01] )

## OXYGEN
# referenceEnergies = np.array([psi0,psi1,psi2,psi3,psi4])

## H2
##referenceEnergies = np.array([-0.3774974859])


# # ## Benzene
nWavefunctions=23
referenceEnergies = np.array( [ -9.791997000000000, 
                                -9.791806000000000,    
                                -9.791746000000000,    
                                -9.791384000000000,   
                                -9.791313000000000,  
                                -9.791129000000000,   
                                -0.777636800000000,    
                                -0.674908000000000,    
                                -0.674900600000000,    
                                -0.544847700000000,    
                                -0.544836100000000,    
                                -0.478288500000000,     
                                -0.409873800000000,    
                                -0.406600100000000,     
                                -0.378339300000000,    
                                -0.378337500000000,    
                                -0.340971300000000,    
                                -0.304777300000000,    
                                -0.304703800000000,    
                                -0.240343300000000,    
                                -0.240322100000000,
                                0.0000,
                                0.0000]) 

# residualsMatrix = np.zeros((df.shape[0],10))
# errorsMatrix = np.zeros((df.shape[0],10))
# eigenvaluesMatrix = np.zeros((df.shape[0],10))
# errorsMatrix1st = np.zeros((df.shape[0],10))

residualsMatrix = np.zeros((df.shape[0],nWavefunctions))
errorsMatrix = np.zeros((df.shape[0],nWavefunctions))
eigenvaluesMatrix = np.zeros((df.shape[0],nWavefunctions))
errorsMatrix1st = np.zeros((df.shape[0],nWavefunctions))
for i in range(df.shape[0]):
    residualsMatrix[i,:] = np.array(df.orbitalResiduals[i][1:-1].split(),dtype=float)
    errorsMatrix[i,:] = abs( np.array( df.energyEigenvalues[i][1:-1].split(),dtype=float) - referenceEnergies )
    eigenvaluesMatrix[i,:] =  np.array( df.energyEigenvalues[i][1:-1].split(),dtype=float) 
##    errorsMatrix[i,:] = np.array( df.energyEigenvalues[i][1:-1].split(),dtype=float)
    try:
        errorsMatrix1st[i,:] = np.array( df.energyErrorsWRTfirstSCF[i][1:-1].split(),dtype=float) 
    except AttributeError:
        pass

for i in range(nWavefunctions):
    df['residual'+str(i)] = residualsMatrix[:,i]
    df['errors'+str(i)] = np.abs(errorsMatrix[:,i])
    df['eigenvalue'+str(i)] = eigenvaluesMatrix[:,i]





def plotFirstSCF(df):
    
    
#     f, ax = plt.subplots(1,1, figsize=(12,6))
#     df.plot(y='eigenvalue0',ax=ax,label='psi0')
#     df.plot(y='eigenvalue1',ax=ax,label='psi1')
#     df.plot(y='eigenvalue2',ax=ax,label='psi2')
#     df.plot(y='eigenvalue3',ax=ax,label='psi3')
#     df.plot(y='eigenvalue4',ax=ax,label='psi4')
# #     df.plot(y='eigenvalue5',ax=ax,label='psi5')
# #     df.plot(y='eigenvalue6',ax=ax,label='psi6')
# #     df.plot(y='eigenvalue7',ax=ax,label='psi7')
# #     df.plot(y='eigenvalue8',ax=ax,label='psi8')
# #     df.plot(y='eigenvalue9',ax=ax,label='psi9')
#     
#     for i in range(len(referenceEnergies)):
#         ax.axhline(y=referenceEnergies[i],color='k',linestyle='--',linewidth=0.5)
#     ax.set_xlabel('Iteration Number')
#     ax.set_ylabel('Eigenvalues')
#     ax.set_title('Eigenvalues During First SCF Iteration')
    
#     plt.show()
#     return

    
    f0, (ax0, ax1) = plt.subplots(1,2, figsize=(12,6))
    f0, (ax2) = plt.subplots(1,1, figsize=(12,6))
    for i in range(nWavefunctions):
        df.plot(y='residual'+str(i),marker='o',ax=ax0,logy=True,label='psi'+str(i),legend=False)
        df.plot(y='errors'+str(i),marker='o',ax=ax2,logy=True,label='psi'+str(i),legend=False)
        df.plot(y='eigenvalue'+str(i),marker='o',ax=ax1,logy=False,label='psi'+str(i),legend=False)

    
    ax0.set_xlabel('Iteration Number')
    ax0.set_ylabel('Residual L2 Norm')
    ax0.set_title('Orbital Residuals')

    ax2.set_xlabel('Iteration Number')
    ax2.set_ylabel('Energy Error (Hartree)')
##    ax1.set_ylim([1e-4,2e-2])
    ax2.set_title('Eigenvalue Errors')
    ax2.grid()


    ax1.set_xlabel('Iteration Number')
    ax1.set_ylabel('Energy (Hartree)')   
##    ax1.set_ylim([1e-4,2e-2])
    ax1.set_title('Wavefunction Eigenvalues')


    plt.suptitle('Convergence of Green Iterations for Carbon Monoxide')
#     plt.suptitle('Convergence of Green Iterations for Benzene')
#     plt.suptitle('Convergence of Green Iterations for Oxygen')


    plt.show()
    
def plot_eigenvalues(eigenvaluesMatrix,referenceEnergies):

    (n,m) = np.shape(eigenvaluesMatrix)
    
    # fix some bad initial values from before psi5 was being updated
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
        plt.plot(eigs,label='psi%i'%i)
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


