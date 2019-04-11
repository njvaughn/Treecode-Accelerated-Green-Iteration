'''
Created on Jan 16, 2019

@author: nathanvaughn
'''
import numpy as np
from scipy.optimize import anderson as scipyAnderson

# global tree, orbitals, oldOrbitals

def PsiNorm(psi):
    return np.sqrt( np.sum(psi*psi*weights) )

   
def PowerIteration(psiIn):
    psiOut = np.dot(A,psiIn)
    psiOut /= np.linalg.norm(psiOut)
    return psiOut - psiIn

def PowerIteration2(psiIn):
#     print(psi1)
#     print(weights)
    psiOut = np.dot(A,psiIn)
    psiOut /= np.sqrt( np.dot(psiOut,psiOut) )
    psiOut -= ( np.dot(psiOut, psi1) )/( np.dot(psi1, psi1) ) * psi1
#     psiOut -= np.sqrt( np.dot(psiOut, psi1) )/np.sqrt( np.dot(psi1, psi1) ) * psi1
#     dot = np.dot(psiOut, psi1)
#     print('dot ', dot)
#     overlap=np.sqrt( np.dot(psiOut, psi1) )/ np.linalg.norm(psi1) 
#     print(overlap)
#     psiOut -= overlap * psi1
#     psiOut /= np.linalg.norm(psiOut)
    psiOut /= np.sqrt( np.dot(psiOut,psiOut) )
    
#     r = np.dot( psiOut, np.dot(A,psiOut)) 
#     
#     psiOut *= np.sign(r)
#     print(psiOut)
#     print()

    if np.linalg.norm(psiOut - psiIn) < np.linalg.norm(psiOut + psiIn):
        return psiOut - psiIn
    else:
        psiOut *= -1
        return psiOut - psiIn


def PowerIterationOptional(psiIn):
#     print('From inside PowerIterationOptional:')
#     print('psiOrth: ', psiOrth)
#     print('weights: ', weights)
    psiOut = np.dot(A,psiIn)
    psiOut /= np.sqrt( np.dot(psiOut,psiOut) )
    psiOut -= ( np.dot(psiOut, psiOrth) )/( np.dot(psiOrth, psiOrth) ) * psiOrth

    psiOut /= np.sqrt( np.dot(psiOut,psiOut) )
    

    if np.linalg.norm(psiOut - psiIn) < np.linalg.norm(psiOut + psiIn):
        return psiOut - psiIn
    else:
        psiOut *= -1
        return psiOut - psiIn
    
# def andersonWrapper(F,xin,psiOrth=None, M=None, w0=None, tol_norm=None,f_tol=None,verbose=None):
#     
#     global psiOrth
#     psi2 = anderson(F,xin,M=M,w0=w0,tol_norm=tol_norm,f_tol=f_tol,verbose=verbose)
#     
#     return psi2

def test(N):
#     psiIn = np.random.rand(N)
    global A, psi1, weights
    weights = np.ones(N)
    A = np.random.rand(N,N)
    A = A+A.T
    
    ## First Test Power Iterations
    psi1 = np.random.rand(N)
    psi1 /= np.linalg.norm(psi1)
    residual=1
    count=1
    while ( (residual > 1e-7) and (count<10000) ):
        r = PowerIteration(psi1)
        psi1 += r
        residual = np.linalg.norm(r)
        print('Iteration %i: Residual = %1.2e' %(count,residual))
        count+=1
    eig1=np.dot( psi1, np.dot(A,psi1)) / np.dot(psi1,psi1)
    print('Rayleigh Quotient: ', eig1)
#     psi1 = np.copy(psiIn)
    print()
    print()
    psi2 = np.random.rand(N)
    psi2 /= np.linalg.norm(psi2)
    print(psi2)
    residual=1
    count=1
    while ( (residual > 1e-7) and (count<10000) ):
        r = PowerIteration2(psi2)
        psi2 += r
        residual = np.linalg.norm(r)
        print('Iteration %i: Residual = %1.2e' %(count,residual))
        count+=1
    eig2=np.dot( psi2, np.dot(A,psi2)) / np.dot(psi2,psi2)
    print('Rayleigh Quotient: ', eig2)
    
#     print('psiIn: ', psiIn)
    print()
    psiIn = np.random.rand(N)
    psiIn /= np.linalg.norm(psiIn)
    psi1 = anderson(PowerIteration,psiIn,M=10,w0=0.01,tol_norm=np.linalg.norm,f_tol=1e-7,verbose=True)
    print('Rayleigh Quotient: ', np.dot( psi1, np.dot(A,psi1)) / np.dot(psi1,psi1))
    print('Difference: ', np.dot( psi1, np.dot(A,psi1)) / np.dot(psi1,psi1)-eig1)
    print()
    print()
    
    global psiOrth
    psiOrth = np.copy(psi1)
     
    psiIn = np.random.rand(N)
    psiIn /= np.linalg.norm(psiIn)
    
    residual=1
    count=1
    
#     d = {psiOrth":psiOrth}

    # Preprocessing
    while ( (residual > 1e-1) and (count<10000) ):
        r = PowerIterationOptional(psiIn)
        psiIn += r
        residual = np.linalg.norm(r)
#         print('Iteration %i: Residual = %1.2e' %(count,residual))
        count+=1
#     print('Rayleigh Quotient: ', np.dot( psi2, np.dot(A,psi2)) / np.dot(psi2,psi2))
    
    
    print('Using Anderson for PowerIterationOptional...')
#     Fkw = {"psiOrth":psiOrth}
#     AndersonKW = {"M":10, "w0":0.01,"tol_norm":np.linalg.norm,"f_tol":1e-7,"verbose":True }
    psi2 = anderson(PowerIterationOptional,psiIn,M=10, w0=0.01,tol_norm=np.linalg.norm,f_tol=1e-7,verbose=True)
    print('Rayleigh Quotient: ', np.dot( psi2, np.dot(A,psi2)) / np.dot(psi2,psi2))
    print('Difference: ', np.dot( psi2, np.dot(A,psi2)) / np.dot(psi2,psi2)-eig2)
    print('Used %i iterations in preprocessing.' %count)
#     print('psi1 = ', psi1)
#     print('psi2 = ', psi2)
#     print('Overlap: ', np.dot(psi1,psi2))



# def greensIteration_FixedPoint(psiIn):
#     tree.totalIterationCount += 1
#     
# 
# #     sources = tree.extractPhi(m)
# #     targets = np.copy(sources)
# #     weights = np.copy(targets[:,5])
#     
#     
#             
#             
#     oldOrbitals[:,m] = np.copy(targets[:,3])
# 
#     if GIandersonMixing==True:
#         if firstInputWavefunction==True:
#             temp = np.append( oldOrbitals[:,m], tree.orbitalEnergies[m])
#             inputWavefunctions[:,0] = np.copy(temp) # fill first column of inputWavefunctions
# #                             inputEigenvalues[0] = tree.orbitalEnergies[m]
#             firstInputWavefunction=False
#         else:
#             if (greenIterationsCount-1-mixingStart)<mixingHistoryCutoff:
#                 temp = np.append( oldOrbitals[:,m], tree.orbitalEnergies[m])
#                 inputWavefunctions = np.concatenate( ( inputWavefunctions, np.reshape(np.copy(temp), (numberOfGridpoints+1,1)) ), axis=1)
#                 print('Concatenated inputWavefunction.  Now has shape: ', np.shape(inputWavefunctions))
# #                                 inputEigenvalues = np.concatenate( inputEigenvalues, tree.orbitalEnergies[m])
# #                                 print('Concatenated inputeEigenvalues.  Now has shape: ', np.shape(inputeEigenvalues))
#             else:
#                 print('Beyond mixingHistoryCutoff.  Replacing column ', (greenIterationsCount-1-mixingStart)%mixingHistoryCutoff)
# #                                 print('Shape of oldOrbitals[:,m]: ', np.shape(oldOrbitals[:,m]))
#                 temp = np.append( oldOrbitals[:,m], tree.orbitalEnergies[m])
#                 inputWavefunctions[:,(greenIterationsCount-1-mixingStart)%mixingHistoryCutoff] = np.copy(temp)
# #                                 inputEigenvalues[(greenIterationsCount-1-mixingStart)%mixingHistoryCutoff] = np.copy(tree.orbitalEnergies[m])
# 
# 
#     if symmetricIteration==False:
#         sources = tree.extractGreenIterationIntegrand(m)
#     elif symmetricIteration == True:
# #                     sources = tree.extractGreenIterationIntegrand_Deflated(m,orbitals,weights)
#         sources, sqrtV = tree.extractGreenIterationIntegrand_symmetric(m)
#     else: 
#         print("symmetricIteration variable not True or False.  What should it be?")
#         return
#     
#     targets = np.copy(sources)
# 
# 
# 
#     if tree.orbitalEnergies[m]<0: 
#         oldEigenvalue =  tree.orbitalEnergies[m] 
#         k = np.sqrt(-2*tree.orbitalEnergies[m])
#     
#         phiNew = np.zeros((len(targets)))
#         if subtractSingularity==0: 
#             print('Using singularity skipping')
#             gpuHelmholtzConvolution[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k) 
#         elif subtractSingularity==1:
#             if tree.orbitalEnergies[m] < 10.25: 
#                 
#                 
#                 if GPUpresent==False:
#                     print('Using Precompiled-C Helmholtz Singularity Subtract')
#                     numTargets = len(targets)
#                     numSources = len(sources)
# 
#                     sourceX = np.copy(sources[:,0])
# 
#                     sourceY = np.copy(sources[:,1])
#                     sourceZ = np.copy(sources[:,2])
#                     sourceValue = np.copy(sources[:,3])
#                     sourceWeight = np.copy(sources[:,4])
#                     
#                     targetX = np.copy(targets[:,0])
#                     targetY = np.copy(targets[:,1])
#                     targetZ = np.copy(targets[:,2])
#                     targetValue = np.copy(targets[:,3])
#                     targetWeight = np.copy(targets[:,4])
#                     
#                     phiNew = directSumWrappers.callCompiledC_directSum_HelmholtzSingularitySubtract(numTargets, numSources, k, 
#                                                                                                           targetX, targetY, targetZ, targetValue, targetWeight, 
#                                                                                                           sourceX, sourceY, sourceZ, sourceValue, sourceWeight)
#                     phiNew += 4*np.pi*targets[:,3]/k**2
# 
# 
#                     phiNew /= (4*np.pi)
#                 elif GPUpresent==True:
#                     if treecode==False:
#                         startTime = time.time()
#                         if symmetricIteration==False:
#                             gpuHelmholtzConvolutionSubractSingularity[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k) 
#                             convolutionTime = time.time()-startTime
#                             print('Using asymmetric singularity subtraction.  Convolution time: ', convolutionTime)
#                         elif symmetricIteration==True:
#                             gpuHelmholtzConvolutionSubractSingularitySymmetric[blocksPerGrid, threadsPerBlock](targets,sources,sqrtV,phiNew,k) 
#                             phiNew *= -1
#                             convolutionTime = time.time()-startTime
#                             print('Using symmetric singularity subtraction.  Convolution time: ', convolutionTime)
# 
#                         
#                     elif treecode==True:
#                         
#                         copyStart = time.time()
#                         numTargets = len(targets)
#                         numSources = len(sources)
# 
#                         sourceX = np.copy(sources[:,0])
# 
#                         sourceY = np.copy(sources[:,1])
#                         sourceZ = np.copy(sources[:,2])
#                         sourceValue = np.copy(sources[:,3])
#                         sourceWeight = np.copy(sources[:,4])
#                         
#                         targetX = np.copy(targets[:,0])
#                         targetY = np.copy(targets[:,1])
#                         targetZ = np.copy(targets[:,2])
#                         targetValue = np.copy(targets[:,3])
#                         targetWeight = np.copy(targets[:,4])
#                     
#                         copytime=time.time()-copyStart
# #                                         print('Time spent copying arrays for treecode call: ', copytime)
#                         potentialType=3
#                         kappa = k
#                         start = time.time()
#                         phiNew = treecodeWrappers.callTreedriver(numTargets, numSources, 
#                                                                        targetX, targetY, targetZ, targetValue, 
#                                                                        sourceX, sourceY, sourceZ, sourceValue, sourceWeight,
#                                                                        potentialType, kappa, treecodeOrder, theta, maxParNode, batchSize)
#                         print('Convolution time: ', time.time()-start)
#                         phiNew /= (4*np.pi)
#                     
#                     else: 
#                         print('treecode true or false?')
#                         return
#             else:
#                 print('Using singularity skipping because energy too close to 0')
#                 gpuHelmholtzConvolution[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k)
#         else:
#             print('Invalid option for singularitySubtraction, should be 0 or 1.')
#             return
#         
#         
#         """ Method where you dont compute kinetics, from Harrison """
#         
#         # update the energy first
#         
# 
#         if ( (gradientFree==True) and (SCFcount>-1) and (freezeEigenvalue==False) ):
#             
#             psiNewNorm = np.sqrt( np.sum( phiNew*phiNew*weights))
#             
#             if symmetricIteration==False:
#                 tree.importPhiNewOnLeaves(phiNew)
# #                                 print('Not updating energy, just for testing Steffenson method')
#                 tree.updateOrbitalEnergies_NoGradients(m, newOccupations=False)
#                 orbitals[:,m] = np.copy(phiNew)
#             elif symmetricIteration==True:
# #                                 tree.importPhiNewOnLeaves(phiNew/sqrtV)
# #                                 tree.updateOrbitalEnergies_NoGradients(m, newOccupations=False)
# 
# 
#                 # import phiNew and compute eigenvalue update
#                 tree.importPhiNewOnLeaves(phiNew)
#                 tree.updateOrbitalEnergies_NoGradients(m, newOccupations=False, symmetric=True)
#                 
#                 # Import normalized psi*sqrtV into phiOld
#                 phiNew /= np.sqrt( np.sum(phiNew*phiNew*weights ))
#                 tree.setPhiOldOnLeaves_symmetric(phiNew)
#                 
#                 
#                 orbitals[:,m] = np.copy(phiNew/sqrtV)
#             
#             n,k = np.shape(orbitals)
#             orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,weights,m, n, k)
#             orbitals[:,m] = np.copy(orthWavefunction)
#             tree.importPhiOnLeaves(orbitals[:,m], m)
#             
# 
# 
#             if greenIterationsCount==1:
#                 eigenvalueHistory = np.array(tree.orbitalEnergies[m])
#             else:
#                 eigenvalueHistory = np.append(eigenvalueHistory, tree.orbitalEnergies[m])
#             print('eigenvalueHistory: \n',eigenvalueHistory)
#             
#             
#             print('Orbital energy after Harrison update: ', tree.orbitalEnergies[m])
#             
# 
#         elif ( (gradientFree==False) or (SCFcount==-1) ):
# 
#             # update the orbital
#             if symmetricIteration==False:
#                 orbitals[:,m] = np.copy(phiNew)
#             if symmetricIteration==True:
#                 orbitals[:,m] = np.copy(phiNew/sqrtV)
#                 
#             n,k = np.shape(orbitals)
#             orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,weights,m, n, k)
#             orbitals[:,m] = np.copy(orthWavefunction)
#             tree.importPhiOnLeaves(orbitals[:,m], m)
#             
# #                             tree.importPhiOnLeaves(orbitals[:,m], m)
# #                             tree.orthonormalizeOrbitals(targetOrbital=m)
#             
#             tree.updateOrbitalEnergies(sortByEnergy=False, targetEnergy=m)
# 
#             
#         else:
#             print('Not updating eigenvalue.  Is that intended?')
# #                             print('Invalid option for gradientFree, which is set to: ', gradientFree)
# #                             print('type: ', type(gradientFree))
#             
#             orbitals[:,m] = np.copy(phiNew)
#             n,k = np.shape(orbitals)
#             orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,weights,m, n, k)
#             orbitals[:,m] = np.copy(orthWavefunction)
#             
#             residualVector = orthWavefunction - psiIn
#             tree.importPhiOnLeaves(orbitals[:,m], m)
#             tree.setPhiOldOnLeaves(m)
#             
#            
#             eigenvalueHistory = np.append(eigenvalueHistory, tree.orbitalEnergies[m])
#             
# 
#         newEigenvalue = tree.orbitalEnergies[m]
#         
#         
#         
# 
# #                         if newEigenvalue > tree.gaugeShift:
#         if newEigenvalue > 0.0:
#             if greenIterationsCount < 10:
#                 tree.orbitalEnergies[m] = tree.gaugeShift-0.5
#                 GIandersonMixing=False
#                 print('Setting energy to gauge shift - 0.5 because new value was positive.')
#         
#             else:
#                 tree.orbitalEnergies[m] = tree.gaugeShift
#                 if greenIterationsCount % 10 == 0:
#                     tree.scrambleOrbital(m)
#                     tree.orthonormalizeOrbitals(targetOrbital=m)
#                     GIandersonMixing=False
#                     print("Scrambling orbital because it's been a multiple of 10.")
# 
#     else:
#         print('Orbital %i energy greater than zero.  Not performing Green Iterations for it...' %m)
#         
#     
# 
#     tempOrbital = tree.extractPhi(m)
# 
#     orbitals[:,m] = np.copy( tempOrbital[:,3] )
#     if symmetricIteration==False:
#         normDiff = np.sqrt( np.sum( (orbitals[:,m]-oldOrbitals[:,m])**2*weights ) )
#     elif symmetricIteration==True:
#         normDiff = np.sqrt( np.sum( (orbitals[:,m]*sqrtV-oldOrbitals[:,m]*sqrtV)**2*weights ) )
#     eigenvalueDiff = abs(newEigenvalue - oldEigenvalue)
#     
#     
# 
#     residuals[m] = normDiff
#     orbitalResidual = np.copy(normDiff)
#     
#     
# 
#     print('Orbital %i error and eigenvalue residual:   %1.3e and %1.3e' %(m,tree.orbitalEnergies[m]-referenceEigenvalues[m]-tree.gaugeShift, eigenvalueDiff))
#     print('Orbital %i wavefunction residual: %1.3e' %(m, orbitalResidual))
#     print()
#     print()
# 
# 
# 
#     header = ['targetOrbital', 'Iteration', 'orbitalResiduals', 'energyEigenvalues', 'eigenvalueResidual']
# 
#     myData = [m, greenIterationsCount, residuals,
#               tree.orbitalEnergies-tree.gaugeShift, eigenvalueDiff]
# 
#     if not os.path.isfile(greenIterationOutFile):
#         myFile = open(greenIterationOutFile, 'a')
#         with myFile:
#             writer = csv.writer(myFile)
#             writer.writerow(header) 
#         
#     
#     myFile = open(greenIterationOutFile, 'a')
#     with myFile:
#         writer = csv.writer(myFile)
#         writer.writerow(myData)
#     
#     
#     residualRatio = orbitalResidual/oldOrbitalResidual
#     eigenvalueResidualRatio = eigenvalueDiff/previousEigenvalueDiff
#     print()
#     print('Wavefunction Relative Residual =          ', residualRatio)
#     print('Wavefunction Previous relative residual = ', previousResidualRatio)
#     print()
#     print('Eigenvalue Relative Residual =          ', eigenvalueResidualRatio)
#     print('Eigenvalue Previous relative residual = ', previousEigenvalueResidualRatio)
#     print()
#                     
#     
#     return residualVector

if __name__=="__main__":
    
    test(10)