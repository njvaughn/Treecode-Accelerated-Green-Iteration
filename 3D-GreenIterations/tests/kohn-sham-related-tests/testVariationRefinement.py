'''
Created on Jul 25, 2018

@author: nathanvaughn
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


import os
import sys
sys.path.append('../src/dataStructures')
sys.path.append('../src/utilities')

from AtomStruct import Atom
from TreeStruct_CC import Tree


def refine_singleWavefunction(atomicNumber, targetWavefunction, rmax, epsilon1, epsilon2, epsilon3):
    
    ### Extract radial wavefunction data
    AtomicDataPath = '/Users/nathanvaughn/AtomicData/allElectron/z'+str(atomicNumber)+'/singleAtomData/'
    print(AtomicDataPath)
    print(os.listdir(AtomicDataPath))
    data = np.genfromtxt(AtomicDataPath+targetWavefunction+'.inp')
    
    r = data[:,0]
    wavefunction = data[:,1]
    
    
    ### Generate interpolator
    interpolator = interp1d(data[:,0],data[:,1],fill_value='extrapolate')

    
    mesh = np.linspace(0,rmax,8)
    wave = interpolator(mesh)
    
    print(wave)
    
    absVarCounter=0
    relVarCounter=0
    integralPsiCounter=0
    needToRefine=True
    while needToRefine==True:
        meshB = []
        meshAbs = []
        meshRel = []
    
        for i in range(len(mesh)-1):
            meshB.append(mesh[i])
            
            absVariation = np.abs(wave[i+1]-wave[i])
            relVariation = np.abs(wave[i+1]-wave[i]) / (  (np.abs(wave[i+1])+np.abs(wave[i]))/2  )
            integralPsi = (np.abs(wave[i+1])+np.abs(wave[i]))/2 * (mesh[i+1]-mesh[i]) # trapezoid rule over interval
                   
            midpoint=(mesh[i]+mesh[i+1])/2                                         
            if ( (absVariation > epsilon1) ):
                print('Refining at %e due to absVar' %midpoint)
                absVarCounter+=1
                meshAbs.append( midpoint)
                meshB.append( midpoint)
            elif (relVariation > epsilon2):
                print('Refining at %e due to relVar' %midpoint)
                relVarCounter+=1
                meshRel.append( midpoint)
                meshB.append( midpoint)
                
            elif (integralPsi > epsilon3):
                print('Refining at %e due to integralPsi' %midpoint)
                integralPsiCounter+=1
                meshB.append( midpoint)
    
            
            
            if i==len(mesh)-2:
                meshB.append(mesh[i+1])
        
        print('Length of previous mesh:   ', len(mesh))      
        print('Length of new mesh:        ', len(meshB))
        print('absVarCounter:             ', absVarCounter)
        print('relVarCounter:             ', relVarCounter)
        print('integralPsiCounter:        ', integralPsiCounter)
        print()
        
        
                
        waveB = interpolator(meshB)
        waveAbs = interpolator(meshAbs)
        waveRel = interpolator(meshRel)
    
        
        if len(meshB)==len(mesh):
            needToRefine=False
        mesh = np.copy(meshB)
        wave = np.copy(waveB)
        
        
    plt.figure()
    plt.plot(r,wavefunction,'k-')
    plt.plot(mesh,wave,'go')
    plt.plot(mesh,np.zeros_like(mesh),'kx')
    plt.title(targetWavefunction + ', epsilon1 = %1.2f, epsilon2 = %1.2f, epsilon3 = %1.2f' %(epsilon1, epsilon2,epsilon3) )
    plt.show()


def refine_anyWavefunction(atomicNumber, targetWavefunctions, rmax, epsilon1, epsilon2):
    
    nWavefunctions = len(targetWavefunctions)
    print('nWavefunctions = ', nWavefunctions)
    ### Extract radial wavefunction data
    AtomicDataPath = '/Users/nathanvaughn/AtomicData/allElectron/z'+str(atomicNumber)+'/singleAtomData/'
    print(AtomicDataPath)
    print(os.listdir(AtomicDataPath))
    mesh = np.linspace(0,rmax,2)
    interpolators={}
    
    for i in range(nWavefunctions):
        data = np.genfromtxt(AtomicDataPath+targetWavefunctions[i]+'.inp')
    
        r = data[:,0]
        wavefunction = data[:,1]
        
        
        ### Generate interpolator
        interpolators[targetWavefunctions[i]] = interp1d(data[:,0],data[:,1],fill_value='extrapolate')
    
    data = np.genfromtxt(AtomicDataPath+'density.inp')
    interpolators['density'] = interp1d(data[:,0],data[:,1],fill_value='extrapolate')

    

    
    absVarCounter=0
    relVarCounter=0
    needToRefine=True
    while needToRefine==True:
        lenMesh = len(mesh)
#         meshAbs = []
#         meshRel = []
        density = interpolators['density'](mesh)
        meshB = []
        for i in range(len(mesh)-1):
            meshB.append(mesh[i])
            
            maxDensity = max(density[i+1], density[i])
            minDensity = min(density[i+1], density[i])
            relVariation = (maxDensity - minDensity) / maxDensity
                   
            midpoint=(mesh[i]+mesh[i+1])/2                                         

                
            if (relVariation > epsilon2):
                print('Refining at %e due to density relVar' %midpoint)
                relVarCounter+=1
#                 meshRel.append( midpoint)
                meshB.append( midpoint)
                
            if i==len(mesh)-2:
                meshB.append(mesh[i+1])
            
        print('Length of previous mesh:   ', len(mesh))      
        print('Length of new mesh:        ', len(meshB))
        print()
        mesh = np.copy(meshB)
                
        for j in range(nWavefunctions):
            print('Refining for wavefunction ', targetWavefunctions[j])
            wave = interpolators[targetWavefunctions[j]](mesh)
            meshB = []

    
            for i in range(len(mesh)-1):
                meshB.append(mesh[i])
                
                absVariation = np.abs(wave[i+1]-wave[i])
#                 relVariation = np.abs(wave[i+1]-wave[i]) / (  (np.abs(wave[i+1])+np.abs(wave[i]))/2  )
                       
                midpoint=(mesh[i]+mesh[i+1])/2                                         
                if ( (absVariation > epsilon1) ):
                    print('Refining at %e due to absVar' %midpoint)
                    absVarCounter+=1
#                     meshAbs.append( midpoint)
                    meshB.append( midpoint)
                    
#                 elif (relVariation > epsilon2):
#                     print('Refining at %e due to relVar' %midpoint)
#                     relVarCounter+=1
#                     meshRel.append( midpoint)
#                     meshB.append( midpoint)
        
                
                
                if i==len(mesh)-2:
                    meshB.append(mesh[i+1])
            
            print('Length of previous mesh:   ', len(mesh))      
            print('Length of new mesh:        ', len(meshB))
            print()
            mesh = np.copy(meshB)
            
#         print('absVarCounter:             ', absVarCounter)
#         print('relVarCounter:             ', relVarCounter)
#         print()
        
        
                
#         waveB = interpolator(meshB)
#         waveAbs = interpolator(meshAbs)
#         waveRel = interpolator(meshRel)
    
        
        if len(meshB)==lenMesh:
            needToRefine=False
        mesh = np.copy(meshB)
#         wave = np.copy(waveB)
        
        
    plt.figure()
#     plt.plot(r,wavefunction,'k-')
#     plt.plot(mesh,wave,'go')
    plt.plot(mesh,np.zeros_like(mesh),'kx')
    plt.title('Applied to all wavefunctions: epsilon1 = %f, epsilon2 = %f' %(epsilon1, epsilon2) )
    for j in range(nWavefunctions):
        wave = interpolators[targetWavefunctions[j]](mesh)
        plt.plot(mesh, wave, 'o-')
#         plt.plot(mesh, wave, 'o-')
    density = interpolators['density'](mesh)
    plt.plot(mesh, density, 'o-')
    plt.show()
    


def refine_densityRange(atomicNumber, targetWavefunction, rmax, maxDepth):
    
    ### Extract radial wavefunction data
    AtomicDataPath = '/Users/nathanvaughn/AtomicData/allElectron/z'+str(atomicNumber)+'/singleAtomData/'
    print(AtomicDataPath)
    print(os.listdir(AtomicDataPath))
    
    ## Density
    data = np.genfromtxt(AtomicDataPath+targetWavefunction+'.inp')
    r_density = data[:,0]
    density = data[:,1]
    interpolator = interp1d(data[:,0],np.abs(data[:,1]),fill_value='extrapolate')
    
    ## Wavefunction
    data = np.genfromtxt(AtomicDataPath+'psi10.inp')
    interpolatorPsi10 = interp1d(data[:,0],data[:,1],fill_value='extrapolate')
    
    data = np.genfromtxt(AtomicDataPath+'psi20.inp')
    interpolatorPsi20 = interp1d(data[:,0],data[:,1],fill_value='extrapolate')
    
    data = np.genfromtxt(AtomicDataPath+'psi21.inp')
    interpolatorPsi21 = interp1d(data[:,0],data[:,1],fill_value='extrapolate')

    
    mesh = np.linspace(0,rmax,8)
    wave = interpolator(mesh)
    maxDensity = interpolator(0.0)
    print(wave)
  
    needToRefine=True
    depth=1
    while needToRefine==True:
        meshB = []
 
    
        for i in range(len(mesh)-1):
            meshB.append(mesh[i])
            
#             absVariation = np.abs(wave[i+1]-wave[i])
#             relVariation = np.abs(wave[i+1]-wave[i]) / (  (np.abs(wave[i+1])+np.abs(wave[i]))/2  )
#             integralPsi = (np.abs(wave[i+1])+np.abs(wave[i]))/2 * (mesh[i+1]-mesh[i]) # trapezoid rule over interval
                   
            midpoint=(mesh[i]+mesh[i+1])/2  
            midpointDensity = interpolator(midpoint)
#             midpointDensity = max(interpolator(mesh[i]), interpolator(mesh[i+1]) )                                     
            if ( (midpointDensity/maxDensity > (depth)/maxDepth) ):
                print('Refining at %e due to density' %midpoint)
                meshB.append( midpoint)
            else:
#                 sqrtDensity = max( np.sqrt(interpolator(mesh[i])), np.sqrt(interpolator(mesh[i+1]) ) )                                     
                sqrtDensity = np.sqrt(interpolator(midpoint))                                     
                if ( (sqrtDensity/np.sqrt(maxDensity) > (depth)/maxDepth) ):
                    print('Refining at %e due to sqrt(density)=%f' %(midpoint,sqrtDensity))
                    meshB.append( midpoint)
            
    
            
            
            if i==len(mesh)-2:
                meshB.append(mesh[i+1])
        
        print('Length of previous mesh:   ', len(mesh))      
        print('Length of new mesh:        ', len(meshB))
        print()
        
        
                
        waveB = interpolator(meshB)
    
        
        if len(meshB)==len(mesh):
            needToRefine=False
        mesh = np.copy(meshB)
        wave = np.copy(waveB)
        depth+=1
        
        
    plt.figure()
    plt.plot(r_density,density,'k-')
    plt.plot(r_density,np.sqrt(density),'b-')
    plt.plot(mesh,wave,'go')
    plt.plot(mesh,np.sqrt(wave),'bo')
    plt.plot(mesh,np.zeros_like(mesh),'kx')
    plt.title(targetWavefunction + ', maxDepth = %i' %(maxDepth) )
    
    psi10 = interpolatorPsi10(mesh)
    psi20 = interpolatorPsi20(mesh)
    psi21 = interpolatorPsi21(mesh)
    plt.figure()
#     plt.plot(r_density,density,'k-')
#     plt.plot(mesh,wave,'go')
    plt.plot(mesh,psi10,'o')
    plt.plot(mesh,psi20,'o')
    plt.plot(mesh,psi21,'o')
#     plt.plot(mesh,np.zeros_like(mesh),'kx')
#     plt.title(targetWavefunction + ', epsilon1 = %1.2f, epsilon2 = %1.2f, epsilon3 = %1.2f' %(epsilon1, epsilon2,epsilon3) )
    plt.title(targetWavefunction + ', maxDepth = %i' %(maxDepth) )
    
    
    plt.show()


if __name__ == "__main__":
    
    
    refine_densityRange(8, 'density', 20, 13)
#     refine_densityRange(8, 'psi10', 20, 13)
    
    
    
#     refine(8,'density',10,0.9,2.0)
#     refine_anyWavefunction(8,['psi10', 'psi21'],15,0.5,1.0)
#     refine_anyWavefunction(8,['psi10','psi20','psi21'],20,10,0.99)
    
#     refine_singleWavefunction(8,'psi21',15,0.15,1000.0, 500)
    
    