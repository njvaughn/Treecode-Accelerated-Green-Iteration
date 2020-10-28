import numpy as np
import csv


def checkC60spacing():
    X,Y,Z = np.loadtxt("C60.csv", usecols=(0,1,2), unpack=True,delimiter=',',dtype=float)
    numAtoms=len(X)
    
    minDistance = 1e10*np.ones(numAtoms)
    for i in range(numAtoms):
        for j in range(numAtoms):
            if i!=j:
                dx = X[i]-X[j]
                dy = Y[i]-Y[j]
                dz = Z[i]-Z[j]
                dist = np.sqrt( dx*dx + dy*dy + dz*dz)
                
                minDistance[i] = min(dist,minDistance[i])
         
    assert np.any(abs(minDistance-1.36705) < 1e-5), "Warning, not all atoms of C60 separated by 1.36705."  
    
def checkNuclearRepulsionEnergy():
    X,Y,Z = np.loadtxt("C60.csv", usecols=(0,1,2), unpack=True,delimiter=',',dtype=float)
    numAtoms=len(X)
    
    minDistance = 1e10*np.ones(numAtoms)
    nuclearEnergy=0.0
    for i in range(numAtoms):
        for j in range(numAtoms):
            if i!=j:
                dx = X[i]-X[j]
                dy = Y[i]-Y[j]
                dz = Z[i]-Z[j]
                dist = np.sqrt( dx*dx + dy*dy + dz*dz)
                
                nuclearEnergy += (1/2) * 4 * 4 / dist  # net charge of 4 for carbon atoms
                
#                 minDistance[i] = min(dist,minDistance[i])
    print("nuclear repulsion energy = %1.10f Ha " %nuclearEnergy)   
    assert abs(nuclearEnergy-7044.040395)/7044.040395 < 1e-8, "Warning, nuclear repulsion energy for C60 is not 7044.040395." 

if __name__=="__main__":
    
    checkC60spacing()
    checkNuclearRepulsionEnergy()
      