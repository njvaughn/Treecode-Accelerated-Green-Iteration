'''
Created on Jul 25, 2018

@author: nathanvaughn
'''
import unittest
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from AtomStruct import Atom
from TreeStruct_CC import Tree

class Test(unittest.TestCase):

#     @unittest.skip('Skipping the plotting of radial data')
    def testReadingRadialData(self):
        atomicNumber = 8
        AtomicDataPath = '/Users/nathanvaughn/AtomicData/allElectron/z'+str(atomicNumber)+'/singleAtomData/'
        print(AtomicDataPath)
        print(os.listdir(AtomicDataPath))
        
        plt.figure()
        data = np.genfromtxt(AtomicDataPath+'density.inp')
        plt.plot(data[:,0],data[:,1],label='Density')
        for orbital in os.listdir(AtomicDataPath):
            if orbital[:3]=='psi':
#             if orbital[:5]=='psi32':
                print(orbital)
                data = np.genfromtxt(AtomicDataPath+orbital)
                plt.plot(data[:,0],data[:,1],label=orbital[:-4])
        
#         data0 = np.genfromtxt(AtomicDataPath+'psi10.inp')
#         r0 = data0[:,0]
#         psi0 = data0[:,1]
#         
#         data1 = np.genfromtxt(AtomicDataPath+'psi20.inp')
#         r1 = data1[:,0]
#         psi1 = data1[:,1]
#         [r0, phi0] = np.genfromtxt(AtomicDataPath+'psi10.inp')
#         [r1, phi1] = np.genfromtxt(AtomicDataPath+'psi20.inp')
        
#         plt.figure()
#         plt.plot(r0,psi0,'b',label='psi10')
#         plt.plot(r1,psi1,'g',label='psi20')
        plt.legend()
        plt.xlabel('radius')
        plt.show()
        
    @unittest.skip('Skipping the printing of the interpolators')
    def testSettingUpAtomInterpolators(self):
        
        LiAtom = Atom(0,0,0,3)
        LiAtom.orbitalInterpolators()
        print(LiAtom.interpolators)
        print(LiAtom.interpolators['psi10'])
   
    @unittest.skip('Skipping the plotting of radial data')     
    def testSettingUpCell(self):
        tree = Tree(-2,1,1,-2,1,1,-2,1,1,nElectrons=10,nOrbitals=5,
                 coordinateFile='../src/utilities/molecularConfigurations/berylliumAtom.csv')
        
        tree.initializeFromAtomicData()
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testReadingRadialData']
    unittest.main()