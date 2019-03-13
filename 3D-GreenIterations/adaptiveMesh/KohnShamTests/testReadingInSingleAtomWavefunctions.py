'''
Created on Jul 25, 2018

@author: nathanvaughn
'''
import unittest
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
sys.path.append('../src/dataStructures')
sys.path.append('../src/utilities')

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
        print('rho[0] = ', data[0,1])
        plt.semilogy(data[:,0],data[:,1],label='Density')
#         plt.semilogy(data[:,0],np.sqrt(data[:,1]),label='sqrt(Density)')
#         plt.plot(data[:,0],np.sqrt(data[:,1])*(1+1/data[:,0]),label='sqrt(Density)')
#         plt.plot(data[:,0],(data[:,1])*(1+1/data[:,0]),label='(Density)(1+1/r)')
#         plt.plot(data[:,0],(data[:,1])*(data[:,0]**2),label='(Density)*r**2')
#         plt.plot(data[:,0], data[0,1]*np.exp(-2*atomicNumber*data[:,0]), 'r-')
#         plt.plot(data[:,0], data[0,1]*np.exp(-2*np.sqrt(2*0.3)*atomicNumber*data[:,0]), 'k-')
        plt.legend()
        print(data[:,0])
        r=data[:,0]
        density=data[:,1]
        print('\n\n\n')
        print(data[:,1])
        left = 0
        right = 0
        for i in range(len(density)-1):
            left += density[i]*(r[i+1]-r[i])
            right += density[i+1]*(r[i+1]-r[i])
        trap = 1/2*(left+right)
        print('Left = ', left)    
        print('Right = ', right)
        print('Integrated density: ', trap)
        
        
#         plt.figure() 
         
        for orbital in os.listdir(AtomicDataPath):
            if orbital[:3]=='psi':
#             if orbital[:5]=='psi32':
                print(orbital)
                data = np.genfromtxt(AtomicDataPath+orbital)
                plt.semilogy(data[:,0],np.abs(data[:,1]),label=orbital[:-4])
#                 plt.semilogy(data[:,0],np.abs(data[:,1]**2),label=orbital[:-4]+' squared')
#                 plt.plot(data[:,0],np.sign(data[-1,1])*data[:,1],label=orbital[:-4])
#         xi = np.sqrt(2*0.2)
#         plt.plot(data[:,0], np.sqrt(xi**3/np.pi) *np.exp(-xi*atomicNumber*data[:,0]), 'r-')
#                 
#         
# #         data0 = np.genfromtxt(AtomicDataPath+'psi10.inp')
# #         r0 = data0[:,0]
# #         psi0 = data0[:,1]
# #         
# #         data1 = np.genfromtxt(AtomicDataPath+'psi20.inp')
# #         r1 = data1[:,0]
# #         psi1 = data1[:,1]
# #         [r0, phi0] = np.genfromtxt(AtomicDataPath+'psi10.inp')
# #         [r1, phi1] = np.genfromtxt(AtomicDataPath+'psi20.inp')
#         
# #         plt.figure()
# #         plt.plot(r0,psi0,'b',label='psi10')
# #         plt.plot(r1,psi1,'g',label='psi20')
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