'''
Created on Aug 28, 2018

@author: nathanvaughn
'''
import unittest
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import broyden1, anderson, brentq

def F(fermiEnergy, nElectrons, orbitalEnergies, sigma):
    exponentialArg = (orbitalEnergies-fermiEnergy)/sigma
#     print(exponentialArg)
#     for i in range(len(exponentialArg)):
#         if exponentialArg[i] > 50:
#             exponentialArg[i] = 50
#         if exponentialArg[i] < -50:
#             exponentialArg[i] = -50
#     print(exponentialArg)
    temp = 1/(1+np.exp( exponentialArg ) )
#     print(temp)
    return nElectrons - 2 * np.sum(temp)
#     return  2 * np.sum(temp)



def computeFermiEnergy(nElectrons, orbitalEnergies, sigma):
    pass

class Test(unittest.TestCase):
    
    def G(self,fermiEnergy):
        exponentialArg = (self.orbitalEnergies-fermiEnergy)/self.sigma
    #     for i in range(len(exponentialArg)):
    #         if exponentialArg[i] > 50:
    #             exponentialArg[i] = 50
    #         if exponentialArg[i] < -50:
    #             exponentialArg[i] = -50
        temp = 1/(1+np.exp( exponentialArg ) )
        return self.nElectrons - 2 * np.sum(temp)

    @classmethod
    def setUpClass(self):
        
#         [ 15.71777254   1.64701124   1.23317729   1.23317729   1.23317729]
# Orbital Potential Energy:  [-25.68467571  -2.15056764  -1.43434912  -1.43434912  -1.43434912]


        KB = 8.6173303e-5/27.211386  # Boltzmann factor in Hartree/Kelvin
        self.nElectrons = 14 # number of electrons
#         self.nOrbitals = 4
#         self.orbitalEnergies = np.array([-5, -1, -0.3, -0.001])
#         self.orbitalEnergies = np.array([-9.9669, -0.5035, -0.20122, -0.20122, -0.20122])
#         self.orbitalEnergies = np.array([-3.85171675  -0.85216851  -0.38889536  -0.38889536  -0.038889536])
#         self.orbitalEnergies = np.array([-3.85171675  -0.85216851  -0.38889536  -0.38889536  -0.038889536])

        self.orbitalEnergies = np.array([-18.71236259, -10.01843142,  -0.72258439,  -0.6591846,   -0.41691493,
                                         -0.35603646,  -0.35603646,  -0.27839483,  -0.27839483,  -0.11320702])
        
        self.T = 100
        self.sigma = self.T*KB
    
#     @unittest.skip('Not plotting right now')
    def testSweepFermiEnergy(self):
#         print('Fermi Energy -0.3, ', F(-0.3,self.N,self.orbitalEnergies,self.sigma))
#         print('Fermi Energy -0.2, ', F(-0.2,self.N,self.orbitalEnergies,self.sigma))
#         print('Fermi Energy -0.15, ', F(-0.15,self.N,self.orbitalEnergies,self.sigma))
#         print('Fermi Energy -0.125, ', F(-0.125,self.N,self.orbitalEnergies,self.sigma))
#         print('Fermi Energy -0.1125, ', F(-0.1125,self.N,self.orbitalEnergies,self.sigma))
#         print('Fermi Energy -0.100001, ', F(-0.100001,self.N,self.orbitalEnergies,self.sigma))
#         print('Fermi Energy -0.1, ', F(-0.1,self.N,self.orbitalEnergies,self.sigma))
#         print('Fermi Energy -0.099999, ', F(-0.099999,self.N,self.orbitalEnergies,self.sigma))
#         print('Fermi Energy -0.05, ', F(-0.05,self.N,self.orbitalEnergies,self.sigma))
        
        eF = np.linspace(-12,1,50000)
        electrons = np.zeros_like(eF)
        for i in range(len(eF)):
#             electrons[i] = self.nElectrons - self.G(eF[i])
            electrons[i] = self.nElectrons - F(eF[i], self.nElectrons, self.orbitalEnergies, self.sigma)
        plt.figure()
#         plt.plot(eF,electrons,'.')
        plt.plot(eF,electrons)
        plt.xlabel('Fermi Energy')
        plt.ylabel('Number of Electrons')
        plt.title('14 Electrons Occupying the Initial Orbitals for Carbon Monoxide')
#         plt.title('4 electrons occupying orbitals with energies [-5, -1, -0.003, -0.001] at T=%i' %self.T)
#         print(errors)
        plt.show()
        
#     @unittest.skip('just plotting right now') 
    def testFindRoot(self):
#         eF = broyden1(self.G, -5.3, f_tol=1e-7)
#         eF = anderson(self.G, -3.5, f_tol=1e-7,verbose=True)
        eF = brentq(self.G, -17, 0)
        print('Fermi energy: ', eF)
        exponentialArg = (self.orbitalEnergies-eF)/self.sigma
        occupations = 1/(1+np.exp( exponentialArg ) )
        print('Occupations: ', occupations)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testFermiEnergy']
    unittest.main()