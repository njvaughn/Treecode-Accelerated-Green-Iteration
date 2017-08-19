"""

Bound State class should inherit the model for a given potential

"""
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy as sp
from scipy import special as spec
from scipy.optimize import fsolve
pi = np.pi

class Poschl_Teller:
    
    def potential(self,x, D):
        return -D*(D+1)/2*(1/np.cosh(x))**2

    def true_energy(self,n, D):
        if n < D:
            return -(D-n)**2/2
        else:
            print('This well does not have that many bound states.')


    def true_wave(self, n, D, grid, plot = 'no'):
        if plot == 'no':
            return spec.lpmv(D-n, D, np.tanh(grid))
        elif plot == 'yes':
            wave = spec.lpmv(D-n, D, np.tanh(grid))
            plt.plot(grid, wave)
            plt.title('Analytic Wave: n = %g' %n)
            plt.show()

            return wave
        else:
            print('error in true wave plot input')
            


class Morse:

    def potential(self,x, D):
        return D*(np.exp(-2*x/np.sqrt(2*D)) -2*np.exp(-x/np.sqrt(2*D)))

    def true_energy(self,n, D):
        return -D + (n+1/2) - 1/(4*D)*(n+1/2)**2       

    def true_wave(self, n, D, grid, plot = 'no'):
        omega = 1/np.sqrt(2*D)
        lmbda = 2*D
        z = 2*lmbda*np.exp(-omega*grid)       
        alpha = 2*lmbda - 2*n - 1   
        N = np.sqrt((math.factorial(n)*(2*lmbda - 2*n - 1))/(sp.special.gamma(2*lmbda-n)))
        L = spec.eval_genlaguerre(n,alpha,z)
        wave =  N*z**(lmbda-n-1/2)*np.exp(-z/2)*L
        if plot == 'no':
            return wave
        elif plot == 'yes':
            plt.plot(grid, wave)
            plt.title('Analytic Wave: n = %g' %n)
            plt.show()
            return wave
        else:
            print('error in true wave plot input')



class Square:
    """
    a is the well width, D is the well depth.  This a = 5*D relation
    is free to be changed, simply did it this way so that the potential
    is defined by a single parameter, D, just like other models.
    """
    
##    a = 5*D 
    
    def potential(self,x, D):
        a = 5*D
        return np.piecewise(x, [np.abs(x) > a, np.abs(x) <= a], [0, -D])

    def f_even(self,E,D):
        a = 5*D
        ell = np.sqrt(2*(D+E))
        z = ell*a
        z0 = a*np.sqrt(2*D)
        return np.tan(z)-np.sqrt((z0/z)**2-1)

    def f_odd(self,E,D):
        a = 5*D
        ell = np.sqrt(2*(D+E))
        z = ell*a
        z0 = a*np.sqrt(2*D)
        return np.tan(z)+ 1/np.sqrt((z0/z)**2-1)



    def Energies(self,E_start,E_end,D):
        a = 5*D
        # divide based on the continuity of tangent(z)
        E_break = []
        E_true  = []
        p = 0;
        while p < 20:
            E_test = -D + 1/(2*a**2)*((p/2)*pi)**2
            p += 1
            if E_test < E_end:
                break
            if E_test > E_start:
                break
            E_break.append(  E_test )
        E_break.insert(0,E_end)
        E_break.append(E_start)
        E_b = np.array(E_break)
        for p in range(1,len(E_b)):
            E_test = fsolve(self.f_even, (E_b[p-1]+E_b[p])/2,D)
            if E_test > E_b[p-1] and E_test < E_b[p]:
                ell = np.sqrt(2*(D+E_test))
                z = ell*a
                if np.tan(z) > 0:
                    E_true.append(E_test)
        for p in range(1,len(E_b)):
            E_test = fsolve(self.f_odd, (E_b[p-1]+E_b[p])/2,D)
            if E_test > E_b[p-1] and E_test < E_b[p]:
                ell = np.sqrt(2*(D+E_test))
                z = ell*a
                if np.tan(z) < 0:
                    E_true.append(E_test)
        E_true = np.sort(E_true,axis=0)
        return np.array(E_true)


    def true_energy(self,n, D):
        a = 5*D
        warnings = []
        try:
            E_list =  self.Energies(-D/1e7,-D,D)
        except Warning as w:
            warning.append(w)
        try:
            return E_list[n]
        except Exception as e:
            print(e)
            print('This well does not have that many bound states.')


    def true_wave(self,n,D, grid, plot = 'no'):
        a = 5*D
        E_true = self.Energies(self,-D/1e7,-D,D)
        E = E_true[n]
        ell = np.sqrt(2*(D+E))
        k =np.sqrt(2*(-E))   
        xvec = grid
        wave = np.zeros(len(xvec))
        D = np.exp(-k*a)
        if n%2 ==0:
            B = np.cos(ell*a)
        else:
            B = np.sin(ell*a)
        for i in range(len(grid)):
            if np.abs(xvec[i]) < a:
                if n%2 ==0:
                    wave[i] = D*np.cos(ell*xvec[i])
                if n%2 ==1:
                    wave[i] = D*np.sin(ell*xvec[i])
            if np.abs(xvec[i]) >= a:
                if xvec[i]>0:
                    wave[i] = B*np.exp(-k*xvec[i])
                if xvec[i]<0:
                    if n%2 ==0:
                        wave[i] = B*np.exp(k*xvec[i])
                    if n%2 ==1:
                        wave[i] = -B*np.exp(k*xvec[i])
        A = np.dot(wave,wave)
        wave = wave/np.sqrt(A)

        if plot == 'no':
            return wave
        elif plot == 'yes':
            plt.plot(grid, wave)
            plt.title('Analytic Wave: n = %g' %n)
            plt.show()
            return wave
        else:
            print('error in true wave plot input')

        


class Harmonic_Oscillator:
    
    def potential(self,x, D):
        return D**2/2*x**2

    def true_energy(self,n, D):
        return np.sqrt(D)*(n+1/2)


    def true_wave(self, n, D, grid, plot = 'no'):
        omega = np.sqrt(D)
        wave = (omega/(pi))**(1/4)/np.sqrt(2**n*sp.misc.factorial(n))*spec.eval_hermite(n,np.sqrt(omega)*grid)*np.exp(-omega/(2)*grid**2)

        
        if plot == 'no':
            return wave
        elif plot == 'yes':
            plt.plot(grid, wave)
            plt.title('Analytic Wave: n = %g' %n)
            plt.show()

            return wave
        else:
            print('error in true wave plot input')
            
