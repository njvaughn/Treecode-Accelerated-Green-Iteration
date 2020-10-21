import numpy as np
import numpy.linalg as la
import os
import matplotlib.pyplot as plt
from models import *
##from GreensIt_EigOne3 import EO_driver

from matplotlib import style
##style.use('ggplot')
style.use('seaborn-whitegrid')

import mpltex

#################################################################################
#################################################################################
"""                              PLOTTING FUNCTIONS                           """ 
#################################################################################
#################################################################################
    
##@mpltex.acs_decorator
def plot_energy_cascade(run, energy_level = 0, save = 'no'):


    if run.model == 'EO':
        method='$Eigenvalue$ $One $ '
    elif run.model == 'GI':
        method='$Green\'s$ $Iterations $ '
    else:
        method = ''

    mesh_levels = len(run.energy_errors[0])

    plt.figure()
    plt.grid(True,which="both",ls="-",color='0.65')
    plt.title(method+'$Energy\ Errors$ $vs.$ $Grid\ Points$')

    count = 0
##    linestyles = mpltex.linestyle_generator()
    for i in range(mesh_levels):
        if count == 0:
            lab = '$Calculated\ Values$'
            col = 'bo-'
        elif count == 1:
            lab = '$1^{st}\ Extrapolation$'
            col = 'rD-'
        elif count ==2:
            lab = '$2^{nd}\ Extrapolation$'
            col = 'gs-'
        elif count == 3:
            lab = '$3^{rd}\ Extrapolation$'
            col = 'cX-'
        elif count == 4:
            lab = '$4^{th}\ Extrapolation$'
            col = 'kP-'
        # if multiple energy levels, this is failing...
        plt.loglog(run.nx_vec[i:],np.abs(run.energy_errors[:mesh_levels-i,i]),col,
                   label=lab)
##        plt.loglog(run.nx_vec[i:],np.abs(run.energy_errors[:mesh_levels-i,i]),
##                    **next(linestyles))
        count +=1
    plt.legend(loc='best',frameon=False)
    plt.xlim([60,10000])
    plt.ylim([1E-15, 1E0])
    plt.xlabel('$Number\ of\ Grid\ Points$')
    plt.ylabel('$Energy\ Errors$')

    if save != 'no':
        cwd = os.getcwd()
        save_dir = '/Users/nathanvaughn/UMich_Drive/Research/Figure_Stockpile/'
        plt.savefig(save_dir + save + '.pdf',format='pdf')
        

        
    plt.show()

def plot_wave_cascade(run, energy_level = 0, save = 'no'):

    if run.model == 'EO':
        method='$Eigenvalue$ $One $ '
    elif run.model == 'GI':
        method='$Green\'s$ $Iterations $ '
    else:
        method = ''

    mesh_levels = len(run.energy_errors[0])

    plt.figure()
    plt.grid(True,which="both",ls="-",color='0.65')
    plt.title(method+'$Wavefunction\ Errors$ $vs.$ $Grid\ Points$')

    count = 0
    for i in range(mesh_levels):
        if count == 0:
            lab = '$Calculated\ Values$'
            col = 'bo-'
        elif count == 1:
            lab = '$1^{st}\ Extrapolation$'
            col = 'rD-'
        elif count ==2:
            lab = '$2^{nd}\ Extrapolation$'
            col = 'gs-'
        elif count == 3:
            lab = '$3^{rd}\ Extrapolation$'
            col = 'cX-'
        elif count == 4:
            lab = '$4^{th}\ Extrapolation$'
            col = 'kP-'
        # if multiple energy levels, this is failing...
        plt.loglog(run.nx_vec[i:],np.abs(run.wave_errors[:mesh_levels-i,i]),col, label=lab)
        count +=1
    plt.legend(loc='best',frameon=False)
##    plt.xlim([60,10000])
    plt.ylim([1E-15, 1E0])
    plt.xlabel('$Number\ of\ Grid\ Points$')
    plt.ylabel('$Wave\ Errors$')

    if save != 'no':
        cwd = os.getcwd()
        save_dir = '/Users/nathanvaughn/UMich_Drive/Research/Figure_Stockpile/'
        plt.savefig(save_dir + save + '.pdf',format='pdf')
        
    plt.show()
    
##@mpltex.acs_decorator
def plot_domain_effects(nx_coarse, nx_fine, xmax_small, xmax_large, save='no'):




    def potential(x, D):
        return -D*(D+1)/2*(1/np.cosh(x))**2


    def G(x,z): # Green's function for 1D Helmholtz Operator
        return -1/(2*np.sqrt(-2*z))*np.exp(-np.sqrt(-2*z)*x)

    def matrix(z): # Discretized integral operator
        return dx*np.fromfunction(lambda i, ii:
                2*V[ii]*G(np.abs(
                xgrid[i]-xgrid[ii]),z), (nx,nx),dtype=int)

    def inverse_iteration(A):
        I = np.eye(len(A))
        psi = psi_in

        eig_old = 2
        eig_new = 10
        LU = sp.linalg.lu_factor(A-I)
        count=0
        while abs(eig_new-eig_old) > eig_tolerance:  #and count < 15000: # 1e-14 sufficeint for Morse, 1e-16 doesnt improve
            eig_old     = eig_new
            psi         = sp.linalg.lu_solve(LU,psi)
            psi         = psi/np.linalg.norm(psi)
            eig_new     = np.dot(psi,np.dot(A,psi))
            count+=1

        return psi, eig_new

    
    D = 4
    E = -0.5 # for 3rd excited state
    n = 3
    omega = 1/np.sqrt(2*D)
    lmbda = 2*D
    eig_tolerance = 1e-14
    refinement_rate = 3 # doesn't get used

    
    xmax = xmax_small
    xmin = -xmax
    nx = nx_coarse
    psi_in = np.random.rand(nx)
    
    xgrid = np.linspace(xmin,xmax,nx)
    dx = (xmax-xmin)/nx
    wave_true = spec.lpmv(D-n, D, np.tanh(xgrid))
    wave_true = wave_true/np.sqrt(np.dot(wave_true,wave_true*dx))
    V = potential(xgrid,D)
    A = matrix(E)
    psi,eig = inverse_iteration(A)
    print(eig)
    psi = psi/np.sqrt(np.dot(psi,psi*dx))

    if np.dot(psi,wave_true) < np.dot(-psi,wave_true):
        psi = -psi


    err_max = np.max(abs(psi-wave_true))
    plt.subplot(221)
##    plt.plot(xgrid, wave_true, label='true')
##    plt.plot(xgrid, psi, label='comp')
    plt.plot(xgrid, wave_true-psi)
##    plt.plot(xgrid, wave_true, 'r')
##    plt.plot(xgrid, psi, 'g')
##    plt.legend()
##    plt.ylim([-err_max,err_max])
    plt.xlim([xmin,xmax])
    plt.title('$Short\ Domain,\ Coarse\ Grid$')
##    plt.yticks(np.linspace(ymin, ymax,3, endpoint=True), ['$-4E-8$', '$0$' ,'$4E-8$']) 
##    plt.xticks(np.linspace(xmin, xmax,5, endpoint=True),['$-7$','$-3.5$', '$0$','$3.5$','$7$']) 


    xmax = xmax_large
    xmin = -xmax
    nx = nx_coarse
    psi_in = np.random.rand(nx)
    
    xgrid = np.linspace(xmin,xmax,nx)
    dx = (xmax-xmin)/nx
    wave_true = spec.lpmv(D-n, D, np.tanh(xgrid))
    wave_true = wave_true/np.sqrt(np.dot(wave_true,wave_true*dx))
    V = potential(xgrid,D)
    A = matrix(E)
    psi,eig = inverse_iteration(A)
    print(eig)
    psi = psi/np.sqrt(np.dot(psi,psi*dx))
    if np.dot(psi,wave_true) < np.dot(-psi,wave_true):
        psi = -psi
        
    plt.subplot(222)
    plt.plot(xgrid, wave_true-psi)
##    plt.plot(xgrid, wave_true, 'r')
##    plt.plot(xgrid, psi, 'g')
    plt.title('$Long\ Domain,\ Coarse\ Grid$')
##    plt.xticks(np.arange(xmin, xmax, 2.0))
##    plt.yticks(np.linspace(ymin, ymax,3, endpoint=True), ['$-4E-8$', '$0$' ,'$4E-8$']) 
##    plt.yticks(np.linspace(ymin, ymax,3, endpoint=True), ['', '' ,'']) 
##    plt.xticks(np.linspace(xmin, xmax,5, endpoint=True),['$-14$','$-7$', '$0$','$7$','$14$'])
##    plt.ylim([-err_max,err_max])
##    plt.xlim([-14,14])

    xmax = xmax_small
    xmin = -xmax
    nx = nx_fine
    psi_in = np.random.rand(nx)

    xgrid = np.linspace(xmin,xmax,nx)
    dx = (xmax-xmin)/nx
    wave_true = spec.lpmv(D-n, D, np.tanh(xgrid))
    wave_true = wave_true/np.sqrt(np.dot(wave_true,wave_true*dx))
    V = potential(xgrid,D)
    A = matrix(E)
    psi,eig = inverse_iteration(A)
    print(eig)
    psi = psi/np.sqrt(np.dot(psi,psi*dx))
    if np.dot(psi,wave_true) < np.dot(-psi,wave_true):
        psi = -psi


    plt.subplot(223)
    plt.plot(xgrid, wave_true-psi)
##    plt.plot(xgrid, wave_true, 'r')
##    plt.plot(xgrid, psi, 'g')
##    plt.ylim([-err_max,err_max])
##    plt.xlim([-7,7])
    plt.title('$Short\ Domain,\ Fine\ Grid$')
##    plt.yticks(np.linspace(ymin, ymax,3, endpoint=True), ['$-4E-8$', '$0$' ,'$4E-8$']) 
##    plt.xticks(np.linspace(xmin, xmax,5, endpoint=True),['$-7$','$-3.5$', '$0$','$3.5$','$7$']) 


    xmax = xmax_large
    xmin = -xmax
    nx = nx_fine
    psi_in = np.random.rand(nx)

    xgrid = np.linspace(xmin,xmax,nx)
    dx = (xmax-xmin)/nx
    wave_true = spec.lpmv(D-n, D, np.tanh(xgrid))
    wave_true = wave_true/np.sqrt(np.dot(wave_true,wave_true*dx))
    V = potential(xgrid,D)
    A = matrix(E)
    psi,eig = inverse_iteration(A)
    print(eig)
    psi = psi/np.sqrt(np.dot(psi,psi*dx))
    if np.dot(psi,wave_true) < np.dot(-psi,wave_true):
        psi = -psi

    plt.subplot(224)
    plt.plot(xgrid, wave_true-psi)
##    plt.plot(xgrid, wave_true, 'r')
##    plt.plot(xgrid, psi, 'g')
    plt.title('$Long\ Domain,\ Fine\ Grid$')
##    plt.yticks(np.arange(xmin, xmax, 2.0))
##    plt.yticks(np.linspace(ymin, ymax,3, endpoint=True), ['', '' ,'']) 
##    plt.xticks(np.linspace(xmin, xmax,5, endpoint=True),['$-14$','$-7$', '$0$','$7$','$14$']) 
##    plt.ylim([-err_max,err_max])
##    plt.xlim([-14,14])

    plt.tight_layout(pad=0.2, w_pad=0.5, h_pad=1.0)

    if save != 'no':
        cwd = os.getcwd()
        save_dir = '/Users/nathanvaughn/UMich_Drive/Research/Figure_Stockpile/'
        plt.savefig(save_dir + save + '.pdf',format='pdf')

    plt.show()


def plot_potential(model, model_name, xmax, xmin, nx, D, n, save='no'):
    Model = model
    xgrid = np.linspace(xmin,xmax,nx)
    dx = (xmax - xmin)/nx

    plt.plot(xgrid, Model.potential(Model,xgrid, D), 'k-', label=r'$Potential$' )
    if model == Square:
        Energy_vec = [-0.96218293, -0.84949808, -0.66465585, -0.41428222,
                      -0.1193073]
        
##        plt.show()
##    return
    linestyles = mpltex.linestyle_generator(lines=['-'],hollow_styles=[])
##    linestyles = mpltex.linestyle_generator(colors=['k'],markers=[],hollow_styles=[])
    for i in range(n):
        if model == Square:
            a = 5*D
            E = Energy_vec[i]
            energy=E
##            print(E)
##            print(E+D)
            ell = np.sqrt(2*(D+E))
            k =np.sqrt(2*(-E))   
            xvec = xgrid
            wave = np.zeros(len(xvec))
            DD = np.exp(-k*a)
            if i%2 ==0:
                B = np.cos(ell*a)
            else:
                B = np.sin(ell*a)
            for ii in range(len(xvec)):
                if np.abs(xvec[ii]) < a:
                    if i%2 ==0:
                        wave[ii] = DD*np.cos(ell*xvec[ii])
                    if i%2 ==1:
                        wave[ii] = DD*np.sin(ell*xvec[ii])
                if np.abs(xvec[ii]) >= a:
                    if xvec[ii]>0:
                        wave[ii] = B*np.exp(-k*xvec[ii])
                    if xvec[ii]<0:
                        if i%2 ==0:
                            wave[ii] = B*np.exp(k*xvec[ii])
                        if i%2 ==1:
                            wave[ii] = -B*np.exp(k*xvec[ii])
##            A = np.dot(wave,wave)
##            wave = wave/np.sqrt(A)
        else:
            wave = Model.true_wave(Model, i, D, xgrid)
            energy = Model.true_energy(Model, i ,D)
        A = np.sqrt( np.dot(wave,wave*dx) )
        wave = wave/A
        if i == 0:
            ell = r'$Ground\ State$'
        elif i == 1:
            ell = r'$First Excited State$'
        elif i == 2:
            ell = r'$Second Excited State$'
        elif i == 3:
            ell = r'$Third Excited State$'
        else:
            ell = []
        
##        plt.plot(xgrid, energy+wave, label = ell)
##        linestyles = mpltex.linestyle_generator()
        plt.plot(0,-50, **next(linestyles))
        plt.plot(0,-50, **next(linestyles))
        plt.plot(0,-50, **next(linestyles))
        if i%2 ==1:
            plt.plot(xgrid, energy+wave ,label = r'$n = %g$'%i, **next(linestyles),Markersize=5,Markevery=10)
    plt.legend(frameon=True,loc='lower left')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$Energy$')
##    plt.ylim([-1.4,0.3])
##    plt.title('Potential and Wavefunctions for ' + model_name)
##    plt.title(model_name)
    if model == Poschl_Teller:
        plt.ylim([-2.75*D,2])
    if model == Morse: 
        plt.ylim([-1.1*D, 2])
    if model == Harmonic_Oscillator:
        plt.ylim([-0.5, 7])
    if model == Square:
        plt.ylim([-1.4, .4])
    plt.title(model_name + ' Potential: Odd States')

    if save != 'no':
        cwd = os.getcwd()
        save_dir = '/Users/nathanvaughn/UMich_Drive/Research/Figure_Stockpile/'
        plt.savefig(save_dir + save + '.pdf',format='pdf')

    plt.show()

def plot_morse_waves(xmax, xmin, nx, D, save='no', tails='no'):

    xgrid = np.linspace(xmin,xmax,nx)
    dx = (xmax-xmin)/nx

    omega = 1/np.sqrt(2*D)
    lmbda = 2*D
    z = 2*lmbda*np.exp(-omega*xgrid)
    n=0
    alpha = 2*lmbda - 2*n - 1   
    N = np.sqrt((math.factorial(n)*(2*lmbda - 2*n - 1))/(sp.special.gamma(2*lmbda-n)))
    L = spec.eval_genlaguerre(n,alpha,z)
    wave0 =  N*z**(lmbda-n-1/2)*np.exp(-z/2)*L
    A = np.dot(wave0,wave0*dx)
    wave0 = wave0/np.sqrt(A)


    n=7
    alpha = 2*lmbda - 2*n - 1   
    N = np.sqrt((math.factorial(n)*(2*lmbda - 2*n - 1))/(sp.special.gamma(2*lmbda-n)))
    L = spec.eval_genlaguerre(n,alpha,z)
    wave7 =  N*z**(lmbda-n-1/2)*np.exp(-z/2)*L
    A = np.dot(wave7,wave7*dx)
    wave7 = wave7/np.sqrt(A)
    
    f, axarr = plt.subplots(2, sharex=True, sharey=False)
##    plt.ylabel(r'$Wavefunctions$')
    axarr[0].plot(xgrid,wave0,'b',label=r'$Ground\ State$')
##    axarr[0].set_title('Morse Wavefunctions')
    axarr[0].set_ylabel(r'$n=0\ Wave$')
    plt.setp(axarr[0].get_xticklabels(), visible=True)
    axarr[1].plot(xgrid,wave7,'r',label=r'$7^{th}\ Excited\ State$')
    axarr[0].legend()
    axarr[1].legend()
    plt.xlabel(r'$x$')
    axarr[1].set_ylabel(r'$n=7\ Wave$')

    if save != 'no':
        cwd = os.getcwd()
        save_dir = '/Users/nathanvaughn/UMich_Drive/Research/Figure_Stockpile/'
        plt.savefig(save_dir + save + '.pdf',format='pdf')
    
    plt.show()

    if tails != 'no':
        nx_end = -int(nx/10)
        f, axarr = plt.subplots(2, sharex=True, sharey=True)
        axarr[0].plot(xgrid[nx_end:],wave0[nx_end:],'b',label=r'$Ground\ State$')
        axarr[0].set_title('Morse Wavefunction Tails')
        axarr[0].set_ylabel(r'$n=0\ Wave$')
        plt.setp(axarr[0].get_xticklabels(), visible=True)
        axarr[1].plot(xgrid[nx_end:],wave7[nx_end:],'r',label=r'$7^{th}\ Excited\ State$')
        axarr[0].legend()
        axarr[1].legend()
        plt.xlabel(r'$x$')
        axarr[1].set_ylabel(r'$n=7\ Wave$')

        if save != 'no':
            cwd = os.getcwd()
            save_dir = '/Users/nathanvaughn/UMich_Drive/Research/Figure_Stockpile/'
            plt.savefig(save_dir + save + 'tails.pdf',format='pdf')
        plt.show()
        
        

def plot_sweep(xmax, xmin, nx,
                   Emax, Emin, nE, D, save='no'):
    def potential(x, D):
        return -D*(D+1)/2*(1/np.cosh(x))**2
    xgrid = np.linspace(xmin,xmax,nx)
    dx = (xmax-xmin)/nx
    V = potential(xgrid,D)

    def G(x,z): # Green's function for 1D Helmholtz Operator
        return -1/(2*np.sqrt(-2*z))*np.exp(-np.sqrt(-2*z)*x)

    def matrix(z): # Discretized integral operator
        return dx*np.fromfunction(lambda i, ii:
                2*V[ii]*G(np.abs(
                xgrid[i]-xgrid[ii]),z), (nx,nx),dtype=int)

    def inverse_iteration(A):
        I = np.eye(len(A))
        psi = psi_in

        eig_old = 2
        eig_new = 10
        LU = sp.linalg.lu_factor(A-I)
        count=0
        while abs(eig_new-eig_old) > eig_tolerance and count < 5000: # 1e-14 sufficeint for Morse, 1e-16 doesnt improve
            eig_old     = eig_new
            psi         = sp.linalg.lu_solve(LU,psi)
            psi         = psi/np.linalg.norm(psi)
            eig_new     = np.dot(psi,np.dot(A,psi))
            count+=1

        return psi, eig_new 

    D = 4
    eig_tolerance = 1e-14
    E_vec = np.linspace(Emin,Emax,nE)
    eig_vec = []
    mesh_levels = 1
    refinement_rate = 1
    
    psi_in = np.random.rand(nx)


    for E in E_vec:
        A = matrix(E)
        psi,eig = inverse_iteration(A)
        eig_vec = np.append(eig_vec, eig)

    plt.figure()
    plt.plot(E_vec, eig_vec, 'k.',label=r'$Computed\ Eigenvalues$')
    plt.plot(E_vec[::1], np.ones(len(E_vec[::1])), 'k-.',Markersize=2)
##    plt.plot([-8, -4.5, -2, -0.5], [1,1,1,1], 'ks',mew=3, label=r'$Analytic\ Energies$')
    plt.plot([-8, -4.5, -2, -0.5], [1,1,1,1], 'rx', Markersize=8, label=r'$Analytic\ Energies$')
    plt.legend(frameon=True)
    plt.title(r'$P\"{o}schl-Teller\ Energy\ Sweep$')
    plt.xlabel(r'$Input\ Energy$')
    plt.ylabel(r'$Eigenvalue\ Nearest\ 1$')

    if save != 'no':
        cwd = os.getcwd()
        save_dir = '/Users/nathanvaughn/UMich_Drive/Research/Figure_Stockpile/'
        plt.savefig(save_dir + save + '.pdf',format='pdf')
    
    plt.show()
    

    


    psi_in = np.random.rand(nx)


    for E in E_vec:
        A = matrix(E)
        psi,eig = inverse_iteration(A)
        eig_vec = np.append(eig_vec, eig)

    plt.figure()
    plt.plot(E_vec, eig_vec, 'k.',label=r'$Computed\ Eigenvalues$')
    plt.plot(E_vec[::1], np.ones(len(E_vec[::1])), 'k-.',Markersize=2)
##    plt.plot([-8, -4.5, -2, -0.5], [1,1,1,1], 'ks',mew=3, label=r'$Analytic\ Energies$')
    plt.plot([-8, -4.5, -2, -0.5], [1,1,1,1], 'rx', Markersize=8, label=r'$Analytic\ Energies$')
    plt.legend(frameon=True)
    plt.title(r'$P\"{o}schl-Teller\ Energy\ Sweep$')
    plt.xlabel(r'$Input\ Energy$')
    plt.ylabel(r'$Eigenvalue\ Nearest\ 1$')

    if save != 'no':
        cwd = os.getcwd()
        save_dir = '/Users/nathanvaughn/UMich_Drive/Research/Figure_Stockpile/'
        plt.savefig(save_dir + save + '.pdf',format='pdf')
    
    plt.show()
    

    

   