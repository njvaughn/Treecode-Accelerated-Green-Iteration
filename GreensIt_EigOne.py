"""

Structure the Schrodinger Calculation Method


Classes
-- Models (to be inherited by Bound_States)
    -- Poschl_Teller
    -- Morse
    -- Finite Square Well
    
-- Bound_States
    -- grid and potential well parameters
    -- inherits the potential, analytic energy, and analytic wave for the
        chosen model.
    -- contains the greens iteration and eigenvalue-one numerical schemes
        

--Drivers
    -- Choice between the Greens Iteration driver and the Eigenvalue-One driver
    -- Selects energy level(s), initial grid, numerical tolerances, etc.
    -- refinement/extrapolation info

"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import scipy as sp
import scipy.optimize as spo
from models import *
from plot_utilities import *

"""  Select Model """
Model = Poschl_Teller # select a class from models.py
##Model = Morse
##Model = Square
##Model = Harmonic_Oscillator


#################################################################################
#################################################################################
"""                         DEFINE BOUND STATE CLASS                          """ 
#################################################################################
#################################################################################
class Bound_States(Model):
    # object inherits self.V, self.true_energy, and self.true_wave for the chosen model
    def __init__(self,  x_min, x_max, nx, D):
        self.xmax = x_max
        self.xmin = x_min
        self.nx = nx
        self.D = D # well parameter
        self.dx = (self.xmax - self.xmin)/(self.nx)
        self.xendpoints = np.linspace(self.xmin, self.xmax, self.nx+1)
        self.xgrid = np.array([(self.xendpoints[i+1]+self.xendpoints[i])/2 for i in range(len(self.xendpoints)-1)])
        
        self.V = self.potential(self.xgrid, self.D)

    def mesh_refinement(self, factor): # updates when grid is refined
        coarse_grid = self.xgrid

        self.nx = (self.nx)*factor
        self.dx = (self.xmax - self.xmin)/(self.nx)
        self.xendpoints = np.linspace(self.xmin, self.xmax, self.nx+1)
        self.xgrid = np.array([(self.xendpoints[i+1]+self.xendpoints[i])/2 for i in range(len(self.xendpoints)-1)])
        self.V = self.potential(self.xgrid, self.D)
        psi_temp = np.zeros((energy_levels, self.nx))
        if self.psi.ndim > 1:
            for i in range(energy_levels):
                psi_temp[i,:] = np.interp(self.xgrid, coarse_grid, self.psi[i,:])
            self.psi_in = psi_temp
        else:
            self.psi_in = np.interp(self.xgrid, coarse_grid, self.psi)            

    def normalize(self,psi):
        return psi/np.sqrt(np.sum(psi**2)*self.dx)

    def G(self,x,z): # Green's function for 1D Helmholtz Operator
##        print(z)
        if Model == Harmonic_Oscillator:
            # positive bound states...
            return 1/(2*np.sqrt(2*z))*np.sin(np.sqrt(2*z)*x)
        
        else: #negative bound state energies
            return -1/(2*np.sqrt(-2*z))*np.exp(-np.sqrt(-2*z)*x)

    def matrix(self,z): # Discretized integral operator
        return self.dx*np.fromfunction(lambda i, ii:
                2*self.V[ii]*self.G(np.abs(
                self.xgrid[i]-self.xgrid[ii]),z), (self.nx,self.nx),dtype=int)

    def hamiltonian(self):
        H = ( -(1/2)*(1/self.dx**2)*( np.diag(np.ones(self.nx-1),-1) + 
        np.diag(np.ones(self.nx-1),1) -np.diag(2*np.ones(self.nx)) ) +
        np.diag(self.V) )
        return H

    def orthogonalize(self,psi_in,psi_orth):
        return psi_in - np.dot(psi_in,psi_orth)/np.dot(psi_orth,psi_orth)*psi_orth

    def RE(self,E_coarse,E_fine,order,k):
    # Richardson Extrapolation
    # order is the previous order of convergence
    # k is the step size reduction factor.  2 if dx was halved.
        return (k**order*E_fine-E_coarse)/(k**order-1)    


    def greens_iterations(self):
        H = self.hamiltonian()
        z= self.z_in
        z_old = 2*self.z_in
        psi = self.psi_in
        # initial orthogonalization
        for i in range(1,energy_levels):
            for j in range(i):
                psi[i,:] = self.orthogonalize(psi[i,:], psi[j,:])
        for i in range(energy_levels):
            psi[i,:] = self.normalize(psi[i,:])
            

        count=0
        totalcount = 0
        error_vec = np.ones((energy_levels,))

        for i in range(energy_levels):  # perform serially
            count = 0
            while abs(error_vec[i]) > self.z_tolerance and count<4000:
                z_old[i] = z[i]
                G_z = self.matrix(z[i])
                psi[i,:] = self.normalize(np.dot(G_z,psi[i,:]))
            
            # orthogonalize against all lower energies
                if i > 0:
                    for j in range(i):
                        psi[i,:] = self.orthogonalize(psi[i,:], psi[j,:])
                psi[i,:] = self.normalize(psi[i,:])
                z[i] = np.dot(np.dot(H,psi[i,:]),psi[i,:]*self.dx)

                if (z[i]>0) and count < 50 and Model != Harmonic_Oscillator:
                    print((i+1),'energy went positive z at step ', count)
                    print(z-z_old)
                    z[i] = -np.random.random(1)*D

                error_vec[i] = z_old[i] - z[i]
##                error_vec = z_old - z
                if totalcount%100==0:
                    print('count ', count)
                    print('error_vec = ', (error_vec))
                count+=1
                totalcount+=1

        return psi,z


    ###Eigenvalue-One Methods###

    def inverse_iteration(self,A):
        I = np.eye(len(A))
        psi = self.psi_in

        eig_old = 2
        eig_new = 10
        LU = sp.linalg.lu_factor(A-I)
        count=0
        while abs(eig_new-eig_old) > self.eig_tolerance and count < 5000: # 1e-14 sufficeint for Morse, 1e-16 doesnt improve
            eig_old     = eig_new
            psi         = sp.linalg.lu_solve(LU,psi)
            psi         = psi/np.linalg.norm(psi)
            eig_new     = np.dot(psi,np.dot(A,psi))
            count+=1

        return psi, eig_new 


    def E_to_Eig_Minus1(self,z):
        A = self.matrix(z)
        psi,eig = self.inverse_iteration(A)
        return eig-1

    def E_guess_BI(self,E_low,E_high):
        brent_time = time.time()
        try:
            z = spo.brenth(self.E_to_Eig_Minus1, self.e_low, self.e_high, xtol=2e-16, rtol=8.8817841970012523e-16,
                           maxiter=100, full_output=False, disp=True)
            print('brent time = ', time.time()-brent_time)
            return z
        except ValueError as VE:
            print('Brent Rootfinding failed, no sign change.  Trying to rememdy...')

        low = self.E_to_Eig_Minus1(E_low)
        high = self.E_to_Eig_Minus1(E_high)
        if abs(low) < abs(high):
            E_low = E_low*1.1
        else:
            E_high = E_high*0.9

        try:
            z = spo.brenth(self.E_to_Eig_Minus1, E_low, E_high, xtol=2e-16, rtol=8.8817841970012523e-16,
                           maxiter=100, full_output=False, disp=True)
            return z
        except ValueError as VE:
            print('Brent Rootfinding still failed.  Adjust E_low and E_high.')

    def E_guess_Newton(self,E_low,E_high):
        newt_time = time.time()
##        z_old = self.z_in
##        z_old = (E_high+E_low)/2
        
        z_old = self.e_true # this is "cheating".  But I could instead manually
                            # pick starting values in the target energy's basin
                            # of attraction.  Just have to be careful.
        print('initial z :', z_old)
        z_new = z_old - 1e-6
        while abs(z_new-z_old) > 1e-13:
            z_old = z_new
            A = self.matrix(z_old)
            psi, eig = self.inverse_iteration(A)
##            print('eig = ', eig)
            z_new = z_old - (1-eig)/eig**2* np.dot( psi, self.V*psi )/np.dot(psi,psi)
        print('Newton search time: ', time.time()-newt_time)
        return z_new


        
    def eigenvalue_one(self):
        
        z = self.E_guess_Newton(self.e_low, self.e_high)
##        z = self.E_guess_BI(self.e_low, self.e_high)
        A = self.matrix(z)
        psi,eig = self.inverse_iteration(A)

        return psi,z
        


        
#################################################################################
#################################################################################
"""             END OF BOUND STATE CLASS :: BEGINNING OF DRIVERS              """ 
#################################################################################
#################################################################################
        


def GI_driver(x_min,x_max,nx,D,psi_in,z_in,z_tolerance,mesh_levels,energy_levels,refinement_rate):

    def normalize(psi,dx):
        B = np.sum(psi**2)*dx
        return psi/np.sqrt(B) 
        
    def norm(psi,dx):
        # norm computed with the midpoint rule
        return np.sqrt(np.sum(psi**2)*dx)
    
    # initiate the run, and add in the runtime iteration parameters

    run = Bound_States(xmin,xmax,nx,D)
    dx = (xmax-xmin)/nx
    e_true_vec = []
    run.wave_true = np.empty((energy_levels,),dtype=object)
    for i in range(energy_levels):
        e_true_vec = np.append(e_true_vec, run.true_energy(i, D))
        run.wave_true[i] = normalize(run.true_wave(i,D,run.xgrid),dx)
    run.e_true = e_true_vec
    run.energy_errors = np.zeros((mesh_levels,mesh_levels),dtype=object)
    run.energy_computed = np.zeros((mesh_levels,),dtype=object)
    
    run.psi_computed = np.zeros((mesh_levels,),dtype=object)
    run.psi_in = psi_in
    if Model == Harmonic_Oscillator:
        run.z_in = np.abs(z_in)
    else:
        run.z_in = -np.abs(z_in)
    run.z_tolerance = z_tolerance
    run.nx_vec = []
    run.model = 'GI'




    # Compute energies and wavefunctions for each mesh level
    for l in range(mesh_levels):
        driver_time = time.time()
        print('nx = ', run.nx)
        print('dx = ', run.dx)
        run.nx_vec.append(run.nx)
        run.psi, run.e = run.greens_iterations()
        run.z_in = run.e.copy()
        run.energy_computed[l] = run.e.copy()
        run.psi_computed[l] = run.psi.copy()
        print('time = ', time.time()-driver_time, '\n')
        if l < mesh_levels-1:
            run.mesh_refinement(refinement_rate)

    # Extrapolations and Error Calculations
    convergence_rate = 2
    E_matrix = np.zeros((mesh_levels,mesh_levels),dtype=object)
    Wave_matrix = np.empty((mesh_levels,mesh_levels,energy_levels),dtype=object)
    Wave_errors = np.zeros((mesh_levels,mesh_levels,energy_levels))
    waves_n = np.empty((mesh_levels,),dtype=object)
    E_matrix[:,0] = run.energy_computed.copy()

        # pick out the right terms from the refined wavefunctions
    try:
        for l in range(energy_levels):
            for i in range(mesh_levels):
                first = 0
                for j in range(1,i+1):
                    first += refinement_rate**(j-1)
                waves_n[i] = normalize(run.psi_computed[i][l][first::refinement_rate**i],dx)
            Wave_matrix[:,0,l] = waves_n
    except ValueError as VE:
        print(VE)


    
    for i in range(mesh_levels):
        run.energy_errors[i,0] = run.e_true - E_matrix[i,0]
    for j in range(1,mesh_levels):
        for i in range(mesh_levels-j):
            E_matrix[i,j] = run.RE(E_matrix[i,j-1],E_matrix[i+1,j-1],convergence_rate*j,refinement_rate)
            run.energy_errors[i,j]= run.e_true - E_matrix[i,j]
    for l in range(energy_levels):
        for j in range(1,mesh_levels):
            for i in range(mesh_levels-j):
                Wave_matrix[i,j,l] = normalize( run.RE(Wave_matrix[i,j-1,l],Wave_matrix[i+1,j-1,l],convergence_rate*j,refinement_rate),dx )
                Wave_errors[i,j,l] = min( norm(Wave_matrix[i,j,l]-run.wave_true[l],dx), norm(Wave_matrix[i,j,l]+run.wave_true[l],dx) )

    for l in range(energy_levels):
        for i in range(mesh_levels):
            Wave_errors[i,0,l] = min( norm(Wave_matrix[i,0,l]-run.wave_true[l],dx), norm(Wave_matrix[i,0,l]+run.wave_true[l],dx) )



    run.wave_matrix = Wave_matrix
    run.wave_errors = Wave_errors
    run.E_matrix = E_matrix

    # clean out values that don't exist
    for i in range(mesh_levels):
        for j in range(mesh_levels):
            if i+j >= mesh_levels:
                run.energy_errors[i,j] = None
                run.wave_errors[i,j] = None
                run.E_matrix[i,j] = None
    print('='*70)
    print('='*70+'\n' + '\n')

    
    for l in range(energy_levels):        
        print('Energy Results for level ', l)
        print('True Energy = ', run.e_true[l])
        print('Final Extrapolated Value = ', E_matrix[0,mesh_levels-1][l])
        print('-'*70)
        for i in range(mesh_levels):
            for j in range(mesh_levels-i):
                if j == 0:
                    print('Absolute Energy Error at Mesh Level %2.2g = %2.2e' %(i , run.energy_errors[j,i][l]) )
                else:
                    print('Absolute Energy Error at Mesh Level %2.2g = %2.2e' %(i , run.energy_errors[j,i][l]) , ', ratio = %.1f' %(run.energy_errors[j-1,i][l]/run.energy_errors[j,i][l] ) )
            print()

        print('Wavefunction Error') 
        print('-'*70)
        for i in range(mesh_levels):
            for j in range(mesh_levels-i):
                if j == 0:
                    print('Wave Error at Mesh Level %2.2g = %2.2e' %(i , Wave_errors[j,i,l]) )
                else:
                    print('Wave Error at Mesh Level %2.2g = %2.2e' %(i , Wave_errors[j,i,l]), ', ratio = %.1f' %(Wave_errors[j-1,i,l]/Wave_errors[j,i,l] ) )
            print()
            
        
        print('='*70,'\n')
        if l < energy_levels-1:
            print(input('hit enter to continue to next energy...'))
        print('='*70,'\n')       

    return run







def EO_driver(x_min,x_max,nx,D,psi_in,eig_tolerance,mesh_levels,refinement_rate, n=None):

    if n == None:
        print('='*70,'\n')
        n = int(input('Input the target energy level... (n = 0, 1, 2, ...)' + '\n'))
        print('='*70,'\n')


        
    def normalize(psi,dx):
        B = np.sum(psi**2)*dx
        return psi/np.sqrt(B) 
        
    def norm(psi,dx):
        # norm computed with the midpoint rule
        return np.sqrt(np.sum(psi**2)*dx)
    
    # initiate the run, and add in the runtime iteration parameters

    run = Bound_States(xmin,xmax,nx,D)
    dx = (xmax-xmin)/nx
    run.e_true = run.true_energy(n,D)
    run.e_low = 1.4*run.e_true
    run.e_high = 1*run.e_true
##    run.e_low = z_in
##    run.e_high = z_in
    run.n = n
    run.model = 'EO'

    if Model == Harmonic_Oscillator:
        run.z_in = np.abs(z_in)
    else:
        run.z_in = -np.abs(z_in)

    
    run.wave_true = normalize(run.true_wave(n,D,run.xgrid),dx)
    
    run.energy_errors = np.zeros((mesh_levels,mesh_levels),dtype=object)
    run.energy_computed = np.zeros((mesh_levels,),dtype=object)
    
    run.psi_computed = np.zeros((mesh_levels,),dtype=object)
    run.psi_in = psi_in
    run.eig_tolerance = eig_tolerance
    run.nx_vec = []



    # Compute energies and wavefunctions for each mesh level
    for l in range(mesh_levels):
        driver_time = time.time()
        print('nx = ', run.nx)
        print('dx = ', run.dx)
        run.nx_vec.append(run.nx)
        run.psi, run.e = run.eigenvalue_one()
        run.energy_computed[l] = run.e
        run.z_in = run.e.copy()
        run.psi_computed[l] = run.psi.copy()
        print('time = ', time.time()-driver_time, '\n')
        if l < mesh_levels-1:
            run.mesh_refinement(refinement_rate)

    # Extrapolations and Error Calculations
    convergence_rate = 2
    E_matrix = np.zeros((mesh_levels,mesh_levels),dtype=object)
    Wave_matrix = np.empty((mesh_levels,mesh_levels),dtype=object)
    Wave_errors = np.zeros((mesh_levels,mesh_levels))
    waves_n = np.empty((mesh_levels,),dtype=object)
    E_matrix[:,0] = run.energy_computed.copy()

        # pick out the right terms from the refined wavefunctions
    try:
        for i in range(mesh_levels):
            first = 0
            for j in range(1,i+1):
                first += refinement_rate**(j-1)
            waves_n[i] = normalize(run.psi_computed[i][first::refinement_rate**i],dx)
        Wave_matrix[:,0] = waves_n
    except ValueError as VE:
        print(VE)


    
    for i in range(mesh_levels):
        run.energy_errors[i,0] = run.e_true - E_matrix[i,0]
    for j in range(1,mesh_levels):
        for i in range(mesh_levels-j):
            E_matrix[i,j] = run.RE(E_matrix[i,j-1],E_matrix[i+1,j-1],convergence_rate*j,refinement_rate)
            run.energy_errors[i,j]= run.e_true - E_matrix[i,j]
    for j in range(1,mesh_levels):
        for i in range(mesh_levels-j):
            Wave_matrix[i,j] = normalize( run.RE(Wave_matrix[i,j-1],Wave_matrix[i+1,j-1],convergence_rate*j,refinement_rate),dx )
            Wave_errors[i,j] = min( norm(Wave_matrix[i,j]-run.wave_true,dx), norm(Wave_matrix[i,j]+run.wave_true,dx) )

    for i in range(mesh_levels):
        Wave_errors[i,0] = min( norm(Wave_matrix[i,0]-run.wave_true,dx), norm(Wave_matrix[i,0]+run.wave_true,dx) )



    run.wave_matrix = Wave_matrix
    run.wave_errors = Wave_errors
    run.E_matrix = E_matrix

    # clean out values that don't exist
    for i in range(mesh_levels):
        for j in range(mesh_levels):
            if i+j >= mesh_levels:
                run.energy_errors[i,j] = None
                run.wave_errors[i,j] = None
                run.E_matrix[i,j] = None
    print('='*70)
    print('='*70+'\n' + '\n')
        
    print('Energy Results for level ', n)
    print('True Energy = ', run.e_true)
    print('Final Extrapolated Value = ', E_matrix[0,mesh_levels-1])
    print('-'*70)
    for i in range(mesh_levels):
        for j in range(mesh_levels-i):
            if j == 0:
                print('Absolute Energy Error at Mesh Level %2.2g = %2.2e' %(i , run.energy_errors[j,i]) )
            else:
                print('Absolute Energy Error at Mesh Level %2.2g = %2.2e' %(i , run.energy_errors[j,i]) , ', ratio = %.1f' %(run.energy_errors[j-1,i]/run.energy_errors[j,i] ) )
        print()

    print('Wavefunction Error') 
    print('-'*70)
    for i in range(mesh_levels):
        for j in range(mesh_levels-i):
            if j == 0:
                print('Wave Error at Mesh Level %2.2g = %2.2e' %(i , Wave_errors[j,i]) )
            else:
                print('Wave Error at Mesh Level %2.2g = %2.2e' %(i , Wave_errors[j,i]), ', ratio = %.1f' %(Wave_errors[j-1,i]/Wave_errors[j,i] ) )
        print()


       
    print('='*70,'\n')
##    cont = input('type \'yes\' to calculate another energy level...' + '\n')
    try:
        n = int(input('enter another energy level...' + '\n'))
        print('='*70,'\n')
        run = EO_driver(xmin,xmax,nx,D,psi_in,eig_tolerance,mesh_levels,refinement_rate,n)
    except ValueError as VE:
        return run
              
    return run



#################################################################################
#################################################################################
"""           END OF DRIVERS :: BEGINNING OF RUN INITIALIZATIONS             """ 
#################################################################################
#################################################################################




# for fsw
# need  self.xendpoints = np.linspace(self.xmin, self.xmax, self.nx+1)
# to put endpoints at +/- a
##D = 1
##a = 5*D
##len_ratio = 20
####xmin = -len_ratio*a
####xmax = len_ratio*a
####nx = int( len_ratio * 10 ) # using some multiple of len ratio ensures an endpoint occurs at +/- a
##xmin = -6
##xmax = 6
##nx = 100


# for others
D = 4
xmin = -10  # for PT, 10 is good for EO
xmax = 10
nx = 100


z_tolerance = 1e-13
mesh_levels = 5
energy_levels = 1
psi_in = np.random.rand(energy_levels,nx)
z_in = -1.1*np.ones((energy_levels,))
##z_in = 2*np.ones((energy_levels,))
refinement_rate = 3

runGI = GI_driver(xmin,xmax,nx,D,psi_in,z_in,z_tolerance,mesh_levels,energy_levels,refinement_rate)


eig_tolerance = 1e-14
psi_in = np.random.rand(nx)

runEO = EO_driver(xmin,xmax,nx,D,psi_in,eig_tolerance,mesh_levels,refinement_rate)


