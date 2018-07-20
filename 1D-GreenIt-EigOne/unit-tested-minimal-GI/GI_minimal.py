<<<<<<< HEAD
from model import Model
import numpy as np

class GIrun(Model):
    
    def __init__(self, model_name, nx, xmin, xmax, D):
        ''' raw input attributes '''
        Model.__init__(self, model_name)
        self.nx = nx
        self.xmin = xmin
        self.xmax = xmax
        self.D = D
        ''' computed and derived attributes '''
        self.dx = (self.xmax - self.xmin)/(self.nx)
        xendpoints = np.linspace(self.xmin, self.xmax, self.nx+1) # set up nx+1 endpoints containing nx intervals.  Next identify midpoints
        self.xgrid = np.array([(xendpoints[i+1]+xendpoints[i])/2 for i in range(len(xendpoints)-1)])
        self.V = self.potential(self.xgrid,self.D)
    
    def Hamiltonian(self):
        '''
        Construct Hamiltonian using 2nd order central differences
        '''
        H = ( -(1/2)*(1/self.dx**2)*( np.diag(np.ones(self.nx-1),-1) + 
        np.diag(np.ones(self.nx-1),1) -np.diag(2*np.ones(self.nx)) ) +
        np.diag(self.V) )
        return H

    def G(self, x, z):
        return -1/(2*np.sqrt(-2*z))*np.exp(-np.sqrt(-2*z)*x)
    
    def ConvolutionOperator(self, z):
        return self.dx*np.fromfunction(lambda i, ii:
                2*self.V[ii]*self.G(np.abs(
                self.xgrid[i]-self.xgrid[ii]),z), (self.nx,self.nx),dtype=int)
    
    def normalize(self, psi):
        ''' integral psi^2 dx = 1 '''
        return psi/np.sqrt(np.sum(psi**2)*self.dx)

    def compute_energy(self, eig_tol, z_in, psi_in):
        '''
        Compute the energy using Green Iterations.  Without including orthogonalization, 
        this will converge to the ground state (when it converges).
        :param eig_tol: tolerance for the energy eigenvalue convergence
        :param z_in: input energy value
        :param psi_in: input wavefunction guess
        '''
        H = self.Hamiltonian()
        psi = self.normalize(psi_in)
        z_new = z_in
        deltaZ = 1
        while deltaZ > eig_tol:
            z_old = z_new
            psi = self.normalize( np.dot( self.ConvolutionOperator(z_old),psi))
            z_new = np.dot(np.dot( H, psi), psi*self.dx)
            deltaZ = np.abs(z_old-z_new) 
        return z_new
    

=======
from model import Model
import numpy as np

class GIrun(Model):
    
    def __init__(self, model_name, nx, xmin, xmax, D):
        ''' raw input attributes '''
        Model.__init__(self, model_name)
        self.nx = nx
        self.xmin = xmin
        self.xmax = xmax
        self.D = D
        ''' computed and derived attributes '''
        self.dx = (self.xmax - self.xmin)/(self.nx)
        xendpoints = np.linspace(self.xmin, self.xmax, self.nx+1) # set up nx+1 endpoints containing nx intervals.  Next identify midpoints
        self.xgrid = np.array([(xendpoints[i+1]+xendpoints[i])/2 for i in range(len(xendpoints)-1)])
        self.V = self.potential(self.xgrid,self.D)
    
    def Hamiltonian(self):
        '''
        Construct Hamiltonian using 2nd order central differences
        '''
        H = ( -(1/2)*(1/self.dx**2)*( np.diag(np.ones(self.nx-1),-1) + 
        np.diag(np.ones(self.nx-1),1) -np.diag(2*np.ones(self.nx)) ) +
        np.diag(self.V) )
        return H

    def G(self, x, z):
        return -1/(2*np.sqrt(-2*z))*np.exp(-np.sqrt(-2*z)*x)
    
    def ConvolutionOperator(self, z):
        return self.dx*np.fromfunction(lambda i, ii:
                2*self.V[ii]*self.G(np.abs(
                self.xgrid[i]-self.xgrid[ii]),z), (self.nx,self.nx),dtype=int)
    
    def normalize(self, psi):
        ''' integral psi^2 dx = 1 '''
        return psi/np.sqrt(np.sum(psi**2)*self.dx)

    def compute_energy(self, eig_tol, z_in, psi_in):
        '''
        Compute the energy using Green Iterations.  Without including orthogonalization, 
        this will converge to the ground state (when it converges).
        :param eig_tol: tolerance for the energy eigenvalue convergence
        :param z_in: input energy value
        :param psi_in: input wavefunction guess
        '''
        H = self.Hamiltonian()
        psi = self.normalize(psi_in)
        z_new = z_in
        deltaZ = 1
        while deltaZ > eig_tol:
            z_old = z_new
            psi = self.normalize( np.dot( self.ConvolutionOperator(z_old),psi))
            z_new = np.dot(np.dot( H, psi), psi*self.dx)
            deltaZ = np.abs(z_old-z_new) 
        return z_new
    

>>>>>>> refs/remotes/eclipse_auto/master
