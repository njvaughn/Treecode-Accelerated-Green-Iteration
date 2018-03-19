import numpy as np
from cellDataStructure import cell
from hydrogenPotential import trueWavefunction, smoothedPotential




class Mesh(object):
    '''
    Given domain parameters, generate a uniform (coarse) structure of cells
    '''
    
    
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, nx, ny, nz, initialWavefunction, initialNormalization = "Midpoint"):
        """
        Initialize a 3D array of cells containing the analytic psi (for testing)
        """
        self.xmesh = np.linspace(xmin,xmax,2*nx+1)
        self.ymesh = np.linspace(ymin,ymax,2*ny+1)
        self.zmesh = np.linspace(zmin,zmax,2*nz+1)
        
        self.cells = np.empty((nx*ny*nz),dtype=object)
        for i in range(nx):
            x = self.xmesh[i*2:i*2+3]
            for j in range(ny):
                y = self.ymesh[j*2:j*2+3]
                for k in range(nz):
                    z = self.zmesh[k*2:k*2+3]
                    xm,ym,zm = np.meshgrid(x,y,z,indexing='ij')
                    if initialWavefunction == "analytic":
                        psi = trueWavefunction(1, xm, ym, zm)
                    elif initialWavefunction == "random":
                        psi = np.random.rand(3,3,3)
                    else:
                        print("warning: initial psi needs option 'analytic' or 'random'")
                    index = ny*nz*i + nz*j + k
                    self.cells[index] = cell(x,y,z,psi)
        if initialNormalization == "Midpoint":
            W = self.MidpointWeightMatrix()
        elif initialNormalization == "Simpson":
            W = self.SimpsonWeightMatrix()
        else:
            print('Error, give an initial normalization method')           
        self.normalizePsi(W)
    
    def MeshRefinement(self,psiVariationThreshold):
        count = 0
        numcells = len(self.cells)
        children_array = np.array([])
        for index in range(numcells):  # note, after one level of refinement won't know the shape of cells necessarily
            self.cells[index].checkDivide(psiVariationThreshold)
            if self.cells[index].NeedsDividing == True:
                count+=1           
                children = self.cells[index].divide().flatten()     
                children_array = np.concatenate([children_array, children])
        print('dividing ', count,' cells')
        indices_for_deleting = []
        for index in range(numcells):
            if self.cells[index].NeedsDividing == True:
                indices_for_deleting.append(index)
        self.cells = np.delete(self.cells,indices_for_deleting) # remove the divided cells
        self.cells = np.concatenate([self.cells, children_array]) # append the children cells
    
    def extractMeshFromCellArray(self):
        x = np.array([])
        y = np.array([])
        z = np.array([])
        psi = np.array([])
        for index in range(len(self.cells)):
            x = np.append(x,self.cells[index].x[1])
            y = np.append(y,self.cells[index].y[1])
            z = np.append(z,self.cells[index].z[1])
            psi = np.append(psi, self.cells[index].psi[1,1,1])
        return x,y,z,psi
        
    def normalizePsi(self,W):
        '''
        Normalize Psi so that integral W*psi^2 dxdydz = 1 for chosen weight matrix W
        '''
        sumPsiSquaredDxDyDz = 0.0
        for index in range(len(self.cells)):
            sumPsiSquaredDxDyDz += np.sum( W*self.cells[index].psi**2*self.cells[index].volume )
        for index in range(len(self.cells)):
            self.cells[index].psi /= np.sqrt(sumPsiSquaredDxDyDz)
    
    def computeKinetic(self,W):
        '''
        Compute the kinetic energy over the entire domain
        '''
        computedKinetic = 0.0
        for index in range(len(self.cells)):                
                # evaluate kinetic due to this cell
#                 self.cells[index].evaluateKinetic_MidpointMethod()
                self.cells[index].evaluateKinetic(W)
                # increment kinetic
                computedKinetic += self.cells[index].Kinetic
        # set the mesh's kinetic energy value, corresponding to total over whole domain
        self.Kinetic = computedKinetic
        
    def setExactWavefunction(self):
        '''
        Set equal to analytic wavefunction for hydrogen test case, want to use true wavefunction values not interpolated values
        '''
        for index in range(len(self.cells)): 
                xm,ym,zm = np.meshgrid(self.cells[index].x,self.cells[index].y,
                            self.cells[index].z,indexing='ij')
                self.cells[index].psi = trueWavefunction(1, xm, ym, zm)
                    
    def MidpointWeightMatrix(self):
        W = np.zeros((3,3,3))
        W[1,1,1] = 1
        return W

    def SimpsonWeightMatrix(self):
        w1 = np.array([1, 4, 1])/3/2
        w2 = np.multiply.outer(w1,w1)
        w3 = np.multiply.outer(w2,w1)
        return w3
#         w1 = np.array([[ 1, 4, 1],
#                        [ 4,16, 4],
#                        [ 1, 4, 1]])
#         w2 = np.array([[ 4,16, 4],
#                        [16,64,16],
#                        [ 4,16, 4]])
#         w3 = w1
#         W = np.array([w1,w2,w3])/27/8
#         return W

    
    def computePotential(self,W,epsilon):
        '''
        Compute the potential energy over the entire domain
        '''
        computedPotential = 0.0
        for index in range(len(self.cells)):                
                # evaluate potential due to this cell
#                 self.cells[index].evaluateKinetic_MidpointMethod()
                self.cells[index].evaluatePotential(W,epsilon)
                # increment potential
                computedPotential += self.cells[index].Potential
        # set the mesh's potential energy value, corresponding to total over whole domain
        self.Potential = computedPotential

    
    def convolution(self, targetCell,W,E):
        psiNew = np.zeros((3,3,3))
        eps = 0.05
        # loop over the 27 points of the target cell
        for i in range(1):
            xt = targetCell.x[1]
            for j in range(1):
                yt = targetCell.y[1]
                for k in range(1):
                    zt = targetCell.z[1]
                    
                    # loop over all source cells, interact with
#                     print('ijk',i,j,k)
                    for index in range(len(self.cells)):
#                         xs,ys,zs = np.meshgrid(self.cells[index].x,self.cells[index].y,self.cells[index].z, indexing='ij')
#                         r = np.sqrt( (xt-xs)**2 + (yt-ys)**2 + (zt-zs)**2 + eps**2 )
#                         G = np.exp(-np.sqrt(-2*E)*r)/(4*np.pi*r)
#                         V = smoothedPotential(xt-xs,yt-ys,zt-zs,eps)
                        rMid = np.sqrt( (xt-self.cells[index].x[1])**2 + (yt-self.cells[index].y[1])**2 + (zt-self.cells[index].z[1])**2 )
                        if rMid == 0:
                            GMid = 0
                        else:
                            GMid = np.exp(-np.sqrt(-2*E)*rMid)/(4*np.pi*rMid)
                        VMid = smoothedPotential(xt-self.cells[index].x[1],yt-self.cells[index].y[1],zt-self.cells[index].z[1],eps)
                        psiSourceMid = self.cells[index].psi[1,1,1]
                        dxdydz = self.cells[index].volume
#                         psiNew[i,j,k] += -2*np.sum(W*V*G*psiSource*dxdydz)
                        psiNew[1,1,1] += -2*VMid*GMid*psiSourceMid*dxdydz
        
        return psiNew
                        
                    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                
            
    
    
  
        
        
    
    



