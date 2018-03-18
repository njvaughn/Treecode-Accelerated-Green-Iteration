from numba import cuda
from cmath import sqrt,exp
        
@cuda.jit
def gpuConvolution(targets,sources,psiNew,E):
    globalID = cuda.grid(1)
#     x_t, y_t, z_t, psi_t, V_t, volume_t = targets[globalID]
    x_t, y_t, z_t = targets[globalID][0:3]
    for i in range(len(sources)):
        x_s, y_s, z_s, psi_s, V_s, volume_s = sources[i]
        if not ( (x_s==x_t) and (y_s==y_t) and (z_s==z_t) ): 
            r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 )
            psiNew[globalID] += -2*V_s*volume_s*psi_s*exp(-sqrt(-2*E)*r)/r
            
            
# testing push