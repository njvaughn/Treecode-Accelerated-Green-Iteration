import numpy as np


def Green(x,y,z,E):
    k = np.sqrt(-2*E)
    if k<0:
        VE = 'ValueError: this Green function is valid for k>=0.'
        return VE
    else:
        r = np.sqrt(x*x + y*y + z*z)
        return -np.exp(-k*r)/(4*np.pi*r)  # Poisson equation Green's function if k = 0

def conv(V,E,psi,x,y,z):
    
    out = np.zeros(np.shape(psi))
    # loop through all target points
    for itarget in range(np.shape(out)[0]):
        for jtarget in range(np.shape(out)[1]):
            for ktarget in range(np.shape(out)[2]):
                # perform computation...
                for isource in range(np.shape(out)[0]):
                    for jsource in range(np.shape(out)[1]):
                        for ksource in range(np.shape(out)[2]):
                            dx = x[itarget,jtarget,ktarget] - x[isource,jsource,ksource]
                            dy = y[itarget,jtarget,ktarget] - y[isource,jsource,ksource]
                            dz = z[itarget,jtarget,ktarget] - z[isource,jsource,ksource]
                            if np.sqrt(dx*dx + dy*dy + dz*dz) > 1e-12:  # skip the singular Green function evaluations
                                out[itarget,jtarget,ktarget] += 2*V[isource,jsource,ksource]*psi[isource,jsource,ksource]*Green(dx,dy,dz,E)   
                                # Convolving 2*V*G with psi                       
    return out