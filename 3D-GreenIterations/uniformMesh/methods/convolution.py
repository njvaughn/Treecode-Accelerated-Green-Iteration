<<<<<<< HEAD
from numpy import sqrt, exp, pi, zeros, shape

# 
# def Green(x,y,z,E):
# #     k = np.sqrt(-2*E)
# #     if k<0:
# #         VE = 'ValueError: this Green function is valid for k>=0.'
# #         return VE
# #     else:
#     r = sqrt(x*x + y*y + z*z)
#     return -exp(-sqrt(-2*E)*r)/(4*pi*r)  # Poisson equation Green's function if k = 0
# #     return -np.exp(-k*r)/(4*np.pi*r)  # Poisson equation Green's function if k = 0

def conv(V,E,psi,x,y,z):
    
    out = zeros(shape(psi))
    # loop through all target points
    arrshape = shape(out)
    for itarget in range(arrshape[0]):
        for jtarget in range(arrshape[1]):
            for ktarget in range(arrshape[2]):
                xt = x[itarget,jtarget,ktarget]
                yt = y[itarget,jtarget,ktarget]
                zt = z[itarget,jtarget,ktarget]
                # perform computation...
                for isource in range(arrshape[0]):
                    for jsource in range(arrshape[1]):
                        for ksource in range(arrshape[2]):
                            dx = xt - x[isource,jsource,ksource]
                            dy = yt - y[isource,jsource,ksource]
                            dz = zt - z[isource,jsource,ksource]
                            r = sqrt(dx*dx + dy*dy + dz*dz)
                            if r > 0:
#                             if dx*dx + dy*dy + dz*dz > 1e-20:  # skip the singular Green function evaluations
#                                 out[itarget,jtarget,ktarget] += 2*V[isource,jsource,ksource]*psi[isource,jsource,ksource]*Green(dx,dy,dz,E)   
                                out[itarget,jtarget,ktarget] -= 2*V[isource,jsource,ksource]*psi[isource,jsource,ksource]*exp(-sqrt(-2*E)*r)/(4*pi*r)  
                                # Convolving 2*V*G with psi                       
    return out



# def conv_for_vectorization(V,E,psi,x,y,z):
#     out = zeros(shape(psi))
#     arrshape = shape(out)
#     xt = x
#     yt = y
#     zt = z
#     # perform computation...
#     for isource in range(arrshape[0]):
#         for jsource in range(arrshape[1]):
#             for ksource in range(arrshape[2]):
#                 dx = xt - x[isource,jsource,ksource]
#                 dy = yt - y[isource,jsource,ksource]
#                 dz = zt - z[isource,jsource,ksource]
#                 r = sqrt(dx*dx + dy*dy + dz*dz)
#                 if r > 0:
#                     out -= 2*V[isource,jsource,ksource]*psi[isource,jsource,ksource]*exp(-sqrt(-2*E)*r)/(4*pi*r)  
#                     # Convolving 2*V*G with psi  
#     return out

=======
from numpy import sqrt, exp, pi, zeros, shape

# 
# def Green(x,y,z,E):
# #     k = np.sqrt(-2*E)
# #     if k<0:
# #         VE = 'ValueError: this Green function is valid for k>=0.'
# #         return VE
# #     else:
#     r = sqrt(x*x + y*y + z*z)
#     return -exp(-sqrt(-2*E)*r)/(4*pi*r)  # Poisson equation Green's function if k = 0
# #     return -np.exp(-k*r)/(4*np.pi*r)  # Poisson equation Green's function if k = 0

def conv(V,E,psi,x,y,z):
    
    out = zeros(shape(psi))
    # loop through all target points
    arrshape = shape(out)
    for itarget in range(arrshape[0]):
        for jtarget in range(arrshape[1]):
            for ktarget in range(arrshape[2]):
                xt = x[itarget,jtarget,ktarget]
                yt = y[itarget,jtarget,ktarget]
                zt = z[itarget,jtarget,ktarget]
                # perform computation...
                for isource in range(arrshape[0]):
                    for jsource in range(arrshape[1]):
                        for ksource in range(arrshape[2]):
                            dx = xt - x[isource,jsource,ksource]
                            dy = yt - y[isource,jsource,ksource]
                            dz = zt - z[isource,jsource,ksource]
                            r = sqrt(dx*dx + dy*dy + dz*dz)
                            if r > 0:
#                             if dx*dx + dy*dy + dz*dz > 1e-20:  # skip the singular Green function evaluations
#                                 out[itarget,jtarget,ktarget] += 2*V[isource,jsource,ksource]*psi[isource,jsource,ksource]*Green(dx,dy,dz,E)   
                                out[itarget,jtarget,ktarget] -= 2*V[isource,jsource,ksource]*psi[isource,jsource,ksource]*exp(-sqrt(-2*E)*r)/(4*pi*r)  
                                # Convolving 2*V*G with psi                       
    return out



# def conv_for_vectorization(V,E,psi,x,y,z):
#     out = zeros(shape(psi))
#     arrshape = shape(out)
#     xt = x
#     yt = y
#     zt = z
#     # perform computation...
#     for isource in range(arrshape[0]):
#         for jsource in range(arrshape[1]):
#             for ksource in range(arrshape[2]):
#                 dx = xt - x[isource,jsource,ksource]
#                 dy = yt - y[isource,jsource,ksource]
#                 dz = zt - z[isource,jsource,ksource]
#                 r = sqrt(dx*dx + dy*dy + dz*dz)
#                 if r > 0:
#                     out -= 2*V[isource,jsource,ksource]*psi[isource,jsource,ksource]*exp(-sqrt(-2*E)*r)/(4*pi*r)  
#                     # Convolving 2*V*G with psi  
#     return out

>>>>>>> refs/remotes/eclipse_auto/master
    