import numpy as np
from math import factorial
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def G(r):
    return -np.exp(-r)/r

def G_ep(r,epsilon):
    return -np.exp(-r)/np.sqrt(r**2 + epsilon**2)

def G_n(r,epsilon,n):
    Gvalue = 0.0
#     for i in range(n+1):
#         coefficient = 1.0
#         for k in range(i):
#             coefficient /= (k+1) # this is replacing the 1/factorial(i)
#             coefficient *= ((-1/2)-k)
#         print('%ith coefficient: %f' %(i,coefficient))
#         
#         Gvalue += coefficient* (-epsilon**2)**i * (r**2 + epsilon**2)**(-1/2-i)
        
        
        
    for ii in range(n+1):
        coefficient = 1.0
        for jj in range(ii):
            coefficient /= (jj+1) # this is replacing the 1/factorial(i)
            coefficient *= ((-1/2)-jj)
        print('%ith coefficient: %f' %(ii,coefficient))                  
        Gvalue += coefficient* (-epsilon**2)**ii * (r**2 + epsilon**2)**(-1/2-ii)
                    
    return -np.exp(-r)*Gvalue
        
def plot_fixed_epsilon(rmax,nmax,epsilon):
    r = np.linspace(0,rmax,1000)
    plt.figure()
    plt.title('Varying n for epsilon = %1.3f'%epsilon)
    plt.plot(r[1:],G(r[1:]),'k--',label='exp(-r)/r')

    for n in range(nmax)[::3]:
        plt.plot(r,G_n(r,epsilon,n), label='n=%i'%n )
    plt.legend()
    plt.ylim([-50,1])
    plt.show()    
    
def plot_fixed_n(rmax,epsilon_min, epsilon_max, nepsilon, n):
    r = np.linspace(0,rmax,1000)
    plt.figure()
    plt.title('Varying epsilon for n = %i'%n)
    plt.plot(r[1:],G(r[1:]),'k--',label='1/r')
    
    epsilon_vec = np.logspace(epsilon_min,epsilon_max,nepsilon,base=2)
    print(epsilon_vec)
    for epsilon in epsilon_vec[::2]:
#         print()
#         print(G_n(r,epsilon,n))
        plt.plot(r,G_n(r,epsilon,n), label='epsilon=%1.4f'%epsilon )
    plt.legend()
    plt.ylim([-50,1])
    plt.show()       
        
        
if __name__=="__main__":
    
    G_n(1,1,5)
    plot_fixed_epsilon(0.5,13,0.25)
#     plot_fixed_n(1, -1, -7, 7, 2)

#     print(np.logspace(0,-3,4))

#     r = np.linspace(0,1,10)
#     G_n_vec = np.vectorize(G_n)
#     print(r)
#     print()
#     print(G(r[1:]))
#     print()
#     print(G_ep(r,0.1))
#     print()
#     print( G_n(r,0.1,0) )    
#     print( G_n_vec(r,0.0,0) )