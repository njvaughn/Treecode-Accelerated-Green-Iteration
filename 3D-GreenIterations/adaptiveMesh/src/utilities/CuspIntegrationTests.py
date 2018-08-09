'''
1D Cusp Integration

Test midpoint method with cusp located at the endpoint of an interval, 
at the center of an interval, or arbitraily within an interval.  
Test adaptive schemes that refine the region near the cusp.
'''

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def f(x,smooth=False):
    ''' The function with a cusp '''
    if smooth==False:
        return np.exp(-np.abs(x))
    else:
        return np.exp(-x)

def dfdx(x,smooth=False,oneSided=None):
    if smooth == False:
        if x>0:
            return -np.exp(-abs(x))
        elif x<0:
            return np.exp(-abs(x))
        
        if ( (x==0) and (oneSided=='right') ):
            return -np.exp(-abs(x))
        if ( (x==0) and (oneSided=='left') ):
            return np.exp(-abs(x))
        else:
            print('Warning: dfdx evaluation failed, possibly at x=0')
        
    else: # dealing wtih smooth function
        return -np.exp(-x)
    

def Fright(x):
    ''' The antiderivative of f right of the cusp'''
    return -np.exp(-np.abs(x))

def Fleft(x):
    ''' The antiderivative of f left of the cusp'''
    return np.exp(-np.abs(x))

def F_smooth(x):
    ''' The antiderivative of f left of the cusp'''
    return -np.exp(-x)

def analyticIntegral(xlow,xhigh,smooth=False):
    if smooth == False:
        return (Fright(xhigh) - Fright(0)) + (Fleft(0) - Fleft(xlow))
    else:
        return F_smooth(xhigh) - F_smooth(xlow)
    
def generateAdaptiveMesh(xlow,xhigh,tolerance,smooth=False,visualize=False,):
    refinementNeeded=True
    endpoints = np.linspace(xlow,xhigh,2,endpoint=True)
    midpoints = (endpoints[:-1] + endpoints[1:])/2
    
#     print(endpoints)
#     print(midpoints)
    while refinementNeeded:
#         print(endpoints)
        
        newMidpoints = []
        newEndpoints = np.array(endpoints[0])
        for i in range(len(midpoints)):
            dx = endpoints[i+1]-endpoints[i]
            parentIntegral = f(midpoints[i],smooth)*dx
            leftChild = (midpoints[i] + endpoints[i])/2
            rightChild = (midpoints[i] + endpoints[i+1])/2
            childIntegral = ( f(leftChild)+f(rightChild) ) * dx/2
            if abs(parentIntegral - childIntegral ) > tolerance:
#             if ( (abs(dx*dfdx(midpoints[i]))>tolerance) or (midpoints[i]==0.0 ) ):  # gradient tolerance
                newMidpoints = np.append(newMidpoints, leftChild)
                newMidpoints = np.append(newMidpoints, rightChild)
                newEndpoints = np.append(newEndpoints, midpoints[i])
                newEndpoints = np.append(newEndpoints, endpoints[i+1])
            else:
                newMidpoints = np.append(newMidpoints, midpoints[i])
                newEndpoints = np.append(newEndpoints, endpoints[i+1])
        
#         print(newMidpoints)    
#         print(len(midpoints))
#         print(len(newMidpoints))
#         print()
        if len(midpoints)==len(newMidpoints):
            midpoints = newMidpoints
            endpoints = newEndpoints
            refinementNeeded = False
        else:
            midpoints = newMidpoints
            endpoints = newEndpoints
#             refinementNeeded = False
    
    if visualize==True:
        plt.figure()
#         plt.plot(midpoints,np.ones(len(midpoints)),'.')
        plt.plot(midpoints,'.')
        plt.show()
    
    print(len(newMidpoints))
    
    return midpoints,endpoints
         
    

def generateUniformMesh(xlow,xhigh,N):
    endpoints = np.linspace(xlow,xhigh,N+1,endpoint=True)
    midpoints = (endpoints[:-1] + endpoints[1:])/2
    return midpoints,endpoints
    
def MidpointIntegration(xlow,xhigh,meshParameter,smooth=False,cuspLocation=None,meshType='uniform'):
#     if cuspLocation=='center':
#         if ( (xlow!=-xhigh) or (N%2==0) ):
#             print('These input parameters not guaranteed to give a cusp at the center of an interval')
#             return 0,0
#     if cuspLocation=='boundary':
#         if ( (xlow!=-xhigh) or (N%2!=0) ):
#             print('These input parameters not guaranteed to give a cusp at the boundary between intervals')
#             return 0,0
    if meshType == 'uniform':
        midpoints,endpoints = generateUniformMesh(xlow,xhigh,meshParameter)
    elif meshType == 'adaptive':
        midpoints,endpoints = generateAdaptiveMesh(xlow,xhigh,meshParameter)
    sum = 0.0
    for i in range(len(midpoints)):
        dx = endpoints[i+1]-endpoints[i]
        sum+= f(midpoints[i],smooth)*dx
    
    analyticValue = analyticIntegral(xlow,xhigh,smooth)
    error = analyticValue-sum
#     print('Midpoint method with %i intervals produced an error of %1.2e' %(len(midpoints),error))
    return sum, error

def BoundaryCorrectedMidpointIntegration(xlow,xhigh,meshParameter,smooth=False,cuspLocation=None,meshType='uniform'):

    if meshType == 'uniform':
        midpoints,endpoints = generateUniformMesh(xlow,xhigh,meshParameter)
    elif meshType == 'adaptive':
        midpoints,endpoints = generateAdaptiveMesh(xlow,xhigh,meshParameter)    
        
    sum = 0.0
    for i in range(len(midpoints)):
        dx = endpoints[i+1]-endpoints[i]
        if endpoints[i+1] == 0.0:
            sum+= f(midpoints[i],smooth)*dx + dx**2/24*(dfdx(endpoints[i+1],smooth,oneSided='left') - dfdx(endpoints[i],smooth))
        elif endpoints[i] == 0.0:
            sum+= f(midpoints[i],smooth)*dx + dx**2/24*(dfdx(endpoints[i+1],smooth) - dfdx(endpoints[i],smooth,oneSided='right'))
        else:
            sum+= f(midpoints[i],smooth)*dx + dx**2/24*(dfdx(endpoints[i+1],smooth) - dfdx(endpoints[i],smooth))
    
    analyticValue = analyticIntegral(xlow,xhigh,smooth)
    error = analyticValue-sum
#     print('Boundary Corrected Midpoint method with %i intervals produced an error of %1.2e' %(N,error))
    return sum, error

def SimpsonIntegration(xlow,xhigh,meshParameter,smooth=False,cuspLocation=None,meshType='uniform'):
#     if cuspLocation=='center':
#         if ( (xlow!=-xhigh) or (N%2==0) ):
#             print('These input parameters not guaranteed to give a cusp at the center of an interval')
#             return 0,0
#     if cuspLocation=='boundary':
#         if ( (xlow!=-xhigh) or (N%2!=0) ):
#             print('These input parameters not guaranteed to give a cusp at the boundary between intervals')
#             return 0,0
        
        
    if meshType == 'uniform':
        midpoints,endpoints = generateUniformMesh(xlow,xhigh,meshParameter)
    elif meshType == 'adaptive':
        midpoints,endpoints = generateAdaptiveMesh(xlow,xhigh,meshParameter)
    sum = 0.0
    for i in range(len(midpoints)):
        dx = endpoints[i+1]-endpoints[i]
        sum+= ( f(endpoints[i],smooth) +  4*f(midpoints[i],smooth) + f(endpoints[i+1],smooth) )*dx/6
    
    analyticValue = analyticIntegral(xlow,xhigh,smooth)
    error = analyticValue-sum
#     print('Simpson method with %i intervals produced an error of %1.2e' %(N,error))
    return sum, error

def refinementSweepUniform(meshLow, meshHigh, numMeshes,smooth=False):
    boundaryErrorsMidpoint    = np.zeros(numMeshes)
    boundaryValuesMidpoint    = np.zeros(numMeshes)
    centerErrorsMidpoint      = np.zeros(numMeshes)
    centerValuesMidpoint      = np.zeros(numMeshes)
    offCenterErrorsMidpoint   = np.zeros(numMeshes)
    offCenterValuesMidpoint   = np.zeros(numMeshes)
    
    boundaryErrorsSimpson     = np.zeros(numMeshes)
    boundaryValuesSimpson     = np.zeros(numMeshes)
    centerErrorsSimpson       = np.zeros(numMeshes)
    centerValuesSimpson       = np.zeros(numMeshes)
    offCenterErrorsSimpson    = np.zeros(numMeshes)
    offCenterValuesSimpson    = np.zeros(numMeshes)
    
    boundaryErrorsBCM         = np.zeros(numMeshes)
    boundaryValuesBCM         = np.zeros(numMeshes)
    centerValuesBCM           = np.zeros(numMeshes)
    centerErrorsBCM           = np.zeros(numMeshes)
    offCenterValuesBCM        = np.zeros(numMeshes)
    offCenterErrorsBCM        = np.zeros(numMeshes)
    
    
    N = np.logspace(meshLow, meshHigh, num=numMeshes, base=2.0)
    
    for i in range(N.size):
        boundaryValuesMidpoint[i],boundaryErrorsMidpoint[i] = MidpointIntegration(-5,5,N[i],smooth,cuspLocation='boundary')
    print()
    for i in range(N.size):
        centerValuesMidpoint[i],centerErrorsMidpoint[i] = MidpointIntegration(-5,5,N[i]+1,smooth,cuspLocation='center')
    print()   
    for i in range(N.size):
        offCenterValuesMidpoint[i],offCenterErrorsMidpoint[i] = MidpointIntegration(-5,5.2,N[i],smooth)
        
    for i in range(N.size):
        boundaryValuesSimpson[i],boundaryErrorsSimpson[i] = SimpsonIntegration(-5,5,N[i],smooth,cuspLocation='boundary')
    print()
    for i in range(N.size):
        centerValuesSimpson[i],centerErrorsSimpson[i] = SimpsonIntegration(-5,5,N[i]+1,smooth,cuspLocation='center')
    print()   
    for i in range(N.size):
        offCenterValuesSimpson[i],offCenterErrorsSimpson[i] = SimpsonIntegration(-5,5.2,N[i],smooth)
     
    for i in range(N.size):
        boundaryValuesBCM[i],boundaryErrorsBCM[i] = BoundaryCorrectedMidpointIntegration(-5,5,N[i],smooth,cuspLocation='boundary')
    print()
    for i in range(N.size):
        centerValuesBCM[i],centerErrorsBCM[i] = BoundaryCorrectedMidpointIntegration(-5,5,N[i]+1,smooth,cuspLocation='center')
    print()   
    for i in range(N.size):
        offCenterValuesBCM[i],offCenterErrorsBCM[i] = BoundaryCorrectedMidpointIntegration(-5,5.2,N[i],smooth)
     
     
    quadraticReference = N[0]**2/N**2*boundaryErrorsMidpoint[0]
    quarticReference   = N[0]**4/N**4*abs(boundaryErrorsSimpson[0])
    if smooth==False: 
        plt.figure()
        plt.title('Integrating Cusp exp(-|x|)')
        plt.xlabel('Number of Intervals')
        plt.ylabel('Absolute Error')
        plt.loglog(N,quadraticReference,'k-',label='1/N^2 reference')
#         plt.loglog(N,abs(boundaryErrorsMidpoint),'bd',label='Midpoint -- Cusp At Boundary')
#         plt.loglog(N,abs(centerErrorsMidpoint),'bo',label='Midpoint -- Cusp at Midpoint')
#         plt.loglog(N,abs(offCenterErrorsMidpoint),'bs',label='Midpoint -- Cusp Off-Center')
#         plt.loglog(N,abs(boundaryErrorsSimpson),'d',label='Simpson -- Cusp At Boundary')
#         plt.loglog(N,abs(centerErrorsSimpson),'o',label='Simpson -- Cusp at Midpoint')
#         plt.loglog(N,abs(offCenterErrorsSimpson),'s',label='Simpson -- Cusp Off-Center')
        plt.loglog(N,abs(boundaryErrorsBCM),'d',markerSize=5,label='BCM -- Cusp At Boundary')
        plt.loglog(N,abs(centerErrorsBCM),'o',markerSize=5,label='BCM -- Cusp at Midpoint')
        plt.loglog(N,abs(offCenterErrorsBCM),'s',markerSize=5,label='BCM -- Cusp Off-Center')
        plt.loglog(N,quarticReference,'k--',label='1/N^4 reference')
        plt.legend()
        plt.show()
        
    else:
        plt.figure()
        plt.title('Integrating Smooth exp(-x)')
        plt.xlabel('Number of Intervals')
        plt.ylabel('Absolute Error')
        plt.loglog(N,abs(boundaryErrorsMidpoint),'bd',label='Midpoint Method')
        plt.loglog(N,quadraticReference,'k-',label='1/N^2 reference')
        plt.loglog(N,abs(boundaryErrorsSimpson),'gd',label='Simpson Method')
        plt.loglog(N,quarticReference,'k--',label='1/N^4 reference')
        plt.loglog(N,abs(boundaryErrorsBCM),'rd',markerSize=4,label='Boundary Corrected Midpoint')
        plt.legend()
        plt.show()
        


def refinementAdaptive(divTolHigh, divTolLow, numMeshes,smooth=False):
    boundaryErrorsMidpoint    = np.zeros(numMeshes)
    boundaryValuesMidpoint    = np.zeros(numMeshes)
    boundaryErrorsSimpson     = np.zeros(numMeshes)
    boundaryValuesSimpson     = np.zeros(numMeshes)
    boundaryErrorsBCM         = np.zeros(numMeshes)
    boundaryValuesBCM         = np.zeros(numMeshes)
    N = np.zeros(numMeshes)
    
    divTol = np.logspace(divTolHigh, divTolLow, num=numMeshes, base=2.0)
    
    for i in range(N.size):
        print('Divide Tolerance: %1.3e' %divTol[i])
        midpoints,endpoints = generateAdaptiveMesh(-5, 5, divTol[i],visualize=False)
        N[i] = len(midpoints)
        boundaryValuesMidpoint[i],boundaryErrorsMidpoint[i] = MidpointIntegration(-5,5,divTol[i],smooth,cuspLocation='boundary',meshType='adaptive')
        boundaryValuesSimpson[i],boundaryErrorsSimpson[i] = SimpsonIntegration(-5,5,divTol[i],smooth,cuspLocation='boundary',meshType='adaptive')
        boundaryValuesBCM[i],boundaryErrorsBCM[i] = BoundaryCorrectedMidpointIntegration(-5,5,divTol[i],smooth,cuspLocation='boundary',meshType='adaptive')
    
#     for i in range(N.size):
# #         midpoints,endpoints = generateAdaptiveMesh(-5, 5, divTol[i],visualize=True)
# #         N[i] = len(midpoints)
# #         print('Divide Tolerance: %1.3e' %divTol[i])
#         boundaryValuesSimpson[i],boundaryErrorsSimpson[i] = MidpointIntegration(-5,5,divTol[i],smooth,cuspLocation='boundary',meshType='adaptive')
#     print()

    quadraticReference = N[1]**2/N**2*abs(boundaryErrorsMidpoint[1])
    quarticReference = N[1]**4/N**4*abs(boundaryErrorsSimpson[1])
    if smooth==False: 
        plt.figure()
        plt.title('Adaptive Mesh Integrating Cusp exp(-|x|)')
        plt.xlabel('Number of Intervals')
        plt.ylabel('Absolute Error')
        plt.loglog(N,quadraticReference,'k-',label='1/N^2 reference')
        plt.loglog(N,abs(boundaryErrorsMidpoint),'bd',label='Adaptive Mesh Midpoint -- Cusp At Boundary')
        plt.loglog(N,abs(boundaryErrorsSimpson ),'gd',label='Adaptive Mesh Simpson -- Cusp At Boundary')
        plt.loglog(N,abs(boundaryErrorsBCM ),'rd',label='Adaptive Mesh BCM -- Cusp At Boundary')
        plt.loglog(N,quarticReference,'k--',label='1/N^4 reference')
    #     plt.loglog(N,abs(centerErrorsMidpoint+2*boundaryErrorsMidpoint),'g^',label='Midpoint -- Linear Combination')
        plt.legend()
        plt.show()
        
    else:
        plt.figure()
        plt.title('Adaptive Mesh Integrating Smooth exp(-x)')
        plt.xlabel('Number of Intervals')
        plt.ylabel('Absolute Error')
        plt.loglog(N,quadraticReference,'k-',label='1/N^2 reference')
        plt.loglog(N,abs(boundaryErrorsMidpoint),'bd',label='Adaptive Mesh Midpoint')
        plt.loglog(N,abs(boundaryErrorsSimpson ),'go',label='Adaptive Mesh Simpson')
        plt.loglog(N,abs(boundaryErrorsBCM ),'rd',label='Adaptive Mesh BCM')
        plt.loglog(N,quarticReference,'k--',label='1/N^4 reference')

        plt.legend()
        plt.show()
        
        
if __name__=='__main__':
    low = 3
    high = 12
    numMeshes = high-low+1
    refinementSweepUniform(low,high,numMeshes,smooth=False)


#     divTolHigh = -1
#     divTolLow  = -40
#     refinementAdaptive(divTolHigh, divTolLow, 10,smooth=False)
