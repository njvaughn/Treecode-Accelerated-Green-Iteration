import numpy as np




SCFcount=0
TwoMeshStart=0
mixingHistoryCutoff=10

nPoints=5
inputDensities = np.zeros((nPoints,1))
outputDensities = np.zeros((nPoints,1))

# initialize input density
inputDensities[:,0] = np.copy(np.ones(nPoints))
# print(inputDensities)



for xxx in range(22):

    SCFcount += 1
    print('\nSCF Count ', SCFcount)
    SCFindex = SCFcount
    if SCFcount>TwoMeshStart:
        SCFindex = SCFcount - TwoMeshStart
        
    
        
    # fill inputs   
    if SCFcount>1: 
        if (SCFindex-1)<mixingHistoryCutoff:
            inputDensities = np.concatenate( (inputDensities, np.reshape(SCFcount*np.ones(nPoints), (nPoints,1))), axis=1)
        else:
            inputDensities[:,(SCFindex-1)%mixingHistoryCutoff] = np.copy(SCFcount*np.ones(nPoints))
    
        if SCFcount == TwoMeshStart:
            inputDensities = np.zeros((nPoints,1))         
            inputDensities[:,0] = np.copy(SCFcount*np.ones(nPoints))
        
    
    # fill outputs
    if SCFcount==1: 
       outputDensities[:,0] = np.copy(-1*SCFcount*np.ones(nPoints))
    else:

#         if (len(outputDensities[0,:]))<mixingHistoryCutoff:
        if (SCFindex-1)<mixingHistoryCutoff:
            outputDensities = np.concatenate( (outputDensities, np.reshape(-1*SCFcount*np.ones(nPoints), (nPoints,1))), axis=1)
        else:
            outputDensities[:,(SCFindex-1)%mixingHistoryCutoff] = -1*SCFcount*np.ones(nPoints)
        
        if SCFcount == TwoMeshStart:
            outputDensities = np.zeros((nPoints,1))         
            outputDensities[:,0] = np.copy(-1*SCFcount*np.ones(nPoints))
    
        
        
        
        
    print(inputDensities)
    print(outputDensities)
    
    print('\n\n')