import numpy as np
import matplotlib.pyplot as plt



resultsFile='densities.npy'
# resultsDir = '/home/njvaughn/synchronizedDataFiles/krasnyMeshTests/Slice_Testing/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Slice_Testing/H2_LW5_200_SCF_91008_plots/'
# resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Slice_Testing/Be_LW5_200_SCF_75776_plots/'

resultsDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/benzeneTesting/Be_LW5_200_SCF_75776_plots/'


def plotSlicesTogether():
    
    densities = np.load(resultsDir+resultsFile)
    print(densities)
    (n,m) = np.shape(densities)
    print('Shape of densities: ', np.shape(densities))
    plt.figure()
    for i in range(1,m):
        plt.plot(densities[:,0], densities[:,i], label='SCF Iteration %i' %(i-1))
#         plt.semilogy(densities[:,0], densities[:,i], label='SCF Iteration %i' %(i-1))
     
    plt.legend()    
    
    plt.figure()
    for i in range(2,m):
#         plt.plot(densities[:,0], (densities[:,i]-densities[:,1])/densities[:,1], label='SCF Iteration %i' %(i-1))
        plt.semilogy(densities[:,0], np.abs(densities[:,i]-densities[:,1])/densities[:,1], label='SCF Iteration %i' %(i-1))
    plt.title('Relative Error with Respect to Initialized')
    plt.legend()
    
    
    plt.show()
#     return
 

if __name__=="__main__":
    plotSlicesTogether()
    