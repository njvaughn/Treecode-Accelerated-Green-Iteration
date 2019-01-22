#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define min(x,y) (((x) < (y)) ? (x) : (y))

void modifiedGramSchmidt_singleOrbital_C(int numPoints, int numWavefunctions, int targetWavefunction,
		double **wavefunctions, double *weights) {

	int i, j;

	double *U = V[targetWavefunction];
//	double xt, yt, zt, xs, ys, zs, r;


#pragma omp parallel for private(j,xt,yt,zt,xs,ys,zs,r)
    for (i = 0; i < targetWavefunction; i++) {
    	printf("Orthogonalizing %d against %d",targetWavefunction,i)
    	outputArray[i] = 0.0;
    	xt = targetX[i];
    	yt = targetY[i];
    	zt = targetZ[i];

    	for (j=0; j< numSources; j++){
    		xs = sourceX[j];
    		ys = sourceY[j];
    		zs = sourceZ[j];
    		r = sqrt( (xt-xs)*(xt-xs) + (yt-ys)*(yt-ys) + (zt-zs)*(zt-zs)  );
    		if (r > 0.0){
    			outputArray[i] += sourceW[j]/r;
    		}

    	}

    }

    return;
}


//def modifiedGramSchmidt_singleOrbital(V,weights,targetOrbital):
//    n,k = np.shape(V)
//    U = V[:,targetOrbital]
//    for j in range(targetOrbital):
//#         print('Orthogonalizing %i against %i' %(targetOrbital,j))
//#         U -= (np.dot(V[:,targetOrbital],V[:,j]*weights) / np.dot(V[:,j],V[:,j]*weights))*V[:,j]
//        U -= np.dot(V[:,targetOrbital],V[:,j]*weights) *V[:,j]
//        U /= np.sqrt( np.dot(U,U*weights) )
//
//    U /= np.sqrt( np.dot(U,U*weights) )  # normalize again at end (safegaurd for the zeroth orbital, which doesn't enter the above loop)
//
//    return U
