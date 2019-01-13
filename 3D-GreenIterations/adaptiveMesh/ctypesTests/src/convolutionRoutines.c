#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define min(x,y) (((x) < (y)) ? (x) : (y))

//void directSum(int numTargets, int numSources,
//		double *targetX, double *targetY, double *targetZ, double *targetW,
//		double *sourceX, double *sourceY, double *sourceZ, double *sourceW,
//		double *outputArray) {
//
//	int i, j;
//	double xt, yt, zt, xs, ys, zs, r;
//    printf("\n Entered C Direct Sum routine: \n\n");
//
//
////#pragma omp parallel for num_threads(10) private(j,xt,yt,zt,xs,ys,zs,r)
//    #pragma omp parallel for num_threads(10) private(j,xt,yt,zt,r)
//    //#pragma omp parallel for num_threads(10) private(j)
//    for (i = 0; i < numTargets; i++) {
//    	outputArray[i] = 0.0;
//    	xt = targetX[i];
//    	yt = targetY[i];
//    	zt = targetZ[i];
//
//    	for (j=0; j< numSources; j++){
////    		xs = sourceX[j];
////    		ys = sourceY[j];
////    		zs = sourceZ[j];
////    		r = sqrt( (xt-xs)*(xt-xs) + (yt-ys)*(yt-ys) + (zt-zs)*(zt-zs)  );
//    		r = sqrt( (xt-sourceX[j])*(xt-sourceX[j]) + (yt-sourceY[j])*(yt-sourceY[j]) + (zt-sourceZ[j])*(zt-sourceZ[j])  );
//    		if (r > 0.0){
//    			outputArray[i] += sourceW[j]/r;
//    		}
////    		if ( i != j){
////    			outputArray[i] += sourceW[j]/sqrt( (targetX[i]-sourceX[j])*(targetX[i]-sourceX[j]) + (targetY[i]-sourceY[j])*(targetY[i]-sourceY[j]) + (targetZ[i]-sourceZ[j])*(targetZ[i]-sourceZ[j])  );
////    		}
//
//    	}
//
//    }
//
//    return;
//}


void directSum(int numTargets, int numSources,
		double *targetX, double *targetY, double *targetZ, double *targetW,
		double *sourceX, double *sourceY, double *sourceZ, double *sourceW,
		double *outputArray) {

	int i, j;
	double xt, yt, zt, xs, ys, zs, r;


#pragma omp parallel for private(j,xt,yt,zt,xs,ys,zs,r)
    for (i = 0; i < numTargets; i++) {
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

void directSum_HelmholtzSingularitySubtraction(int numTargets, int numSources, double helmholtzK,
		double *targetX, double *targetY, double *targetZ, double *targetW,
		double *sourceX, double *sourceY, double *sourceZ, double *sourceW,
		double *outputArray) {

	int i, j;
	double xt, yt, zt, xs, ys, zs, r;


#pragma omp parallel for private(j,xt,yt,zt,xs,ys,zs,r)
    for (i = 0; i < numTargets; i++) {
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






