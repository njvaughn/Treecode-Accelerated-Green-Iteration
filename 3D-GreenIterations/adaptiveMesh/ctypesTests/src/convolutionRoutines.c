#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define min(x,y) (((x) < (y)) ? (x) : (y))



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
		double *targetX, double *targetY, double *targetZ, double *targetValue, double *targetWeight,
		double *sourceX, double *sourceY, double *sourceZ, double *sourceValue, double *sourceWeight,
		double *outputArray) {

	int i, j;
	double xt, yt, zt, xs, ys, zs, r, targetVal;


#pragma omp parallel for private(j,xt,yt,zt,xs,ys,zs,r, targetVal)
    for (i = 0; i < numTargets; i++) {
    	outputArray[i] = 0.0;
    	xt = targetX[i];
    	yt = targetY[i];
    	zt = targetZ[i];

    	targetVal = targetValue[i];

    	for (j=0; j< numSources; j++){
    		xs = sourceX[j];
    		ys = sourceY[j];
    		zs = sourceZ[j];
    		r = sqrt( (xt-xs)*(xt-xs) + (yt-ys)*(yt-ys) + (zt-zs)*(zt-zs)  );
//    		if (r > 0.0){
    		if (i!=j){
    			outputArray[i] += sourceWeight[j]*( sourceValue[j] - targetVal )*exp(-helmholtzK*r) /r;
    		}

//		outputArray[i] /= 4*M_PI;

    	}

    }

    return;
}


void directSum_PoissonSingularitySubtraction(int numTargets, int numSources, double gaussianAlphaSq,
		double *targetX, double *targetY, double *targetZ, double *targetValue, double *targetWeight,
		double *sourceX, double *sourceY, double *sourceZ, double *sourceValue, double *sourceWeight,
		double *outputArray) {


	int i, j;
	double xt, yt, zt, xs, ys, zs, r, targetVal;



#pragma omp parallel for private(j,xt,yt,zt,xs,ys,zs,r, targetVal)
		for (i = 0; i < numTargets; i++) {
			outputArray[i] = 0.0;
			xt = targetX[i];
			yt = targetY[i];
			zt = targetZ[i];

			targetVal = targetValue[i];

			for (j=0; j< numSources; j++){
				xs = sourceX[j];
				ys = sourceY[j];
				zs = sourceZ[j];
				r = sqrt( (xt-xs)*(xt-xs) + (yt-ys)*(yt-ys) + (zt-zs)*(zt-zs)  );
				if (i!=j){
					outputArray[i] += sourceWeight[j] * ( sourceValue[j] - targetVal*exp(-gaussianAlphaSq*r*r) ) /r;
				}


			}

		}

    return;
}


void directSum_Poisson(int numTargets, int numSources,
		double *targetX, double *targetY, double *targetZ, double *targetValue, double *targetWeight,
		double *sourceX, double *sourceY, double *sourceZ, double *sourceValue, double *sourceWeight,
		double *outputArray) {


	int i, j;
	double xt, yt, zt, xs, ys, zs, r, targetVal;



#pragma omp parallel for private(j,xt,yt,zt,xs,ys,zs,r, targetVal)
		for (i = 0; i < numTargets; i++) {
			outputArray[i] = 0.0;
			xt = targetX[i];
			yt = targetY[i];
			zt = targetZ[i];

			targetVal = targetValue[i];

			for (j=0; j< numSources; j++){
				xs = sourceX[j];
				ys = sourceY[j];
				zs = sourceZ[j];
				r = sqrt( (xt-xs)*(xt-xs) + (yt-ys)*(yt-ys) + (zt-zs)*(zt-zs)  );
				if (i!=j){
					outputArray[i] += sourceWeight[j] * ( sourceValue[j] ) /r;
				}


			}

		}

    return;
}






