#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <string.h>


/* Cell-wise interpolation routines */


void interapolateBetweenTwoMeshesSingleCell(double *coarseX, double *coarseY, double *coarseZ, double *coarseF, int pointsInCoarseCell, int coarseIdx,
                                            double *fineX,   double *fineY,   double *fineZ,   double *fineF,   int pointsInFineCell,   int fineIdx,
                                            int interpolationOrder){


    int interpOrderLim = interpolationOrder + 1;
    double nodeX[interpOrderLim], nodeY[interpOrderLim], nodeZ[interpOrderLim];
    double weights[interpOrderLim];

//    // Compute the interpolation weights
//    double *wx = malloc((interpOrderLim) * sizeof(double));
//    double *wy = malloc((interpOrderLim) * sizeof(double));
//    double *wz = malloc((interpOrderLim) * sizeof(double));
//



#ifdef OPENACC_ENABLED
    int streamID = rand() % 4;
    #pragma acc kernels async(streamID) present(coarseX, coarseY, coarseZ, coarseF, fineX,fineY,fineZ,fineF) \
                create(nodeX[0:interpOrderLim], nodeY[0:interpOrderLim], nodeZ[0:interpOrderLim], weights[0:interpOrderLim])
    {
#endif

    //  Fill in arrays of unique x, y, and z coordinates for the interpolation points.
#ifdef OPENACC_ENABLED
    #pragma acc loop independent
#endif
    for (int i = 0; i < interpOrderLim; i++) {
        nodeX[i] = coarseX[coarseIdx+i*interpOrderLim*interpOrderLim];
        nodeY[i] = coarseY[coarseIdx+i*interpOrderLim];
        nodeZ[i] = coarseZ[coarseIdx+i];
    }

#ifdef OPENACC_ENABLED
    #pragma acc loop independent
#endif
    for (int i=0;i<interpOrderLim;i++){
        weights[i] = pow(-1,i) * sin(  (2*i+1)*M_PI / (2*(interpOrderLim-1)+2)  );
    }

#ifdef OPENACC_ENABLED
    #pragma acc loop independent
#endif
    for (int i = 0; i < pointsInFineCell; i++) { // loop through the target points

        double sumX = 0.0;
        double sumY = 0.0;
        double sumZ = 0.0;

        double tx = fineX[fineIdx+i];
        double ty = fineY[fineIdx+i];
        double tz = fineZ[fineIdx+i];

        int eix = -1;
        int eiy = -1;
        int eiz = -1;

#ifdef OPENACC_ENABLED
        #pragma acc loop independent reduction(+:sumX,sumY,sumZ) reduction(max:eix,eiy,eiz)
#endif
        for (int j = 0; j < interpOrderLim; j++) {  // loop through the degree

            double cx = tx - nodeX[j];
            double cy = ty - nodeY[j];
            double cz = tz - nodeZ[j];

            if (fabs(cx)<DBL_MIN) eix = j;
            if (fabs(cy)<DBL_MIN) eiy = j;
            if (fabs(cz)<DBL_MIN) eiz = j;

            // Increment the sums
            double wx_t = weights[j];
            double wy_t = weights[j];
            double wz_t = weights[j];
            sumX += wx_t / cx;
            sumY += wy_t / cy;
            sumZ += wz_t / cz;

        }

        double denominator = 1.0;
        if (eix==-1) denominator /= sumX;
        if (eiy==-1) denominator /= sumY;
        if (eiz==-1) denominator /= sumZ;

        double temp = 0.0;

#ifdef OPENACC_ENABLED
        #pragma acc loop independent reduction(+:temp)
#endif
        for (int j = 0; j < pointsInCoarseCell; j++) { // loop over interpolation points, set (cx,cy,cz) for this point

            int k1 = j%interpOrderLim;
            int kk = (j-k1)/interpOrderLim;
            int k2 = kk%interpOrderLim;
            kk = kk - k2;
            int k3 = kk / interpOrderLim;

            double w3 = weights[k1];
            double w2 = weights[k2];
            double w1 = weights[k3];

            double cx = nodeX[k3];
            double cy = nodeY[k2];
            double cz = nodeZ[k1];
            double cq = coarseF[coarseIdx + j];

            double numerator = 1.0;

            // If exactInd[i] == -1, then no issues.
            // If exactInd[i] != -1, then we want to zero out terms EXCEPT when exactInd=k1.
            if (eix == -1) {
                numerator *= w1 / (tx - cx);
            } else {
                if (eix != k1) numerator *= 0;
            }

            if (eiy == -1) {
                numerator *= w2 / (ty - cy);
            } else {
                if (eiy != k2) numerator *= 0;
            }

            if (eiz == -1) {
                numerator *= w3 / (tz - cz);
            } else {
                if (eiz != k3) numerator *= 0;
            }

            temp += numerator * denominator * cq;
        }
#ifdef OPENACC_ENABLED
        #pragma acc atomic
#endif
        fineF[i + fineIdx] += temp;
    }
#ifdef OPENACC_ENABLED
    } //end ACC kernels
#endif
    return;
}









void InterpolateBetweenTwoMeshes(
        double *coarseX, double *coarseY, double *coarseZ, double *coarseF, int *pointsPerCell_coarse, int coarseN,
        double *fineX,   double *fineY,   double *fineZ,   double *fineF,   int *pointsPerCell_fine, int fineN, int numberOfCells, int cellOrder)
{

    double *fineF_sameAsCoarse = malloc((fineN) * sizeof(double));
    for (int i=0; i<fineN;i++){
        fineF_sameAsCoarse[i]=0.0;
    }

    int interpOrderLim=cellOrder+1;
//    printf("Beginning external interpolation.\n");
    if (coarseN==fineN){  // meshes are the same, don't need to interpolate.
        for (int i=0; i<coarseN;i++){
            fineF[i] = coarseF[i];
        }
        return;
    }

//    printf("coarseN, fineN = %i, %i\n", coarseN,fineN);



#ifdef OPENACC_ENABLED
    #pragma acc data copyin(coarseX[0:coarseN], coarseY[0:coarseN], coarseZ[0:coarseN], coarseF[0:coarseN], \
                            fineX[0:fineN], fineY[0:fineN], fineZ[0:fineN], \
                            pointsPerCell_coarse[0:numberOfCells], pointsPerCell_fine[0:numberOfCells]) \
                       copy(fineF[0:fineN])
    {
#endif



    // Loop over cells, calling interpolation routine

    int coarseIdx=0;
    int fineIdx=0;

    for (int i=0;i<numberOfCells;i++){
        if (pointsPerCell_coarse[i]==pointsPerCell_fine[i]){ // if coarse and fine cell are the same, simply fill fine with coarse.
            for (int j=0;j<pointsPerCell_fine[i];j++){
                fineF_sameAsCoarse[fineIdx+j] = coarseF[coarseIdx+j];
            }
        }else{


            interapolateBetweenTwoMeshesSingleCell( coarseX, coarseY, coarseZ, coarseF, pointsPerCell_coarse[i], coarseIdx,
                                                    fineX,   fineY,   fineZ,   fineF,   pointsPerCell_fine[i],   fineIdx,
                                                    cellOrder);
        }

        coarseIdx = coarseIdx + pointsPerCell_coarse[i];
        fineIdx = fineIdx + pointsPerCell_fine[i];
#ifdef OPENACC_ENABLED
    #pragma acc wait
#endif

//        printf("Cell %i of %i: Coarse idx, fine idx: %i and %i, after adding %i and %i\n", i, numberOfCells, coarseIdx,fineIdx,pointsPerCell_coarse[i],pointsPerCell_fine[i]);
    }

#ifdef OPENACC_ENABLED
    #pragma acc wait
    } // end ACC DATA REGION
#endif

    for (int i=0; i<fineN;i++){
        fineF[i] = fineF[i] + fineF_sameAsCoarse[i];
    }


    return;
}
