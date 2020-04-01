#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>


/* Data Movement routines */


#include "moveData.h"

void copyToDevice(double *X, double *Y, double *Z, double *W, double **V, int numPoints, int numV)
{

#ifdef OPENACC_ENABLED
    #pragma acc enter data copyin(X[0:numPoints],Y[0:numPoints],Z[0:numPoints],W[0:numPoints],V[0:numV][0:numPoints])
#endif
    return;
}

void copyVectorToDevice(double *U, int numPoints)
{

#ifdef OPENACC_ENABLED
    #pragma acc enter data copyin(U[0:numPoints])
    #pragma acc wait
#endif
    return;
}

void copyMatrixToDevice(double **V, int numPoints, int numV)
{

#ifdef OPENACC_ENABLED
    #pragma acc enter data copyin(V[0:numV][0:numPoints])
#endif
    return;
}

void copyVectorFromDevice(double *U, int numPoints)
{

#ifdef OPENACC_ENABLED
    #pragma acc exit data copyout(U[0:numPoints])
#endif
    return;
}

void copyMatrixFromDevice(double **V, int numPoints, int numV)
{

#ifdef OPENACC_ENABLED
    #pragma acc exit data copyout(V[0:numV][0:numPoints])
#endif
    return;
}


void copyFromDevice(double *X, double *Y, double *Z, double *W, double **V, int numPoints, int numV)
{

#ifdef OPENACC_ENABLED
    #pragma acc exit data copyout(X[0:numPoints],Y[0:numPoints],Z[0:numPoints],W[0:numPoints],V[0:numV][0:numPoints])
#endif
    return;
}

void removeVectorFromDevice(double *U, int numPoints)
{

#ifdef OPENACC_ENABLED
    #pragma acc exit data delete(U[0:numPoints])
#endif
    return;
}


void removeMatrixFromDevice(double **V, int numPoints, int numV)
{

#ifdef OPENACC_ENABLED
    #pragma acc exit data delete(V[0:numV][0:numPoints])
#endif
    return;
}


