#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <float.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>


/* Orthogonalization routines */
#include "Orthogonalization.h"

void modifiedGramSchmidt_singleWavefunction(double *V, double *U, double *W, int targetWavefunction, int numPoints, int numWavefunctions){

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

//#ifdef OPENACC_ENABLED
////#pragma acc data copyin(W) copy(V) create(r_local,r_global)
//    #pragma acc enter data copyin(V[0:numWavefunctions][0:numPoints])
//#endif

    // Step 1:  Compute all local dot products on GPU.  If the target is 6, there need to orthogonalize against 0-5.  Target 6 is readlly the 7th wavefunction.
    double *r_local = malloc((targetWavefunction) * sizeof(double));  // r[j] stores the inner product between the target and jth wavefunctions.
    double *r_global = malloc((targetWavefunction) * sizeof(double));  // r[j] stores the inner product between the target and jth wavefunctions.

    printf("Made it here0.\n");

//#ifdef OPENACC_ENABLED
//    #pragma acc enter data copyin(U[0:numPoints])
//#endif

    for (int j=0;j<numPoints;j++){
        for (int i=0;i<numWavefunctions;i++){
            printf("% 5f\t",V[i*numPoints+j]);
        }
        printf("\n");
    }
//    copyMatrixToDevice(V,numPoints,numWavefunctions);

#ifdef OPENACC_ENABLED
    #pragma acc kernels present(U, V, W), copy(r_local[0:targetWavefunction]) //update self(V[targetWavefunction])
    {
#endif



#ifdef OPENACC_ENABLED
    #pragma acc loop independent
#endif
    for (int i=0;i<targetWavefunction;i++){
        r_local[i]=0.0;
//        r_global[i]=0.0;
    }

//    MPI_Barrier(MPI_COMM_WORLD);
#ifdef OPENACC_ENABLED
    #pragma acc loop independent
#endif
    for (int i=0;i<targetWavefunction;i++){
        double local_sum=0.0;
//        printf("target, i = %i, %i\n",targetWavefunction,i);
//        r_local[i] = local_dot_product( V[i],V[targetWavefunction],W,numPoints );
#ifdef OPENACC_ENABLED
    #pragma acc loop independent //reduction(+:local_sum)
#endif
        for (int j=0;j<numPoints;j++){
            local_sum += V[i*numPoints+j]*U[j]*W[j];
        }
        r_local[i]=local_sum;

//        printf("Orthogonalizing target %i against wavefunction %i. Dot product = %f\n", targetWavefunction, i, r_local[i]);
//        printf("Before reduction, rank %i r_local[%i] = %f\n", rank, i, r_local[i]);
    }
#ifdef OPENACC_ENABLED
    } //end ACC kernels
#endif


    printf("Made it here1.\n");

    // Step 2:  Global reduction with MPI
    MPI_Allreduce(r_local, r_global, targetWavefunction, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (int i=0;i<targetWavefunction;i++){
//        printf("After reduction, rank %i r_global[%i] = %f\n", rank, i, r_global[i]);
    }
//    usleep(1500);

    // Step 4:  Subtract Projections
//    MPI_Barrier(MPI_COMM_WORLD);
    double local_norm_squared=0.0;
#ifdef OPENACC_ENABLED
//    int streamID = rand() % 4;
    #pragma acc kernels present(V, W, U) copyin(r_global[0:targetWavefunction]) copy(local_norm_squared) //create(local_norm_squared)
    {
#endif
    for (int i=0;i<targetWavefunction;i++){
//        printf("Subtracting projection of %i from %i\n", i, targetWavefunction);
//        subtract_projection( U, V[i], r_global[i], numPoints );

#ifdef OPENACC_ENABLED
        #pragma acc loop independent
#endif
        for (int j=0;j<numPoints;j++){
            U[j] -= V[i*numPoints+j]*r_global[i];
        }
    }

    // Step 5: Normalize
//    MPI_Barrier(MPI_COMM_WORLD);
//    double local_norm_squared = local_dot_product(U,U,W,numPoints);

//#ifdef OPENACC_ENABLED
//    #pragma acc wait
//#endif

#ifdef OPENACC_ENABLED
    #pragma acc loop independent
#endif
    for (int j=0;j<numPoints;j++){
        local_norm_squared += U[j]*U[j]*W[j];
    }
#ifdef OPENACC_ENABLED
    } //end ACC kernels
#endif
    printf("Made it here2.\n");
    double global_norm_squared;
    MPI_Allreduce(&local_norm_squared, &global_norm_squared, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//    printf("Global norm squared = %f\n", sqrt(global_norm_squared));
//    normalize(U, sqrt(global_norm_squared), numPoints);

//    copyMatrixToDevice(V,numPoints,numWavefunctions);
#ifdef OPENACC_ENABLED
//    int streamID = rand() % 4;
    #pragma acc kernels present(V, W, U) copyin(global_norm_squared) //create(local_norm_squared)
    {
#endif
#ifdef OPENACC_ENABLED
    #pragma acc loop independent
#endif
    for (int j=0;j<numPoints;j++){
        U[j] /= sqrt(global_norm_squared);
        V[targetWavefunction*numPoints+j]=U[j];
    }
#ifdef OPENACC_ENABLED
    } //end ACC kernels
#endif
    printf("Made it here3.\n");

//#ifdef OPENACC_ENABLED
////#pragma acc data copyin(W) copy(V) create(r_local,r_global)
//    #pragma acc exit data copyout(V[0:numWavefunctions][0:numPoints])
//#endif

//#ifdef OPENACC_ENABLED
//    #pragma acc exit data delete(U[0:numPoints])
//#endif


    // Step 6:  Clean up
//    MPI_Barrier(MPI_COMM_WORLD);
    free(r_global);
    free(r_local);
    printf("Freed r and returning.\n");
//    fflush(stdout);
//    MPI_Barrier(MPI_COMM_WORLD);

//#ifdef OPENACC_ENABLED
//    } //end ACC kernels
//#endif

//#ifdef OPENACC_ENABLED
//    #pragma acc wait
//    } // end ACC DATA REGION
//#endif

    return;
}


double local_dot_product(double *A, double *B, double *W, int N){

    double local_sum=0.0;
    for (int j=0;j<N;j++){
        local_sum += A[j]*B[j]*W[j];
    }
    return local_sum;
}

void subtract_projection(double *A, double *B, double r, int N){
    for (int j=0;j<N;j++){
        A[j] -= B[j]*r;
    }
    return;
}

void normalize(double *A, double norm, int N){
    for (int j=0;j<N;j++){
        A[j] /= norm;
    }
}







