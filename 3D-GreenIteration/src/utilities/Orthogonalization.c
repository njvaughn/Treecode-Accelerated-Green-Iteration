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


    // Step 1:  Compute all local dot products on GPU.  If the target is 6, there need to orthogonalize against 0-5.  Target 6 is readlly the 7th wavefunction.
    double *r_local = malloc((targetWavefunction) * sizeof(double));  // r[j] stores the inner product between the target and jth wavefunctions.
    double *r_global = malloc((targetWavefunction) * sizeof(double));  // r[j] stores the inner product between the target and jth wavefunctions.
    double norm=0.0;


#ifdef OPENACC_ENABLED
    #pragma acc kernels present(U, V, W), copy(r_local[0:targetWavefunction]) //update self(V[targetWavefunction])
    {
#endif



#ifdef OPENACC_ENABLED
    #pragma acc loop independent
#endif
    for (int i=0;i<targetWavefunction;i++){
        r_local[i]=0.0;
    }

#ifdef OPENACC_ENABLED
    #pragma acc loop independent
#endif
    for (int i=0;i<targetWavefunction;i++){
        double local_sum=0.0;
#ifdef OPENACC_ENABLED
    #pragma acc loop independent //reduction(+:local_sum)
#endif
        for (int j=0;j<numPoints;j++){
            local_sum += V[i*numPoints+j]*U[j]*W[j];
        }
        r_local[i]=local_sum;

    }
#ifdef OPENACC_ENABLED
    } //end ACC kernels
#endif


    // Step 2:  Global reduction with MPI
    MPI_Allreduce(r_local, r_global, targetWavefunction, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (int i=0;i<targetWavefunction;i++){
    }

    // Step 4:  Subtract Projections
    double local_norm_squared=0.0;
#ifdef OPENACC_ENABLED
    #pragma acc kernels present(V, W, U) copyin(r_global[0:targetWavefunction]) copy(local_norm_squared) //create(local_norm_squared)
    {
#endif
    for (int i=0;i<targetWavefunction;i++){

#ifdef OPENACC_ENABLED
        #pragma acc loop independent
#endif
        for (int j=0;j<numPoints;j++){
            U[j] -= V[i*numPoints+j]*r_global[i];
        }
    }

    // Step 5: Normalize
#ifdef OPENACC_ENABLED
    #pragma acc loop independent
#endif
    for (int j=0;j<numPoints;j++){
        local_norm_squared += U[j]*U[j]*W[j];
    }
#ifdef OPENACC_ENABLED
    } //end ACC kernels
#endif

    double global_norm_squared;
    MPI_Allreduce(&local_norm_squared, &global_norm_squared, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    norm=sqrt(global_norm_squared);
#ifdef OPENACC_ENABLED
    #pragma acc kernels present(V, W, U) copyin(global_norm_squared) //create(local_norm_squared)
    {
#endif
#ifdef OPENACC_ENABLED
    #pragma acc loop independent
#endif
    for (int j=0;j<numPoints;j++){
        U[j] /= norm;
        V[targetWavefunction*numPoints+j]=U[j];
    }
#ifdef OPENACC_ENABLED
    } //end ACC kernels
#endif



    // Step 6:  Clean up
    free(r_global);
    free(r_local);


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







