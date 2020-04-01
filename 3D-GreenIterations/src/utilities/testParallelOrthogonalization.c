#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <float.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>
#include<time.h>


/* Orthogonalization routines */


#include "Orthogonalization.h"
#include "moveData.h"

int main(int argc, char **argv)
{
    int rank, numProcs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

#ifdef OPENACC_ENABLED
    #pragma acc set device_num(rank) device_type(acc_device_nvidia)
    #pragma acc init device_type(acc_device_nvidia)
#endif

    int numPoints = atoi(argv[1]);
    int numWavefunctions = atoi(argv[2]);

//    if (rank==0){
//        printf("\n\nNumber of points:          %i\n", numPoints);
//        printf("Number of wavefunctions:   %i\n", numWavefunctions);
//    }
//    fflush(stdout);

//    double *wavefunctions = (double **)malloc(numWavefunctions * sizeof(double *));
//    for (int i=0; i<numWavefunctions; i++){
//         wavefunctions[i] = (double *)malloc(numPoints * sizeof(double));
//    }

    double *wavefunctions = malloc((numPoints*numWavefunctions) * sizeof(double));
//    for (int i=0; i<numWavefunctions; i++){
//         wavefunctions[i] = (double *)malloc(numPoints * sizeof(double));
//    }

    // set wavefunctions to something regular
    srand(rank);
    for (int i=0;i<numWavefunctions;i++){
//        printf("\nrank %i, wavefunction %i\n",rank,i);
        for (int j=0;j<numPoints;j++){
            wavefunctions[i*numPoints + j]=pow(j+1,i+1) - (5-j)*pow(rank,2);

//            wavefunctions[i*numPoints + j]=(double)rand()/RAND_MAX;
//            printf("rank %i, wavefunctions[%i][%i] = %f\n", rank, i, j, wavefunctions[i*numPoints + j]);
        }
    }


    // set weight array to ones
    double *W = malloc((numPoints) * sizeof(double));
    double *U = malloc((numPoints) * sizeof(double));
    for (int i=0;i<numPoints;i++){
        U[i]=1.0;
        W[i]=1.0;
    }




    // test Normalization
    double local_norm_squared=0.0;
    double global_norm_squared=0.0;
    for (int i=0;i<numWavefunctions;i++){
        local_norm_squared=local_dot_product(&wavefunctions[i*numPoints],&wavefunctions[i*numPoints],W,numPoints);
        MPI_Allreduce(&local_norm_squared, &global_norm_squared, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//        printf("\nrank %i, wavefunction %i local_norm_squared = %f, global_norm_squared = %f\n", rank, i, local_norm_squared,global_norm_squared);
        normalize(&wavefunctions[i*numPoints], sqrt(global_norm_squared), numPoints);
        local_norm_squared=local_dot_product(&wavefunctions[i*numPoints],&wavefunctions[i*numPoints],W,numPoints);
        MPI_Allreduce(&local_norm_squared, &global_norm_squared, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//        printf("rank %i, wavefunction %i local_norm_squared = %f, global_norm_squared = %f\n", rank, i, local_norm_squared,global_norm_squared);
        if (fabs(global_norm_squared-1)>1e-16*numPoints){
            printf("ERROR: Initial wavefunction %i is not normalized.\n", i);
        }
    }


    MPI_Barrier(MPI_COMM_WORLD);
    double start=MPI_Wtime();


    copyVectorToDevice(W,numPoints);
    copyVectorToDevice(wavefunctions,numPoints*numWavefunctions);

    for (int targetWavefunction=0; targetWavefunction<numWavefunctions;targetWavefunction++){
        printf("targetWavefunction = %i\n",targetWavefunction);
        for (int j=0; j<numPoints;j++){
//            wavefunctions[targetWavefunction*numPoints + j]+=sin(targetWavefunction*j);
            U[j]=wavefunctions[targetWavefunction*numPoints + j];
        }
//        modifiedGramSchmidt_singleWavefunction(wavefunctions, wavefunctions[targetWavefunction], W, targetWavefunction, numPoints, numWavefunctions);
        copyVectorToDevice(U,numPoints);
        modifiedGramSchmidt_singleWavefunction(wavefunctions, U, W, targetWavefunction, numPoints, numWavefunctions);
        removeVectorFromDevice(U,numPoints);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    copyVectorFromDevice(wavefunctions,numPoints*numWavefunctions);

    double end=MPI_Wtime();


    // Check orthogonality

    // Check normalization
    local_norm_squared=0.0;
    global_norm_squared=0.0;
    for (int i=0;i<numWavefunctions;i++){
        local_norm_squared=local_dot_product(&wavefunctions[i*numPoints],&wavefunctions[i*numPoints],W,numPoints);
        MPI_Allreduce(&local_norm_squared, &global_norm_squared, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (fabs(global_norm_squared-1)>1e-16*numPoints){
            if (rank==0){
                printf("ERROR: Wavefunction %i is not normalized. Norm-1 = %1.3e\n", i, sqrt(global_norm_squared)-1);
            }
        }
//        printf("\nrank %i, wavefunction %i local_norm_squared = %f, global_norm_squared = %f\n", rank, i, local_norm_squared,global_norm_squared);
//        usleep(500);
    }


//    for (int i=0;i<numWavefunctions;i++){
//        printf("\nrank %i, wavefunction %i\n",rank,i);
//        for (int j=0;j<numPoints;j++){
//            printf("rank %i, wavefunctions[%i][%i] = %f\n", rank, i, j, wavefunctions[i*numPoints + j]);
//        }
//    }

//    usleep(1500);
    double overlap=1.0;
    double global_overlap=1.0;
    for (int i=0;i<numWavefunctions;i++){
        for (int j=0; j<i;j++){
//            printf("i,j = %i,%i\n", i, j);
            fflush(stdout);
            overlap = local_dot_product(&wavefunctions[i*numPoints],&wavefunctions[j*numPoints],W,numPoints);
            MPI_Allreduce(&overlap, &global_overlap, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            if (fabs(global_overlap)>1e-15*numPoints){
                if (rank==0){
                    printf("ERROR: Wavefunctions %i and %i not orthogonal. Overlap = %1.3e\n", i,j, global_overlap);
                }
            }
        }
    }

//    for (int j=0;j<numPoints;j++){
//        for (int i=0;i<numWavefunctions;i++){
//            printf("% 5f\t",wavefunctions[i*numPoints + j]);
//        }
//        printf("\n");
//    }
    for (int j=0;j<numWavefunctions;j++){
        printf("% 5f\t",wavefunctions[j*numPoints + 0]);
    }



    if (rank==0){
        printf("\n\n     C time to orthogonalize %i wavefunctions of %i points distributed over %i processors: %f seconds.\n\n\n", numWavefunctions, numPoints*numProcs, numProcs, end-start);

    }

//    printf("Freeing wavefunctions.\n");
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
//    for (int i=0; i<numWavefunctions; i++){
//        free(wavefunctions[i]);
////        printf("Freed wavefunction %i\n", i);
//        fflush(stdout);
//
//    }
    free(wavefunctions);
//    printf("Freed wavefunctions.\n");
    fflush(stdout);
    free(W);
//    printf("Freed W.\n");
    fflush(stdout);
//    free_matrix(wavefunctions);
//    free_vector(W);
    MPI_Barrier(MPI_COMM_WORLD);


    MPI_Finalize();

    return 0;
}
