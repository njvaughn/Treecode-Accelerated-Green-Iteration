#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <float.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
//#include <zoltan.h>

const unsigned mrand = 1664525u;
const unsigned crand = 1013904223u;

//#include "zoltan_fns.h"
#include "ZoltanRCB.h"

/* Test the Zoltan RCB routines */

int main(int argc, char **argv)
{
    int rank, numProcs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

//    /* Zoltan variables */
//    int rc;
//    float ver;
//    struct Zoltan_Struct *zz;
//    int changes, numGidEntries, numLidEntries, numImport, numExport;
//    ZOLTAN_ID_PTR importGlobalGids, importLocalGids, exportGlobalGids, exportLocalGids;
//    int *importProcs, *importToPart, *exportProcs, *exportToPart;
//    int *parts;
//    MESH_DATA myCells;



    printf("Rank %i checking in.\n", rank);
    /* Zoltan initialization */
//    if (Zoltan_Initialize(argc, argv, &ver) != ZOLTAN_OK) {
//        if (rank == 0) printf("Zoltan failed to initialize. Exiting.\n");
//        MPI_Finalize();
//        exit(0);
//    }
//    MPI_Barrier(MPI_COMM_WORLD);
//    if (rank==0) printf("\nZoltan initialized.\n\n");


    /* Set up a variable number of cells for each rank */

    double * cellsX;
    double * cellsY;
    double * cellsZ;
    double * cellsDX;
    double * cellsDY;
    double * cellsDZ;

//    double ** newCellsX;
//    double ** newCellsY;
//    double ** newCellsZ;
//    double ** newCellsDX;
//    double ** newCellsDY;
//    double ** newCellsDZ;

    int r=100;
    int numCells=r*(rank+1);
    int initialNumCells=numCells;
    int globalStart=0;
    for (int i=0;i<rank;i++){
        globalStart+= r*(i+1);
    }
    printf("Rank %i starts at %i\n", rank, globalStart);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    cellsX  = malloc(numCells * sizeof(double));
    cellsY  = malloc(numCells * sizeof(double));
    cellsZ  = malloc(numCells * sizeof(double));
    cellsDX = malloc(numCells * sizeof(double));
    cellsDY = malloc(numCells * sizeof(double));
    cellsDZ = malloc(numCells * sizeof(double));
//    myCells.myGlobalIDs = (ZOLTAN_ID_TYPE *)malloc(sizeof(ZOLTAN_ID_TYPE) * numCells);


    time_t t = time(NULL);
    unsigned t_hashed = (unsigned) t;
    t_hashed = mrand * t_hashed + crand;
    srand(t_hashed ^ rank);
    if (rank==0) printf("Allocated arrays.\n");
    for (int i = 0; i < numCells; ++i) {
        cellsX[i] = ((double)rand()/(double)(RAND_MAX)) * 2. - 1.;
        cellsY[i] = ((double)rand()/(double)(RAND_MAX)) * 2. - 1.;
        cellsZ[i] = ((double)rand()/(double)(RAND_MAX)) * 2. - 1.;
        cellsDX[i] = 1.0;
        cellsDY[i] = 1.0;
        cellsDZ[i] = 1.0;
    }


    if (rank==0) printf("Initialized cell arrays.\n");

    int newNumCells;

    loadBalanceRCB( &cellsX,&cellsY,&cellsZ,&cellsDX,&cellsDY,&cellsDZ,
//                    newCellsX, newCellsY, newCellsZ, newCellsDX, newCellsDY, newCellsDZ,
                    numCells,globalStart,&newNumCells);


    for (int i=0;i<5;i++){
        if (rank==0) printf("after load balancing, value = %f\n", cellsX[i]);
    }


    if (rank==0) printf("Called load balancer.\n");
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    double xmean=0.0;
    double ymean=0.0;
    double zmean=0.0;

    for (int j=0;j<newNumCells;++j){
        xmean += cellsX[j];
        ymean += cellsY[j];
        zmean += cellsZ[j];
//        printf("j=%i", j);
    }
    xmean /= (double)newNumCells;
    ymean /= (double)newNumCells;
    zmean /= (double)newNumCells;


    MPI_Barrier(MPI_COMM_WORLD);
    printf("Rank %i began with %i cells and now has %i cells, with means (x,y,z)=(%f,%f,%f).\n\n", rank, initialNumCells,newNumCells,xmean,ymean,zmean);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);


    MPI_Finalize();

    return 0;
}
