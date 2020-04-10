#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <string.h>
#include <mpi.h>
#include <zoltan.h>


#include "zoltan_fns.h"


/* Performs Recursive Coordinate Bisection (RCB) on a set of quadrature cells */


void loadBalanceRCB(double **cellsX, double **cellsY, double **cellsZ,
                    double **cellsDX, double **cellsDY, double **cellsDZ,
//                    double **newCellsX, double **newCellsY, double **newCellsZ,
//                    double **newCellsDX, double **newCellsDY, double **newCellsDZ,
                    int numCells, int globalStart, int * newNumCells){

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);


    /* Zoltan variables */
    int rc;
    float ver;
    struct Zoltan_Struct *zz;
    int changes, numGidEntries, numLidEntries, numImport, numExport;
    ZOLTAN_ID_PTR importGlobalGids, importLocalGids, exportGlobalGids, exportLocalGids;
    int *importProcs, *importToPart, *exportProcs, *exportToPart;
    int *parts;
    MESH_DATA myCells;


    /* Zoltan initialization */
    int argc;
    char **argv;
    if (Zoltan_Initialize(argc, argv, &ver) != ZOLTAN_OK) {
        if (rank == 0) printf("Zoltan failed to initialize. Exiting.\n");
        MPI_Finalize();
        exit(0);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0) printf("\nZoltan initialized.\n\n");
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    /* Set up a variable number of cells for each rank */


    printf("Rank %i starts at %i\n", rank, globalStart);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    myCells.numMyPoints = numCells;
    myCells.x  = *cellsX;
    myCells.y  = *cellsY;
    myCells.z  = *cellsZ;
    myCells.dx = *cellsDX;
    myCells.dy = *cellsDY;
    myCells.dz = *cellsDZ;
    myCells.myGlobalIDs = (ZOLTAN_ID_TYPE *)malloc(sizeof(ZOLTAN_ID_TYPE) * numCells);



    for (int i = 0; i < numCells; ++i) {
        myCells.myGlobalIDs[i] = (ZOLTAN_ID_TYPE)(globalStart + i);
    }

    int totalNumberOfCells;
    MPI_Allreduce(&numCells, &totalNumberOfCells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    myCells.numGlobalPoints = totalNumberOfCells;

    if (rank==0) printf("Initialized myCells.\n");

    for (int i=0;i<5;i++){
        printf("rank %i, before load balancing, value = %f\n", rank, myCells.x[i]);
//        printf("rank %i, before load balancing, value = %f\n", rank, *cellsX[i]);
    }




    /* Set up load balancer */

    zz = Zoltan_Create(MPI_COMM_WORLD);

    /* General parameters */

    Zoltan_Set_Param(zz, "DEBUG_LEVEL", "0");
    Zoltan_Set_Param(zz, "LB_METHOD", "RCB");
    Zoltan_Set_Param(zz, "NUM_GID_ENTRIES", "1");
    Zoltan_Set_Param(zz, "NUM_LID_ENTRIES", "1");
    Zoltan_Set_Param(zz, "OBJ_WEIGHT_DIM", "1");
    Zoltan_Set_Param(zz, "RETURN_LISTS", "ALL");
    Zoltan_Set_Param(zz, "AUTO_MIGRATE", "TRUE");

    /* RCB parameters */

    Zoltan_Set_Param(zz, "KEEP_CUTS", "1");
    Zoltan_Set_Param(zz, "RCB_OUTPUT_LEVEL", "0");
    Zoltan_Set_Param(zz, "RCB_RECTILINEAR_BLOCKS", "1");

    /* Query functions, to provide geometry to Zoltan */

    Zoltan_Set_Num_Obj_Fn(zz, ztn_get_number_of_objects, &myCells);
    Zoltan_Set_Obj_List_Fn(zz, ztn_get_object_list, &myCells);
    Zoltan_Set_Num_Geom_Fn(zz, ztn_get_num_geometry, &myCells);
    Zoltan_Set_Geom_Multi_Fn(zz, ztn_get_geometry_list, &myCells);
    Zoltan_Set_Obj_Size_Fn(zz, ztn_obj_size, &myCells);
    Zoltan_Set_Pack_Obj_Fn(zz, ztn_pack, &myCells);
    Zoltan_Set_Unpack_Obj_Fn(zz, ztn_unpack, &myCells);

    rc = Zoltan_LB_Partition(zz, /* input (all remaining fields are output) */
        &changes,        /* 1 if partitioning was changed, 0 otherwise */
        &numGidEntries,  /* Number of integers used for a global ID */
        &numLidEntries,  /* Number of integers used for a local ID */
        &numImport,      /* Number of vertices to be sent to me */
        &importGlobalGids,  /* Global IDs of vertices to be sent to me */
        &importLocalGids,   /* Local IDs of vertices to be sent to me */
        &importProcs,    /* Process rank for source of each incoming vertex */
        &importToPart,   /* New partition for each incoming vertex */
        &numExport,      /* Number of vertices I must send to other processes*/
        &exportGlobalGids,  /* Global IDs of the vertices I must send */
        &exportLocalGids,   /* Local IDs of the vertices I must send */
        &exportProcs,    /* Process to which I send each of the vertices */
        &exportToPart);  /* Partition to which each vertex will belong */


    if (rank==0) printf("Set up load balancer.\n");

     /* Call load balancer */
    int i = 0;
    while (i < myCells.numMyPoints) {
        if ((int)myCells.myGlobalIDs[i] < 0) {
            myCells.x[i] = myCells.x[myCells.numMyPoints-1];
            myCells.y[i] = myCells.y[myCells.numMyPoints-1];
            myCells.z[i] = myCells.z[myCells.numMyPoints-1];
            myCells.dx[i] = myCells.dx[myCells.numMyPoints-1];
            myCells.dy[i] = myCells.dy[myCells.numMyPoints-1];
            myCells.dz[i] = myCells.dz[myCells.numMyPoints-1];
            myCells.myGlobalIDs[i] = myCells.myGlobalIDs[myCells.numMyPoints-1];
            myCells.numMyPoints--;
        } else {
          i++;
        }
    }

    *cellsX=myCells.x;
    *cellsY=myCells.y;
    *cellsZ=myCells.z;
    *cellsDX=myCells.dx;
    *cellsDY=myCells.dy;
    *cellsDZ=myCells.dz;

    for (int i=0;i<5;i++){
        printf("rank %i, after load balancing, value = %f\n", rank, myCells.x[i]);
//        printf("rank %i, after load balancing, value = %f\n", rank, *cellsX[i]);
    }


    newNumCells[0]=myCells.numMyPoints;
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
//    free(myCells.myGlobalIDs);


    if (rc != ZOLTAN_OK) {
        printf("Error! Zoltan has failed. Exiting. \n");
        MPI_Finalize();
        Zoltan_Destroy(&zz);
        exit(0);
    }

    return;
}




