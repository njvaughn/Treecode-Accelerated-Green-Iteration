#ifndef H_ZoltanRCB_H
#define H_ZoltanRCB_H


void loadBalanceRCB(double **cellsX, double **cellsY, double **cellsZ,
                    double **cellsDX, double **cellsDY, double **cellsDZ,
                    int **coarsePtsPerCell, int **finePtsPerCell,
                    int numCells, int globalStart, int * newNumCells);

#endif /* H_ZoltanRCB_H */

