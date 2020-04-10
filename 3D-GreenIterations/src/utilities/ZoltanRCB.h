#ifndef H_ZoltanRCB_H
#define H_ZoltanRCB_H


void loadBalanceRCB(double **cellsX, double **cellsY, double **cellsZ,
                    double **cellsDX, double **cellsDY, double **cellsDZ,
//                    double **newCellsX, double **newCellsY, double **newCellsZ,
//                    double **newCellsDX, double **newCellsDY, double **newCellsDZ,
                    int numCells, int globalStart, int * newNumCells);

#endif /* H_ZoltanRCB_H */

