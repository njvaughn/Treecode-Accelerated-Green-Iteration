#ifndef H_MoveData_H
#define H_MoveData_H


#include "moveData.h"

void copyToDevice(double *X, double *Y, double *Z, double *W, double **Wavefunctions, int numPoints, int numWavefunctions);
void copyFromDevice(double *X, double *Y, double *Z, double *W, double **Wavefunctions, int numPoints, int numWavefunctions);

void copyVectorToDevice(double *U, int numPoints);
void copyVectorFromDevice(double *U, int numPoints);
void removeVectorFromDevice(double *U, int numPoints);

void copyMatrixToDevice(double **Wavefunctions, int numPoints, int numWavefunctions);
void copyMatrixFromDevice(double **Wavefunctions, int numPoints, int numWavefunctions);
void removeMatrixFromDevice(double **Wavefunctions, int numPoints, int numWavefunctions);

#endif /* H_MoveData_H */
