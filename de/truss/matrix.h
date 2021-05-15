#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

/*Pass*/
void multiplyMatrices(double **firstMatrix,
                      int firstMatrixRows,
                      int firstMatrixCols,
                      double **secondMatrix,
                      int secondMatrixRows,
                      int secondMatrixCols,
                      double **outputMatrix);
/*Pass*/
void addMatrices(double **firstMatrix,
                 int firstMatrixRows,
                 int firstMatrixCols,
                 double **secondMatrix,
                 int secondMatrixRows,
                 int secondMatrixCols,
                 double **outputMatrix, 
                 int outputMatrixRows,
                 int outputMatrixCols);          

int inverse(double **A, double **inverse, int sizeOfMatrixA);

void printMatrix(double **matrix, int rows, int cols);

void cleanMatrix(double **matrix, int rows, int cols);

void LUdecomposition(double **a, double **l, double **u, int n);

/* Pass */
void LU_getInverseMatrix(double **pLU, double **pInvMat, int dimOfMat);

#endif