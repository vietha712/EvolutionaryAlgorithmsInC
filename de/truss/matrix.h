#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

typedef struct MatrixTag
{
    int rows;
    int cols;
    double** pMatrix;
} MatrixT;

void allocateMatrix(MatrixT* matrix, int rows, int cols);

void deallocateMatrix(MatrixT* matrix);

void zerosMatrix(MatrixT* matrix);

void multiplyScalarMatrix(double scalar, MatrixT *matrix, MatrixT *outputMatrix);

/*Pass*/
void multiplyMatrices(MatrixT* firstMatrix,
                      MatrixT* secondMatrix,
                      MatrixT* outputMatrix);

/*Pass*/
void addMatrices(MatrixT* firstMatrix,
                 MatrixT* secondMatrix,
                 MatrixT* outputMatrix);       

int inverse(double **A, double **inverse, int sizeOfMatrixA);

void printMatrix(MatrixT* matrix);

void copyMatrix(MatrixT* srcMat, MatrixT* desMat);

void LUdecomposition(double **a, double **l, double **u, int n);

/* Pass */
void LU_getInverseMatrix(MatrixT *inputMat, MatrixT *outInvMat);

#endif