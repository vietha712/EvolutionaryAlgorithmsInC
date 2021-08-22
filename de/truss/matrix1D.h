#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

typedef struct Matrix1DTag
{
    int rows;
    int cols;
    int isInit;
    double* pMatrix;
} Matrix1DT;

void initMatrix(Matrix1DT* matrix);

void allocateMatrix1D(Matrix1DT* matrix, int rows, int cols);

void deallocateMatrix1D(Matrix1DT* matrix);

void zerosMatrix1D(Matrix1DT* matrix);

void multiplyScalarMatrix1D(double scalar, Matrix1DT *matrix, Matrix1DT *outputMatrix);

/*Pass*/
void multiplyMatrices1D(Matrix1DT* firstMatrix,
                      Matrix1DT* secondMatrix,
                      Matrix1DT* outputMatrix);

void printMatrix1D(Matrix1DT* matrix);

void copyMatrix1D(Matrix1DT* srcMat, Matrix1DT* desMat);

/* Pass */
void LU_getInverseMatrix1D(Matrix1DT *inputMat, Matrix1DT *outInvMat);

double findMaxMember(Matrix1DT *inputMatrix);

#endif