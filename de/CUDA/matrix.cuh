#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#define MAX_ROWS 20
#define MAX_COLS 20

typedef struct Matrix1DTag
{
    int rows;
    int cols;
    int isInit;
    float pMatrix[MAX_ROWS * MAX_COLS];
} Matrix1DT;

__host__ __device__ void initMatrix(Matrix1DT* matrix);

__host__ __device__ void allocateMatrix1D(Matrix1DT* matrix, int rows, int cols);

__host__ __device__ void deallocateMatrix1D(Matrix1DT* matrix);

__host__ __device__ void zerosMatrix1D(Matrix1DT* matrix);

__host__ __device__ void multiplyScalarMatrix1D(float scalar, Matrix1DT *matrix, Matrix1DT *outputMatrix);

/*Pass*/
__host__ __device__ void multiplyMatrices1D(Matrix1DT* firstMatrix,
                      Matrix1DT* secondMatrix,
                      Matrix1DT* outputMatrix);

__host__ __device__ void printMatrix(Matrix1DT* matrix);

__host__ __device__ void copyMatrix1D(Matrix1DT* srcMat, Matrix1DT* desMat);

/* Pass */
__host__ __device__ void LU_getInverseMatrix1D(Matrix1DT *inputMat, Matrix1DT *outInvMat);

#endif