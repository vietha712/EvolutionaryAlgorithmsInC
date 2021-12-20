#ifndef _MATRIX_IMPROVED_H_
#define _MATRIX_IMPROVED_H_

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#define MAX_ROWS 100
#define MAX_COLS 100

typedef struct Matrix1DTag
{
    int rows;
    int cols;
    int isInit;
    float *pMatrix;
} Matrix1DT;

__host__ __device__ void initMatrix(Matrix1DT* matrix);

__host__ __device__ void allocateMatrix1D(Matrix1DT* matrix, float arrayForMat[], int rows, int cols);

__host__ __device__ void deallocateMatrix1D(Matrix1DT* matrix);

__host__ __device__ void zerosMatrix1D(Matrix1DT* matrix);

__host__ __device__ void multiplyScalarMatrix1D(float scalar, Matrix1DT *matrix, float outputArray[], Matrix1DT *outputMatrix);

/*Pass*/
__host__ __device__ void multiplyMatrices1D(Matrix1DT* firstMatrix,
                                            Matrix1DT* secondMatrix,
                                            float      outputArray[],
                                            Matrix1DT* outputMatrix);

__host__ __device__ void printMatrix(Matrix1DT* matrix);

__host__ __device__ void copyMatrix1D(Matrix1DT* srcMat, Matrix1DT* desMat);

/* Pass */
__host__ __device__ void LU_getInverseMatrix1D(Matrix1DT *inputMat, float outputArray[], Matrix1DT *outInvMat);

#endif