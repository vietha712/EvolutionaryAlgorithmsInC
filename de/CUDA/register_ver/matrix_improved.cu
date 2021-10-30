#include "matrix_improved.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include "Utilities.cuh"

/* Declaration */

__host__ __device__ static void LU1D(Matrix1DT *matrix, int n);

/* Exposed functions */
__host__ __device__ void multiplyMatrices1D(Matrix1DT* firstMatrix,
                                            Matrix1DT* secondMatrix,
                                            float      outputArray[],
                                            Matrix1DT* outputMatrix)
{
    assert(firstMatrix->cols == secondMatrix->rows);
    if (0 == outputMatrix->isInit)
    {
        allocateMatrix1D(outputMatrix, outputArray, firstMatrix->rows, secondMatrix->cols);
    }

    zerosMatrix1D(outputMatrix);

    for(int i = 0; i < firstMatrix->rows; i++)
    {
        for(int j = 0; j < secondMatrix->cols; j++)
        {
            float sum = 0;
            for(int k = 0; k < firstMatrix->cols; k++)
                sum = sum + firstMatrix->pMatrix[i * firstMatrix->cols + k] * secondMatrix->pMatrix[k * secondMatrix->cols + j];
            outputMatrix->pMatrix[i * secondMatrix->cols + j] = sum;
        }
    }
}

__host__ __device__ void multiplyScalarMatrix1D(float scalar, Matrix1DT *matrix, float outputArray[], Matrix1DT *outputMatrix)
{

    if (0 == outputMatrix->isInit)
    {
        allocateMatrix1D(outputMatrix, outputArray, matrix->rows, matrix->cols);
    }

    for (int i = 0; i < outputMatrix->rows; i++)
        for (int j = 0; j < outputMatrix->cols; j++)
            outputMatrix->pMatrix[i * outputMatrix->cols + j] = scalar * matrix->pMatrix[i * matrix->cols + j];
}

__host__ __device__ void printMatrix1D(Matrix1DT* matrix)
{
    unsigned int i, j;
    
    printf("Matrix value:\n");
    for(i = 0; i < matrix->rows; ++i)
    {
        for(j = 0; j < matrix->cols; ++j)
            printf("%.12f ", matrix->pMatrix[i * matrix->cols + j]);
        printf("\n");
    }
}

__host__ __device__ static void LU1D(Matrix1DT *matrix, int n)
{
	int i,j,k;
	float x;

    for(k = 0; k <= n - 1; k++)
    {
	  for(j = k + 1; j <= n; j++)
	  {
	    x = matrix->pMatrix[j * matrix->cols + k] / matrix->pMatrix[k * matrix->cols + k];
	    for(i = k; i <= n; i++)
        {  
	       matrix->pMatrix[j * matrix->cols + i] = matrix->pMatrix[j * matrix->cols + i] - x * matrix->pMatrix[k * matrix->cols + i];
        }
	    matrix->pMatrix[j * matrix->cols + k] = x;
	  }
    } 
}

__host__ __device__ void LU_getInverseMatrix1D(Matrix1DT *inputMat, float outputArray[], Matrix1DT *outInvMat)
{
    Matrix1DT localLU, s;
    int i, j;
    float x;

    assert(inputMat->rows == inputMat->cols);
    if (0 == outInvMat->isInit)
    {
        allocateMatrix1D(outInvMat, outputArray, inputMat->rows, inputMat->cols);
    }

    int localDim = inputMat->rows - 1;

    float localLU_array[MAX_ROWS*MAX_COLS];
    allocateMatrix1D(&localLU, localLU_array, inputMat->rows, inputMat->cols); 
    //localLU.rows = inputMat->rows; localLU.cols = inputMat->cols; localLU.isInit = 1;
    float s_array[MAX_ROWS*MAX_COLS];
    allocateMatrix1D(&s, s_array, inputMat->rows, inputMat->cols);
    //s.rows = inputMat->rows; s.cols = inputMat->cols; s.isInit = 1;

    float d[MAX_ROWS], y[MAX_ROWS];

    copyMatrix1D(inputMat, &localLU);

    // Perform decomposition
    LU1D(&localLU, localDim);

    zerosMatrix1D(&s);

    for (int init = 0; init < localDim; init++)
    {
        y[init] = 0.0;
    }

    for(int m = 0; m <= localDim; m++)
    { 
        for (int init = 0; init <= localDim; init++)
        {
            d[init] = 0.0;
        }
	    d[m] = 1.0;

	    for(i = 0; i <= localDim; i++)
        {
            x = 0.0; 
	        for(j = 0; j <= i - 1; j++)
            {
                x = x + localLU.pMatrix[i * localLU.cols + j] * y[j];
            }
 	        y[i] = (d[i]-x);
	    }

	    for(i = localDim; i >= 0; i--)
        {
            x = 0.0; 
	        for(j = i+1; j <= localDim; j++)
            {
                x = x + localLU.pMatrix[i * localLU.cols + j] * s.pMatrix[j * s.cols + m];
            }
 	        s.pMatrix[i * s.cols + m] = (y[i]-x)/localLU.pMatrix[i * localLU.cols + i];
	    }

	}

    /* Copy results */
    copyMatrix1D(&s, outInvMat);

    //deallocateMatrix1D(&s);
    //deallocateMatrix1D(&localLU);
    //free(d);
    //free(y);
}

__host__ __device__ void allocateMatrix1D(Matrix1DT* matrix, float arrayForMat[], int rows, int cols)
{
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->pMatrix = &arrayForMat[0];
    matrix->isInit = 1;
}

__host__ __device__ void initMatrix(Matrix1DT* matrix)
{
    matrix->rows = 0.0;
    matrix->cols = 0.0;
    matrix->isInit = 0;
}

__host__ __device__ void deallocateMatrix1D(Matrix1DT* matrix)
{
    cudaFree(&matrix->pMatrix[0]);
    matrix->isInit = 0;
    matrix->rows = 0.0;
    matrix->cols = 0.0;
}

__host__ __device__ void zerosMatrix1D(Matrix1DT* matrix)
{
    for (int i = 0; i < matrix->rows; i++)
        for (int j = 0; j < matrix->cols; j++)
            matrix->pMatrix[i * matrix->cols + j] = 0.0;
}

__host__ __device__ void copyMatrix1D(Matrix1DT* srcMat, Matrix1DT* desMat)
{
    for (int i = 0; i < srcMat->rows; i++)
        for (int j = 0; j < srcMat->cols; j++)
            desMat->pMatrix[i * srcMat->cols + j] = srcMat->pMatrix[i * srcMat->cols + j];
}