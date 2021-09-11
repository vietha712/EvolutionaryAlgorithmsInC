#include "matrix1d.h"

/* Declaration */

static void LU1D(Matrix1DT *matrix, int n);

/* Exposed functions */
void multiplyMatrices1D(Matrix1DT* firstMatrix,
                         Matrix1DT* secondMatrix,
                         Matrix1DT* outputMatrix)
{
    assert(firstMatrix->cols == secondMatrix->rows);
    if (0 == outputMatrix->isInit)
    {
        allocateMatrix1D(outputMatrix, firstMatrix->rows, secondMatrix->cols);
    }

    zerosMatrix1D(outputMatrix);

    for(int i = 0; i < firstMatrix->rows; i++)
    {
        for(int j = 0; j < secondMatrix->cols; j++)
        {
            double sum = 0;
            for(int k = 0; k < firstMatrix->cols; k++)
                sum = sum + firstMatrix->pMatrix[i * firstMatrix->cols + k] * secondMatrix->pMatrix[k * secondMatrix->cols + j];
            outputMatrix->pMatrix[i * secondMatrix->cols + j] = sum;
        }
    }
}

void printMatrix1D(Matrix1DT* matrix)
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

void multiplyScalarMatrix1D(double scalar, Matrix1DT *matrix, Matrix1DT *outputMatrix)
{
    allocateMatrix1D(outputMatrix, matrix->rows, matrix->cols);

    for (int i = 0; i < outputMatrix->rows; i++)
        for (int j = 0; j < outputMatrix->cols; j++)
            outputMatrix->pMatrix[i * outputMatrix->cols + j] = scalar * matrix->pMatrix[i * matrix->cols + j];
}

static void LU1D(Matrix1DT *matrix, int n)
{
	int i,j,k;
	double x;

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

void LU_getInverseMatrix1D(Matrix1DT *inputMat, Matrix1DT *outInvMat)
{
    double *d, *y;
    Matrix1DT localLU, s;
    int i, j;
    double x;

    assert(inputMat->rows == inputMat->cols);
    if (0 == outInvMat->isInit)
    {
        allocateMatrix1D(outInvMat, inputMat->rows, inputMat->cols);
    }

    int localDim = inputMat->rows - 1;

    allocateMatrix1D(&localLU, inputMat->rows, inputMat->cols);
    allocateMatrix1D(&s, inputMat->rows, inputMat->cols);

    d = (double *)malloc(inputMat->rows * sizeof(double *));
    y = (double *)malloc(inputMat->rows * sizeof(double *));

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

    deallocateMatrix1D(&s);
    deallocateMatrix1D(&localLU);
    free(d);
    free(y);
}

void allocateMatrix1D(Matrix1DT* matrix, int rows, int cols)
{
    matrix->rows = rows;
    matrix->cols = cols;

    matrix->pMatrix = (double *)malloc(matrix->rows * matrix->cols * sizeof(double *));

    matrix->isInit = 1;
}

void initMatrix(Matrix1DT* matrix)
{
    matrix->rows = 0.0;
    matrix->cols = 0.0;
    matrix->isInit = 0;
}

void deallocateMatrix1D(Matrix1DT* matrix)
{
    free(&matrix->pMatrix[0]);
    matrix->isInit = 0;
    matrix->rows = 0.0;
    matrix->cols = 0.0;
}

void zerosMatrix1D(Matrix1DT* matrix)
{
    for (int i = 0; i < matrix->rows; i++)
        for (int j = 0; j < matrix->cols; j++)
            matrix->pMatrix[i * matrix->cols + j] = 0.0;
}

void copyMatrix1D(Matrix1DT* srcMat, Matrix1DT* desMat)
{
    for (int i = 0; i < srcMat->rows; i++)
        for (int j = 0; j < srcMat->cols; j++)
            desMat->pMatrix[i * srcMat->cols + j] = srcMat->pMatrix[i * srcMat->cols + j];
}