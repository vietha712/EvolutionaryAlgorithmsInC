#include "matrix.h"

/* Declaration */
static void getCoFactor(double** matrix, 
                 double** tempMatrix, 
                 int matrixRows,
                 int matrixCols,
                 int dimension);

static double det(int matrixDimension, double** matrix);

static void adjoint(double **A, double **adj, int sizeOfMat);

static void LU(MatrixT *matrix, int n);

/* Static functions */
static void getCoFactor(double** matrix, 
                 double** tempMatrix, 
                 int matrixRows,
                 int matrixCols,
                 int dimension)
{
    int i = 0, j = 0;
    int indexRow, indexCol;
 
    // Looping for each element of the matrix
    for(indexRow = 0; indexRow < dimension; ++indexRow)
    {
        for (indexCol = 0; indexCol < dimension; ++indexCol) 
        {
            //  Copying into temporary matrix only those
            //  element which are not in given row and
            //  column
            if (indexRow != matrixRows && indexCol != matrixCols) 
            {
                tempMatrix[i][j++] = matrix[indexRow][indexCol];
 
                // Row is filled, so increase row index and
                // reset col index
                if (j == dimension - 1) 
                {
                    j = 0;
                    i++;
                }
            }
        }
    }
}

static double det(int matrixDimension, double** matrix)
{
    double determinant = 0;
    // temporary matrix to store cofactors
    double **tempMatrix;
    // multiplier sign
    int sign = 1;
    int i;

    // early return to reduce the complexity
    if(1 == matrixDimension)
    {
        return matrix[0][0];
    }

    tempMatrix = (double **)malloc(matrixDimension * sizeof(double *));
    for(int i = 0; i < matrixDimension; ++i)
    {
        tempMatrix[i] = (double *)malloc(matrixDimension*sizeof(double));
    }

    for(i = 0; i < matrixDimension; ++i)
    {
        getCoFactor(matrix, tempMatrix, 0, i, matrixDimension);

        determinant += (sign * matrix[0][i] * det(matrixDimension - 1, tempMatrix));

        sign = -sign;
    }

    for (int i=0; i < matrixDimension; i++)
    {
       free(tempMatrix[i]);
    }
    free(tempMatrix);

    return determinant;
}

static void adjoint(double **A, double **adj, int sizeOfMat)
{
    if (sizeOfMat== 1)
    {
        adj[0][0] = 1;
        return;
    }

    double **tempMatrix;
    tempMatrix = (double **)malloc(sizeOfMat * sizeof(double *));
    for(int i = 0; i < sizeOfMat; ++i)
    {
        tempMatrix[i] = (double *)malloc(sizeOfMat*sizeof(double));
    }
  
    // temp is used to store cofactors of A[][]
    int sign = 1;
  
    for (int i=0; i<sizeOfMat; i++)
    {
        for (int j=0; j<sizeOfMat; j++)
        {
            // Get cofactor of A[i][j]
            getCoFactor(A, tempMatrix, i, j, sizeOfMat);
  
            // sign of adj[j][i] positive if sum of row
            // and column indexes is even.
            sign = ((i+j)%2==0)? 1: -1;
  
            // Interchanging rows and columns to get the
            // transpose of the cofactor matrix
            adj[j][i] = (sign)*(det(sizeOfMat-1, tempMatrix));
        }
    }

    for (int i=0; i < sizeOfMat; i++)
   {
      free(tempMatrix[i]);
   }
         free(tempMatrix);
}

/* Exposed functions */

void multiplyMatrices(MatrixT* firstMatrix,
                      MatrixT* secondMatrix,
                      MatrixT* outputMatrix)
{
    assert(firstMatrix->cols == secondMatrix->rows);
    allocateMatrix(outputMatrix, firstMatrix->rows, secondMatrix->cols);
    zerosMatrix(outputMatrix);

    for(int i = 0; i < firstMatrix->rows; ++i)
    {
        for(int j = 0; j < secondMatrix->cols; ++j)
        {
            for(int k = 0; k < firstMatrix->cols; ++k)
            {
                outputMatrix->pMatrix[i][j] += (firstMatrix->pMatrix[i][k] * secondMatrix->pMatrix[k][j]);
            }
        }
    }
}

void addMatrices(MatrixT* firstMatrix,
                 MatrixT* secondMatrix,
                 MatrixT* outputMatrix)
{
    assert(firstMatrix->rows == secondMatrix->rows);
    assert(firstMatrix->cols == secondMatrix->cols);
    allocateMatrix(outputMatrix, firstMatrix->rows, firstMatrix->cols);

    int i,j;
    for(i = 0; i < outputMatrix->rows; ++i)
    {
        for(j = 0; j < outputMatrix->cols; ++j)
        {
            outputMatrix->pMatrix[i][j] = firstMatrix->pMatrix[i][j] + secondMatrix->pMatrix[i][j];
        }
    }
}

void multiplyScalarMatrix(double scalar, MatrixT *matrix, MatrixT *outputMatrix)
{
    allocateMatrix(outputMatrix, matrix->rows, matrix->cols);

    for (int i = 0; i < outputMatrix->rows; i++)
        for (int j = 0; j < outputMatrix->cols; j++)
            outputMatrix->pMatrix[i][j] = scalar * matrix->pMatrix[i][j];
}

int inverse(double **A, double **inverse, int sizeOfMatrixA)
{
    // Find determinant of A[][]
    double determinant = det(sizeOfMatrixA, A);
    if (determinant == 0)
    {
        printf( "Singular matrix, can't find its inverse\n");
        return 0;
    }
  printf( "determinant: %.3f\n", determinant);
    // Find adjoint
    double **pAdj;
    pAdj = (double **)malloc(sizeOfMatrixA * sizeof(double *));
    for(int i = 0; i < sizeOfMatrixA; ++i)
    {
        pAdj[i] = (double *)malloc(sizeOfMatrixA*sizeof(double));
    }

    adjoint(A, pAdj, sizeOfMatrixA);
  
    // Find Inverse using formula "inverse(A) = adj(A)/det(A)"
    for (int i = 0; i < sizeOfMatrixA; i++)
        for (int j = 0; j < sizeOfMatrixA; j++)
            inverse[i][j] = pAdj[i][j]/determinant;

    for (int i=0; i < sizeOfMatrixA; i++)
   {
      free(pAdj[i]);
   }
     free(pAdj);
  
    return 1;
}

void printMatrix(MatrixT* matrix)
{
    unsigned int i, j;
    
    printf("Matrix value:\n");
    for(i = 0; i < matrix->rows; ++i)
    {
        for(j = 0; j < matrix->cols; ++j)
            printf("matrix[%d][%d] = %.15f\n", i, j, matrix->pMatrix[i][j]);
    }
}

static void LU(MatrixT *matrix, int n)
{
	int i,j,k;
	double x;

    for(k = 0; k <= n - 1; k++)
    {
	  for(j = k + 1; j <= n; j++)
	  {
	    x = matrix->pMatrix[j][k] / matrix->pMatrix[k][k];
	    for(i = k; i <= n; i++)
        {  
	       matrix->pMatrix[j][i] = matrix->pMatrix[j][i] - x * matrix->pMatrix[k][i];
        }
	    matrix->pMatrix[j][k] = x;
	  }
    } 
}

void LUdecomposition(double **a, double **l, double **u, int n) 
{
   int i = 0, j = 0, k = 0;
   for (i = 0; i < n; i++) 
   {
        for (j = 0; j < n; j++) {
         if (j < i)
         l[j][i] = 0;
         else {
            l[j][i] = a[j][i];
            for (k = 0; k < i; k++) {
               l[j][i] = l[j][i] - l[j][k] * u[k][i];
            }
         }
      }
      for (j = 0; j < n; j++) {
         if (j < i)
         u[i][j] = 0;
         else if (j == i)
         u[i][j] = 1;
         else {
            u[i][j] = a[i][j] / l[i][i];
            for (k = 0; k < i; k++) {
               u[i][j] = u[i][j] - ((l[i][k] * u[k][j]) / l[i][i]);
            }
         }
      }
   }
}

void LU_getInverseMatrix(MatrixT *inputMat, MatrixT *outInvMat)
{
    double *d, *y;
    MatrixT localLU, s;
    int i, j;
    double x;

    assert(inputMat->rows == inputMat->cols);
    allocateMatrix(outInvMat, inputMat->rows, inputMat->cols);
    int localDim = inputMat->rows - 1;

    allocateMatrix(&localLU, inputMat->rows, inputMat->cols);
    allocateMatrix(&s, inputMat->rows, inputMat->cols);

    d = (double *)malloc(inputMat->rows * sizeof(double *));
    y = (double *)malloc(inputMat->rows * sizeof(double *));

    copyMatrix(inputMat, &localLU);

    // Perform decomposition
    LU(&localLU, localDim);

    zerosMatrix(&s);

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
                x = x + localLU.pMatrix[i][j] * y[j];
            }
 	        y[i] = (d[i]-x);
	    }

	    for(i = localDim; i >= 0; i--)
        {
            x = 0.0; 
	        for(j = i+1; j <= localDim; j++)
            {
                x = x + localLU.pMatrix[i][j] * s.pMatrix[j][m];
            }
 	        s.pMatrix[i][m] = (y[i]-x)/localLU.pMatrix[i][i];
	    }

	}

    /* Copy results */
    copyMatrix(&s, outInvMat);

    deallocateMatrix(&s);
    deallocateMatrix(&localLU);
    free(d);
    free(y);
}

void allocateMatrix(MatrixT* matrix, int rows, int cols)
{
    matrix->rows = rows;
    matrix->cols = cols;

    matrix->pMatrix = (double **)malloc(matrix->rows * sizeof(double *));
    for (int i = 0; i < matrix->rows; i++)
        matrix->pMatrix[i] = (double *)malloc(matrix->cols*sizeof(double));
}

void deallocateMatrix(MatrixT* matrix)
{
    for (int i = 0; i < matrix->rows; i++)
        free(matrix->pMatrix[i]);
    free(matrix->pMatrix);
}

void zerosMatrix(MatrixT* matrix)
{
    for (int i = 0; i < matrix->rows; i++)
        for (int j = 0; j < matrix->cols; j++)
            matrix->pMatrix[i][j] = 0.0;
}

void copyMatrix(MatrixT* srcMat, MatrixT* desMat)
{
    for (int i = 0; i < srcMat->rows; i++)
        for (int j = 0; j < srcMat->cols; j++)
            desMat->pMatrix[i][j] = srcMat->pMatrix[i][j];
}