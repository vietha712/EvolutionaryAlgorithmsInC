#include "matrix.h"

/* Declaration */
static void getCoFactor(double** matrix, 
                 double** tempMatrix, 
                 int matrixRows,
                 int matrixCols,
                 int dimension);

static double det(int matrixDimension, double** matrix);

static void adjoint(double **A, double **adj, int sizeOfMat);

static void LU(double **pMatrix, int n);

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

void multiplyMatrices(double **firstMatrix,
                    int firstMatrixRows,
                    int firstMatrixCols,
                    double **secondMatrix,
                    int secondMatrixRows,
                    int secondMatrixCols,
                    double **outputMatrix)
{
    assert(firstMatrixCols == secondMatrixRows);

    for(int i = 0; i < firstMatrixRows; ++i)
    {
        for(int j = 0; j < secondMatrixCols; ++j)
        {
            for(int k = 0; k < firstMatrixCols; ++k)
            {
                outputMatrix[i][j] += (firstMatrix[i][k] * secondMatrix[k][j]);
            }
        }
    }
}

void divideMatrices(double **firstMatrix,
                    int firstMatrixRows,
                    int firstMatrixCols,
                    double **secondMatrix,
                    int secondMatrixRows,
                    int secondMatrixCols,
                    double **outputMatrix)
{

}

void addMatrices(double **firstMatrix,
         int firstMatrixRows,
         int firstMatrixCols,
         double **secondMatrix,
         int secondMatrixRows,
         int secondMatrixCols,
         double **outputMatrix, 
         int outputMatrixRows,
         int outputMatrixCols)
{
    assert(firstMatrixRows == secondMatrixRows);
    assert(firstMatrixCols == secondMatrixCols);

    int i,j;
    for(i = 0; i < outputMatrixRows; ++i)
    {
        for(j = 0; j < outputMatrixCols; ++j)
        {
            outputMatrix[i][j] = firstMatrix[i][j] + secondMatrix[i][j];
        }
    }
}

void multiplyScalarMatrix(double scalar, double **matrix, double **output, int numRows, int numCols)
{
    for (int i = 0; i < numRows; i++)
        for (int j = 0; j < numCols; j++)
            output[i][j] = scalar * matrix[i][j];
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

void printMatrix(double **matrix, int rows, int cols)
{
    unsigned int i, j;
    
    printf("Matrix value:\n");
    for(i = 0; i < rows; ++i)
    {
        for(j = 0; j < cols; ++j)
            printf("matrix[%d][%d] = %.3f\n", i, j, matrix[i][j]);
    }
}

void cleanMatrix(double **matrix, int rows, int cols)
{
    unsigned int i, j;
    for(i = 0; i < rows; ++i)
    {
        for(j = 0; j < cols; ++j)
            matrix[i][j] = 0.0;
    }
}

static void LU(double **pMatrix, int n)
{
	int i,j,k,m,an,am;
	float x;
    
    for(k=0;k<=n-1;k++)
    {
	  for(j=k+1;j<=n;j++)
	  {
	    x = pMatrix[j][k]/pMatrix[k][k];
	    for(i=k;i<=n;i++)
        {  
	       pMatrix[j][i]=pMatrix[j][i] - x*pMatrix[k][i];
        }
	    pMatrix[j][k]=x;
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


void LU_getInverseMatrix(double **inputMat, double **pInvMat, int dimOfMat)
{
    double *d, *y, **s, **pLocalLU;
    int localDim = dimOfMat - 1;
    int i, j;
    double x;

    d = (double *)malloc(dimOfMat * sizeof(double *));
    y = (double *)malloc(dimOfMat * sizeof(double *));
    s = (double **)malloc(dimOfMat * sizeof(double *));
    pLocalLU = (double **)malloc(dimOfMat * sizeof(double *));
    for(int alloc = 0; alloc < dimOfMat; alloc++)
    {
        s[alloc] = (double *)malloc(dimOfMat * sizeof(double *));
        pLocalLU[alloc] = (double *)malloc(dimOfMat * sizeof(double *));
    }

    for (int copy = 0; copy <= localDim; copy++)
        for (int copy2 = 0; copy2 <= localDim; copy2++)
            pLocalLU[copy][copy2] = inputMat[copy][copy2];

    // Perform decompose
    LU(pLocalLU, localDim);

    for (int init = 0; init < localDim; init++)
    {
        y[init] = 0.0;
        for (int init2 = 0; init2 <= localDim; init2++)
            s[init][init2] = 0.0;
    }

    for(int m = 0; m <= localDim; m++)
    { 
        for (int init = 0; init <= localDim; init++)
        {
            d[init] = 0.0;
        }
	    d[m]=1.0;

	    for(i = 0; i <= localDim; i++)
        {
            x = 0.0; 
	        for(j=0;j<=i-1;j++)
            {
                x = x + pLocalLU[i][j]*y[j];
            }
 	        y[i] = (d[i]-x);
	    }

	    for(i = localDim; i >= 0; i--)
        {
            x=0.0; 
	        for(j = i+1; j <= localDim; j++)
            {
                x = x + pLocalLU[i][j] * s[j][m];
            }
 	        s[i][m]=(y[i]-x)/pLocalLU[i][i];
	    }
	}

    /* Copy results */
    for (int copy = 0; copy <= localDim; copy++)
        for (int copy2 = 0; copy2 <= localDim; copy2++)
            pInvMat[copy][copy2] = s[copy][copy2];

    for (int i=0; i < dimOfMat; i++)
    {
      free(s[i]);
      free(pLocalLU[i]);
    }
    free(d);
    free(y);
    free(s);
    free(pLocalLU);
}