#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#define NUM_OF_ELEMENTS 10
#define NUM_OF_NODES 6
#define DOF 2
#define TOTAL_DOF 12 // DOF * NUM_OF_NODES
#define TE_NUMCOLS 4
#define TE_NUMROWs 2

void transposeOfTe(double A[2][4], double[4][2]);

/*
const double A[42] = {1.62, 1.80, 1.99, 2.13, 2.38, 2.62, 2.63, 2.88, 2.93, 3.09, 3.13, 3.38,
                      3.47, 3.55, 3.63, 3.84, 3.87, 3.88, 4.18, 4.22, 4.49, 4.59, 4.80, 4.97,
                      5.12, 5.74, 7.22, 7.97, 11.50, 13.50, 13.90, 14.20, 15.50, 16.00, 16.90,
                      18.80, 19.90, 22.00, 22.90, 26.50, 30.00, 33.50}; //Standard cross-sectional areas for design variable
*/

const double preCpted_A[10] = {30, 1.62, 22.9, 13.5, 1.62, 1.62, 7.97, 26.5, 22, 1.8};

int element[NUM_OF_ELEMENTS][2] = { {3, 5}, {1, 3}, {4, 6}, {2, 4}, {3, 4}, 
                                    {1, 2}, {4, 5}, {3, 6}, {2, 3}, {1, 4} };

int gCoord[2][6] = {{720, 720, 360, 360, 0, 0},
                    {360, 0, 360, 0, 360, 0}};

double Xl[NUM_OF_ELEMENTS] = {1.62,1.62,1.62,1.62,1.62,1.62,1.62,1.62,1.62,1.62},
       Xu[NUM_OF_ELEMENTS] = {33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50};

double K[TOTAL_DOF][TOTAL_DOF] = {0};

double F[TOTAL_DOF] = {0};


inline double getWeight(double A, double matDen, double len)
{
    return (A * matDen * len);
}

void fix(double *X, int len)
{
    for (int i = 0; i < len; i++)
    {
        for (int i = 0; i < 42; i++)
        {
            if (X[i] <= A[i])
            {
                X[i] = A[i];
            }
        }
    }
}

void getTransposeOfTe(double **A, double **B)
{
    int i, j;
    for (i = 0; i < 4; i++)
        for (j = 0; j < 2; j++)
            B[i][j] = A[j][i];
}

void multiplyMatrices(double **firstMatrix,
                    int firstMatrixRows,
                    int firstMatrixCols,
                    double **secondMatrix,
                    int secondMatrixRows,
                    int secondMatrixCols,
                    double **outputMatrix)
{
    assert(firstMatrixRows == secondMatrixCols);

    for(int i = 0; i < firstMatrixRows; ++i)
    {
        for(int j = 0; j < secondMatrixCols; ++j)
        {
            for(int k = 0; k < firstMatrixCols; ++k)
            {
                outputMatrix[i][j] = (firstMatrix[i][k] * secondMatrix[k][j]);
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
    assert(firstMatrixRows == secondMatrixCols);

    for(int i = 0; i < firstMatrixRows; ++i)
    {
        for(int j = 0; j < secondMatrixCols; ++j)
        {
            for(int k = 0; k < firstMatrixCols; ++k)
            {
                outputMatrix[i][j] = (firstMatrix[i][k] / secondMatrix[k][j]);
            }
        }
    }
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

const double epsilon_1 = 1.0;
double epsilon_2 = 20.0;
int E = 10000000;
int P = 100000;
double rho = 2770; // density of material kg/m^3
int D = 10;

double func(double *A)
{
    extern int D;
    double sum;
    double le;
    int x[2], y[2];
    double l_ij, m_ij;
    double **Te, **Te_Transpose;
    double **ke2x2, **ke4x4;
    double **matrix2x2_Precomputed, **output2x2, **output4x2, **output4x4; //line 57 in 10 bars
    int index[4];
    int bcDOF[4] = {9, 10, 11, 12};
    double bcValue[4] = {0};

    Te = (double **)malloc(2 * sizeof(double *));
    Te_Transpose = (double **)malloc(4 * sizeof(double *));
    ke2x2 = (double **)malloc(2 * sizeof(double *));
    matrix2x2_Precomputed = (double **)malloc(2 * sizeof(double *));
    output2x2 = (double **)malloc(2 * sizeof(double *));
    output4x2 = (double **)malloc(4 * sizeof(double *));
    output4x4 = (double **)malloc(4 * sizeof(double *));
    for (int memIndx = 0; memIndx < 2; memIndx++)
    {
        ke2x2[memIndx] = (double *)malloc((2)*sizeof(double));
        matrix2x2_Precomputed[memIndx] = (double *)malloc((2)*sizeof(double));
        output2x2[memIndx] = (double *)malloc((2)*sizeof(double));
        Te[memIndx] = (double *)malloc((4)*sizeof(double));
    }

    ke4x4 = (double **)malloc(4 * sizeof(double *));
    for (int memIndx = 0; memIndx < 4; memIndx++)
    {
        ke4x4[memIndx] = (double *)malloc((4)*sizeof(double));
        output4x2[memIndx] = (double *)malloc((2)*sizeof(double));
        output4x2[memIndx] = (double *)malloc((4)*sizeof(double));
        Te_Transpose[memIndx] = (double *)malloc((2)*sizeof(double));
    }
    matrix2x2_Precomputed[0][0] = 1;
    matrix2x2_Precomputed[0][1] = -1;
    matrix2x2_Precomputed[1][0] = -1;
    matrix2x2_Precomputed[1][1] = 1;
    
    /* Calculate stiffness matrix */
    for (int i = 0; i < NUM_OF_ELEMENTS; i++)
    {
        x[0] = gCoord[0][element[i][0]];
        x[1] = gCoord[0][element[i][1]];
        y[0] = gCoord[1][element[i][0]];
        y[1] = gCoord[1][element[i][1]];

        le = sqrt( (x[2] - x[1])^2 + (y[2] - y[1])^2 ); //

        //Compute direction cosin
        l_ij = (x[2] - x[1])/le;
        m_ij = (y[2] - y[1])/le;

        //Compute transform matrix
        Te[1][1] = l_ij; Te[1][2] = m_ij; Te[1][3] = 0; Te[1][4] = 0;
        Te[2][1] = 0; Te[2][2] = 0; Te[2][3] = l_ij; Te[2][4] = m_ij;

        // Compute stiffness martix of element line 56
        multiplyScalarMatrix((A[i]*E/le), matrix2x2_Precomputed, output2x2, 2, 2);
        getTransposeOfTe(Te, Te_Transpose);
        multiplyMatrices(Te_Transpose, TE_NUMCOLS, TE_NUMROWs,
                         ke2x2, 2, 2, output4x2);
        multiplyMatrices(output4x2, 4, 2,
                         Te, 2, 4, output4x4);

        //Find index assemble
        index[0] = 2*element[i][0] - 1;
        index[1] = 2*element[i][0];
        index[2] = 2*element[i][1] - 1;
        index[3] = 2*element[i][1];

        for (int row_i = 0; row_i < 4; row_i++)
        {
            for (int col_i = 0; col_i < 4; col_i++)
                K[index[row_i]][index[col_i]] =  K[index[row_i]][index[col_i]] + output4x4[row_i][col_i];
        }
    }

    F[4] = F[8] = -P;

    for (int bc_i = 0; bc_i < 4; bc_i++)
    {
        int temp = bcDOF[i];
        for (int zeros_i = 0; zeros_i < TOTAL_DOF, zeros_i++)
            K[temp][zeros_i] = 0;

        K[temp][temp] = 1;
        F[temp] = bcValue[bc_i];
    }

    //TO BE CONT. Calculate U line 78

    return;
}