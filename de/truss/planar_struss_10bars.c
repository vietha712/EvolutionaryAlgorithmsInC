#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "matrix.h"

#define NUM_OF_ELEMENTS 10
#define NUM_OF_NODES 6
#define DOF 2
#define TOTAL_DOF 12 // DOF * NUM_OF_NODES
#define TE_NUMCOLS 4
#define TE_NUMROWs 2

void transposeOfTe(double A[2][4], double[4][2]);


const double standard_A[42] = {1.62, 1.80, 1.99, 2.13, 2.38, 2.62, 2.63, 2.88, 2.93, 3.09, 3.13, 3.38,
                      3.47, 3.55, 3.63, 3.84, 3.87, 3.88, 4.18, 4.22, 4.49, 4.59, 4.80, 4.97,
                      5.12, 5.74, 7.22, 7.97, 11.50, 13.50, 13.90, 14.20, 15.50, 16.00, 16.90,
                      18.80, 19.90, 22.00, 22.90, 26.50, 30.00, 33.50}; //Standard cross-sectional areas for design variable


const double preCpted_A[10] = {30, 1.62, 22.9, 13.5, 1.62, 1.62, 7.97, 26.5, 22, 1.8};

int element[NUM_OF_ELEMENTS][2] = { {3, 5}, {1, 3}, {4, 6}, {2, 4}, {3, 4}, 
                                    {1, 2}, {4, 5}, {3, 6}, {2, 3}, {1, 4} };

int gCoord[2][6] = {{720, 720, 360, 360, 0, 0},
                    {360, 0, 360, 0, 360, 0}};

double Xl[NUM_OF_ELEMENTS] = {1.62,1.62,1.62,1.62,1.62,1.62,1.62,1.62,1.62,1.62},
       Xu[NUM_OF_ELEMENTS] = {33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50};

double K[TOTAL_DOF][TOTAL_DOF] = {0};

inline double getWeight(double A, double matDen, double len)
{
    return (A * matDen * len);
}

void fix(double *X, double *A, int len)
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

void getTransposeOfTe(MatrixT* inputMat, MatrixT* outputMat)
{
    int i, j;
    allocateMatrix(outputMat, inputMat->cols, inputMat->rows);
    for (i = 0; i < inputMat->cols; i++)
        for (j = 0; j < inputMat->rows; j++)
            B[i][j] = A[j][i];
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
    MatrixT Te, Te_Transpose, invK, F, productOf_invK_F;
    MatrixT ke2x2, ke4x4;
    MatrixT matrix2x2_Precomputed, output2x2, output4x2, output4x4; //line 57 in 10 bars
    int index[4];
    int bcDOF[4] = {9, 10, 11, 12};
    double bcValue[4] = {0};

    allocateMatrix(&Te, 2, 4);
    allocateMatrix(&Te_Transpose, 4, 2);
    allocateMatrix(&ke2x2, 2, 2);
    allocateMatrix(&matrix2x2_Precomputed, 2, 2);
    allocateMatrix(&F, TOTAL_DOF, 1); //12x1

    matrix2x2_Precomputed[0][0] = 1;
    matrix2x2_Precomputed[0][1] = -1;
    matrix2x2_Precomputed[1][0] = -1;
    matrix2x2_Precomputed[1][1] = 1;
    zerosMatrix(&F);
    

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
        multiplyScalarMatrix((A[i]*E/le), &matrix2x2_Precomputed, &ke2x2);
        getTransposeOfTe(&Te, &Te_Transpose);
        multiplyMatrices(&Te_Transpose, &ke2x2, output4x2); //line 59
        multiplyMatrices(output4x2, Te, output4x4);

        //Find index assemble in line 60
        index[0] = 2*element[i][0] - 1;
        index[1] = 2*element[i][0];
        index[2] = 2*element[i][1] - 1;
        index[3] = 2*element[i][1];

        //line 63
        for (int row_i = 0; row_i < 4; row_i++)
        {
            for (int col_i = 0; col_i < 4; col_i++)
                K[index[row_i]][index[col_i]] =  K[index[row_i]][index[col_i]] + output4x4[row_i][col_i];
        }
    }

    F[3][0] = F[7][0] = -P;

    for (int bc_i = 0; bc_i < 4; bc_i++)
    {
        int temp = bcDOF[bc_i];
        for (int zeros_i = 0; zeros_i < TOTAL_DOF; zeros_i++)
            K[temp][zeros_i] = 0;

        K[temp][temp] = 1;
        F[temp] = bcValue[bc_i];
    }

    //Calculate U = K\F. inv(K)*F
    LU_getInverseMatrix(&K, &invK, TOTAL_DOF);
    multiplyMatrices(&F, &invK, &productOf_invK_F);


    /* TODO: Deallocate */
    return 1;
}