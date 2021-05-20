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
const double rho = 0.1; // density of material lb/in^3
const int length = 360; //in

void getTransposeOfTe(MatrixT* inputMat, MatrixT* outputMat);

const double standard_A[42] = {1.62, 1.80, 1.99, 2.13, 2.38, 2.62, 2.63, 2.88, 2.93, 3.09, 3.13, 3.38,
                      3.47, 3.55, 3.63, 3.84, 3.87, 3.88, 4.18, 4.22, 4.49, 4.59, 4.80, 4.97,
                      5.12, 5.74, 7.22, 7.97, 11.50, 13.50, 13.90, 14.20, 15.50, 16.00, 16.90,
                      18.80, 19.90, 22.00, 22.90, 26.50, 30.00, 33.50}; //Standard cross-sectional areas for design variable in^2

//const double preCpted_A[10] = {30, 1.62, 22.9, 13.5, 1.62, 1.62, 7.97, 26.5, 22, 1.8};

/*
{ {3, 5}, {1, 3}, {4, 6}, {2, 4}, {3, 4}, 
                                    {1, 2}, {4, 5}, {3, 6}, {2, 3}, {1, 4} };*/
//Reindex                                 
int element[NUM_OF_ELEMENTS][2] = { {3, 5}, {1, 3}, {4, 6}, {2, 4}, {3, 4}, 
                                    {1, 2}, {4, 5}, {3, 6}, {2, 3}, {1, 4} };


double stress_e[NUM_OF_ELEMENTS] = {0};

int gCoord[2][6] = {{720, 720, 360, 360, 0, 0},
                    {360, 0, 360, 0, 360, 0}};

double Xl[NUM_OF_ELEMENTS] = {1.62,1.62,1.62,1.62,1.62,1.62,1.62,1.62,1.62,1.62},
       Xu[NUM_OF_ELEMENTS] = {33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50};

inline double getWeight(double A)
{
    return (A * rho * length);
}

void fix(double *X, int length)
{
    double temp1, temp2;
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < 42; j++)
        {
            if (X[i] > standard_A[j])
            {
                continue;
            }
            X[i] = standard_A[j];
            break;
        }
    }

#if 0
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < 42; j++)
        {
            if ((X[i] == standard_A[j]))
            {
                X[i] = standard_A[j];
                break;
            }
            else
            {
               if (j != 0)
               {
                   temp1 = fabs(X[i] - standard_A[j - 1]);
                   temp2 = fabs(X[i] - standard_A[j]);
                   X[i] = ((temp1 < temp2) ? standard_A[j-1] : standard_A[j]);
                   break;
               }
               else
               {
                   X[i] = standard_A[j];
                   break;
               }
            }
        }
    }
#endif
}

void getTransposeOfTe(MatrixT* inputMat, MatrixT* outputMat)
{
    int i, j;
    allocateMatrix(outputMat, inputMat->cols, inputMat->rows);
    for (i = 0; i < inputMat->cols; i++)
        for (j = 0; j < inputMat->rows; j++)
            outputMat->pMatrix[i][j] = inputMat->pMatrix[j][i];
}

double findMaxMemInArray(double *array, int length)
{
    double max = 0;
    for (int i = 0; i < length; i++)
    {
        max = (array[i] > max) ? array[i] : max;
    }
    return max;
}

const double epsilon_1 = 1.0;
double epsilon_2 = 20.0;
int E = 10000000;
int P = 100000;
const int D = 10;
const double minDisp = -2.0, maxDisp = 2.0;
const double minStress = -25, maxStress = 25;

double func(double *A)
{
    extern const int D;
    double sum = 0.0;
    double le;
    int x[2], y[2];
    double l_ij, m_ij;
    MatrixT Te, Te_Transpose, invK, F, K;
    MatrixT ke2x2, ke4x4, Be, U, disp_e, de_o, productOfBe_de;
    MatrixT matrix2x2_Precomputed, output4x2, output4x4; //line 57 in 10 bars
    int index[4];
    int bcDOF[4] = {8, 9, 10, 11}; //reindex in C. Original 9 - 10 - 11 - 12
    double bcValue[4] = {0};

    allocateMatrix(&Te, 2, 4);
    allocateMatrix(&Te_Transpose, 4, 2);
    allocateMatrix(&ke2x2, 2, 2);
    allocateMatrix(&matrix2x2_Precomputed, 2, 2);
    allocateMatrix(&F, TOTAL_DOF, 1); //12x1
    allocateMatrix(&Be, 1, 2);
    allocateMatrix(&disp_e, 4, 1);
    allocateMatrix(&K, TOTAL_DOF, TOTAL_DOF);
    initMatrix(&invK);
    initMatrix(&ke4x4);
    initMatrix(&U);
    initMatrix(&de_o);
    initMatrix(&productOfBe_de);
    initMatrix(&output4x2);
    initMatrix(&output4x4);


    matrix2x2_Precomputed.pMatrix[0][0] = 1;
    matrix2x2_Precomputed.pMatrix[0][1] = -1;
    matrix2x2_Precomputed.pMatrix[1][0] = -1;
    matrix2x2_Precomputed.pMatrix[1][1] = 1;
    zerosMatrix(&F);
    zerosMatrix(&K);
    
    /* Calculate stiffness matrix */
    for (int i = 0; i < NUM_OF_ELEMENTS; i++)
    {
        x[0] = gCoord[0][element[i][0] - 1];
        x[1] = gCoord[0][element[i][1] - 1];
        y[0] = gCoord[1][element[i][0] - 1];
        y[1] = gCoord[1][element[i][1] - 1];

        le = sqrt( pow((x[1] - x[0]), 2) + pow((y[1] - y[0]), 2) ); //

        //Compute direction cosin
        l_ij = (x[1] - x[0])/le;
        m_ij = (y[1] - y[0])/le;

        //Compute transform matrix
        Te.pMatrix[0][0] = l_ij; Te.pMatrix[0][1] = m_ij; Te.pMatrix[0][2] = 0; Te.pMatrix[0][3] = 0;
        Te.pMatrix[1][0] = 0; Te.pMatrix[1][1] = 0; Te.pMatrix[1][2] = l_ij; Te.pMatrix[1][3] = m_ij;

        // Compute stiffness martix of element line 56
        multiplyScalarMatrix((A[i]*E/le), &matrix2x2_Precomputed, &ke2x2);
        getTransposeOfTe(&Te, &Te_Transpose);
        multiplyMatrices(&Te_Transpose, &ke2x2, &output4x2); //line 59
        multiplyMatrices(&output4x2, &Te, &output4x4);

        //Find index assemble in line 60
        index[0] = 2*element[i][0] - 1 - 1;
        index[1] = 2*element[i][0] - 1;
        index[2] = 2*element[i][1] - 1 - 1;
        index[3] = 2*element[i][1] - 1;

        //line 63
        for (int row_i = 0; row_i < 4; row_i++)
        {
            for (int col_i = 0; col_i < 4; col_i++)
                K.pMatrix[index[row_i]][index[col_i]] =  K.pMatrix[index[row_i]][index[col_i]] + output4x4.pMatrix[row_i][col_i];
        }
    } //Pass K

    F.pMatrix[3][0] = F.pMatrix[7][0] = -P;

    for (int bc_i = 0; bc_i < 4; bc_i++)
    {
        int temp = bcDOF[bc_i];
        for (int zeros_i = 0; zeros_i < TOTAL_DOF; zeros_i++)
            K.pMatrix[temp][zeros_i] = 0;

        K.pMatrix[temp][temp] = 1;
        F.pMatrix[temp][0] = bcValue[bc_i];
    } //Pass K

    //Calculate U = K\F. inv(K)*F
    LU_getInverseMatrix(&K, &invK);
    multiplyMatrices(&invK, &F, &U); //U is nodal displacement of each element //Pass U

    //Get absolute value for U
    MatrixT U_Abs;
    allocateMatrix(&U_Abs, U.rows, U.cols);
    for (int abs_i = 0; abs_i < U.rows; abs_i++)
        for (int abs_j = 0; abs_j < U.cols; abs_j++)
            U_Abs.pMatrix[abs_i][abs_j] = fabs(U.pMatrix[abs_i][abs_j]);

    double Cdisp = findMaxMember(&U_Abs) - 2.0; // max value of nodal displacement
    deallocateMatrix(&U_Abs);

    MatrixT temp;
    /* Compute stress for each element */
    for (int i = 0; i < NUM_OF_ELEMENTS; i++)
    {
        x[0] = gCoord[0][element[i][0] - 1];
        x[1] = gCoord[0][element[i][1] - 1];
        y[0] = gCoord[1][element[i][0] - 1];
        y[1] = gCoord[1][element[i][1] - 1];

        le = sqrt( pow((x[1] - x[0]), 2) + pow((y[1] - y[0]), 2) ); //

        //Compute direction cosin
        l_ij = (x[1] - x[0])/le;
        m_ij = (y[1] - y[0])/le;

        //Compute transform matrix
        Te.pMatrix[0][0] = l_ij; Te.pMatrix[0][1] = m_ij; Te.pMatrix[0][2] = 0; Te.pMatrix[0][3] = 0;
        Te.pMatrix[1][0] = 0; Te.pMatrix[1][1] = 0; Te.pMatrix[1][2] = l_ij; Te.pMatrix[1][3] = m_ij;

        //compute strain matrix
        Be.pMatrix[0][0] = -1/le;
        Be.pMatrix[0][1] = 1/le;
        
        //Compute displacement of each bar
        index[0] = 2*element[i][0] - 1 - 1;
        index[1] = 2*element[i][0] - 1;
        index[2] = 2*element[i][1] - 1 - 1;
        index[3] = 2*element[i][1] - 1;
        disp_e.pMatrix[0][0] = U.pMatrix[index[0]][0];
        disp_e.pMatrix[1][0] = U.pMatrix[index[1]][0];
        disp_e.pMatrix[2][0] = U.pMatrix[index[2]][0];
        disp_e.pMatrix[3][0] = U.pMatrix[index[3]][0];

        multiplyMatrices(&Te, &disp_e, &de_o);
        //compute stress of element
        multiplyMatrices(&Be, &de_o, &productOfBe_de);

        multiplyScalarMatrix(E, &productOfBe_de, &temp); //1x1
        stress_e[i] = temp.pMatrix[0][0];
    }

    double LimitSig = 25000;
    double maxAbsStress = 0.0;
    for (int i = 0; i < NUM_OF_ELEMENTS; i++)
        stress_e[i] = fabs(stress_e[i]);

    for (int i = 0; i < NUM_OF_ELEMENTS; i++)
    {
        if(stress_e[i] > maxAbsStress)
        {
            maxAbsStress = stress_e[i];
        }
    }
    double Csig = (maxAbsStress/LimitSig) - 1; //Pass

    /*********************** Check constraints violation ******************************/
    double weight = 0;
    double v = 0.0; // sum of design violated variables
    double dispViolateVar[10] = {0, 0, 0, 0 ,0 ,0 ,0, 0, 0, 0};
    double stressViolateVar[10] = {0, 0, 0, 0 ,0 ,0 ,0, 0, 0, 0};
    int numOfConstraints = 2;
    int isViolated = 0;
#if 0
    //Displacement constraints
    for (int i = 0; i < NUM_OF_ELEMENTS; i++)
    {
        if ((minDisp <= U.pMatrix[i][0]) && (U.pMatrix[i][0] <= maxDisp))
        {
            continue;
        }
        else
        {
            dispViolateVar[i] = A[i];
            isViolated = 1;
        }
    }

    //Stress constraints
    for (int i = 0; i < NUM_OF_NODES; i++)
    {
        if ((minStress <= stress_e[i]) && (stress_e[i] <= maxStress))
        {
            continue;
        }
        else
        {

            isViolated = 1;
            stressViolateVar[i] = A[i];
        }
    }

    if (1 == isViolated)
    {
        v += findMaxMemInArray(dispViolateVar, D);
        v += findMaxMemInArray(stressViolateVar, D);
    }


#endif
    //TODO: calculate total weight
    for (int i = 0; i < D; i++)
    {
        sum += getWeight(A[i]);
    }

    

    /* Deallocate */
    deallocateMatrix(&temp);
    deallocateMatrix(&Te);
    deallocateMatrix(&Te_Transpose);
    deallocateMatrix(&invK);
    deallocateMatrix(&F);
    deallocateMatrix(&ke2x2);
    deallocateMatrix(&ke4x4);
    deallocateMatrix(&Be);
    deallocateMatrix(&U);
    deallocateMatrix(&disp_e);
    deallocateMatrix(&de_o);
    deallocateMatrix(&productOfBe_de);
    deallocateMatrix(&matrix2x2_Precomputed);
    deallocateMatrix(&output4x2);
    deallocateMatrix(&output4x4);

    return pow((1 + epsilon_1*v), epsilon_2)*sum;
}