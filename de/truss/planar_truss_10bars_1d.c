#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "matrix1d.h"

#define NUM_OF_ELEMENTS 10
#define NUM_OF_NODES 6
#define DOF 2
#define TOTAL_DOF 12 // DOF * NUM_OF_NODES
const double rho = 0.1; // density of material lb/in^3

void getTransposeOfTe(Matrix1DT* inputMat, Matrix1DT* outputMat);

const double standard_A[42] = {1.62, 1.80, 1.99, 2.13, 2.38, 2.62, 2.63, 2.88, 2.93, 3.09, 3.13, 3.38,
                      3.47, 3.55, 3.63, 3.84, 3.87, 3.88, 4.18, 4.22, 4.49, 4.59, 4.80, 4.97,
                      5.12, 5.74, 7.22, 7.97, 11.50, 13.50, 13.90, 14.20, 15.50, 16.00, 16.90,
                      18.80, 19.90, 22.00, 22.90, 26.50, 30.00, 33.50}; //Standard cross-sectional areas for design variable in^2

//const double preCpted_A[10] = {30, 1.62, 22.9, 13.5, 1.62, 1.62, 7.97, 26.5, 22, 1.8};

/*
{ {3, 5}, {1, 3}, {4, 6}, {2, 4}, {3, 4}, 
                                    {1, 2}, {4, 5}, {3, 6}, {2, 3}, {1, 4} };*/
//Reindex                                 
//int element[NUM_OF_ELEMENTS][2] = { {3, 5}, {1, 3}, {4, 6}, {2, 4}, {3, 4}, 
//                                    {1, 2}, {4, 5}, {3, 6}, {2, 3}, {1, 4} };
int element1D[NUM_OF_ELEMENTS * 2] = { 3, 5, 1, 3, 4, 6, 2, 4, 3, 4, 1, 2, 4, 5, 3, 6, 2, 3, 1, 4 };

double stress_e[NUM_OF_ELEMENTS] = {0};

//int gCoord[2][6] = {{720, 720, 360, 360, 0, 0},
//                    {360, 0, 360, 0, 360, 0}};
int gCoord1D[2 * 6] = {720, 720, 360, 360, 0, 0, 360, 0, 360, 0, 360, 0};

double Xl[NUM_OF_ELEMENTS] = {1.62,1.62,1.62,1.62,1.62,1.62,1.62,1.62,1.62,1.62},
       Xu[NUM_OF_ELEMENTS] = {33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50};

inline double getWeight(double A, int length)
{
    return (A * rho * length);
}

void fix(double *X, int length)
{
    double temp1, temp2;

    for (int i=0; i<length; i++)
    {
        while (X[i] < Xl[i] || X[i] > Xu[i])
        {
            if (X[i] < Xl[i]) X[i] = 2.0*Xl[i] - X[i];
            if (X[i] > Xu[i]) X[i] = 2.0*Xu[i] - X[i];
        }
    }

    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < 42; j++)
        {
            if ((X[i] > standard_A[j]))
            {
                continue;
            }
            else
            {
                temp1 = X[i] - standard_A[j];
                temp2 = X[i] - standard_A[j - 1];
                X[i] = (fabs(temp1) <= fabs(temp2)) ? standard_A[j] : standard_A[j - 1];
                break;
            }
        }
    }
}

void getTransposeOfTe1D(Matrix1DT* inputMat, Matrix1DT* outputMat)
{
    int i, j;
    allocateMatrix1D(outputMat, inputMat->cols, inputMat->rows);
    for (i = 0; i < inputMat->cols; i++)
        for (j = 0; j < inputMat->rows; j++)
            outputMat->pMatrix[i * outputMat->cols + j] = inputMat->pMatrix[j * inputMat->cols + i];
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
int E = 10000000; //N/m2
int P = 100000;
const int D = 10;
const double minDisp = -2.0, maxDisp = 2.0;
const double minStress = -25000, maxStress = 25000;

double func(double *A)
{
    extern const int D;
    double sum = 0.0;
    double le;
    int x[2], y[2];
    double l_ij, m_ij;
    Matrix1DT Te, Te_Transpose, invK, F, K, temp;
    Matrix1DT ke2x2, Be, U, disp_e, de_o, productOfBe_de;
    Matrix1DT matrix2x2_Precomputed, output4x2, output4x4; //line 57 in 10 bars
    int index[4];
    int bcDOF[4] = {8, 9, 10, 11}; //reindex in C. Original 9 - 10 - 11 - 12
    double bcValue[4] = {0};

    allocateMatrix1D(&Te, 2, 4);
    allocateMatrix1D(&Te_Transpose, 4, 2);
    allocateMatrix1D(&ke2x2, 2, 2);
    allocateMatrix1D(&matrix2x2_Precomputed, 2, 2);
    allocateMatrix1D(&F, TOTAL_DOF, 1); //12x1
    allocateMatrix1D(&Be, 1, 2);
    allocateMatrix1D(&disp_e, 4, 1);
    allocateMatrix1D(&K, TOTAL_DOF, TOTAL_DOF);
    initMatrix(&invK);
    initMatrix(&U);
    initMatrix(&de_o);
    initMatrix(&productOfBe_de);
    initMatrix(&output4x2);
    initMatrix(&output4x4);


    matrix2x2_Precomputed.pMatrix[0] = 1;
    matrix2x2_Precomputed.pMatrix[1] = -1;
    matrix2x2_Precomputed.pMatrix[2] = -1;
    matrix2x2_Precomputed.pMatrix[3] = 1;
    zerosMatrix1D(&F);
    zerosMatrix1D(&K);
    
    /* Calculate stiffness matrix */
    for (int i = 0; i < NUM_OF_ELEMENTS; i++)
    {
        x[0] = gCoord1D[element1D[i * 2] - 1];
        x[1] = gCoord1D[element1D[i * 2 + 1] - 1];
        y[0] = gCoord1D[6 + element1D[i * 2] - 1];
        y[1] = gCoord1D[6 + element1D[i * 2 + 1] - 1];

        le = sqrt( pow((x[1] - x[0]), 2) + pow((y[1] - y[0]), 2) ); //

        //Compute direction cosin
        l_ij = (x[1] - x[0])/le;
        m_ij = (y[1] - y[0])/le;

        //Compute transform matrix
        Te.pMatrix[0] = l_ij; Te.pMatrix[1] = m_ij; Te.pMatrix[2] = 0; Te.pMatrix[3] = 0;
        Te.pMatrix[4] = 0; Te.pMatrix[5] = 0; Te.pMatrix[6] = l_ij; Te.pMatrix[7] = m_ij;

        // Compute stiffness martix of element line 56
        multiplyScalarMatrix1D((A[i]*E/le), &matrix2x2_Precomputed, &ke2x2);
        getTransposeOfTe1D(&Te, &Te_Transpose);
        multiplyMatrices1D(&Te_Transpose, &ke2x2, &output4x2); //line 59
        multiplyMatrices1D(&output4x2, &Te, &output4x4);

        //Find index assemble in line 60
        index[0] = 2*element1D[i * 2] - 1 - 1;
        index[1] = 2*element1D[i * 2] - 1;
        index[2] = 2*element1D[i * 2 + 1] - 1 - 1;
        index[3] = 2*element1D[i * 2 + 1] - 1;

        //line 63
        for (int row_i = 0; row_i < 4; row_i++)
        {
            for (int col_i = 0; col_i < 4; col_i++)
                K.pMatrix[index[row_i] * K.cols + index[col_i]] =  K.pMatrix[index[row_i] * K.cols + index[col_i]] + output4x4.pMatrix[row_i * output4x4.cols + col_i];
        }
    } //Pass K

    F.pMatrix[3 * F.cols] = F.pMatrix[7 * F.cols] = -P;

    for (int bc_i = 0; bc_i < 4; bc_i++)
    {
        int temp = bcDOF[bc_i];
        for (int zeros_i = 0; zeros_i < TOTAL_DOF; zeros_i++)
            K.pMatrix[temp * K.cols + zeros_i] = 0;

        K.pMatrix[temp * K.cols + temp] = 1;
        F.pMatrix[temp * F.cols] = bcValue[bc_i];
    } //Pass K

    //Calculate U = K\F. inv(K)*F
    LU_getInverseMatrix1D(&K, &invK);
    multiplyMatrices1D(&invK, &F, &U); //U is nodal displacement of each element //Pass U
    //printMatrix1D(&U);
    //Get absolute value for U
#if 0
    Matrix1DT U_Abs;
    allocateMatrix1D(&U_Abs, U.rows, U.cols);
    for (int abs_i = 0; abs_i < U.rows; abs_i++)
        for (int abs_j = 0; abs_j < U.cols; abs_j++)
            U_Abs.pMatrix[abs_i][abs_j] = fabs(U.pMatrix[abs_i][abs_j]);

    double Cdisp = findMaxMember(&U_Abs) - 2.0; // max value of nodal displacement
    deallocateMatrix1D(&U_Abs);
#endif
    /* Compute stress for each element */
    for (int i = 0; i < NUM_OF_ELEMENTS; i++)
    {
        x[0] = gCoord1D[element1D[i * 2] - 1];
        x[1] = gCoord1D[element1D[i * 2 + 1] - 1];
        y[0] = gCoord1D[6 + element1D[i * 2] - 1];
        y[1] = gCoord1D[6 + element1D[i * 2 + 1] - 1];

        le = sqrt( pow((x[1] - x[0]), 2) + pow((y[1] - y[0]), 2) ); //

        //Compute direction cosin
        l_ij = (x[1] - x[0])/le;
        m_ij = (y[1] - y[0])/le;

        //Compute transform matrix
        Te.pMatrix[0] = l_ij; Te.pMatrix[1] = m_ij; Te.pMatrix[2] = 0; Te.pMatrix[3] = 0;
        Te.pMatrix[4] = 0; Te.pMatrix[5] = 0; Te.pMatrix[6] = l_ij; Te.pMatrix[7] = m_ij;

        //compute strain matrix
        Be.pMatrix[0] = -1/le;
        Be.pMatrix[1] = 1/le;
        
        //Compute displacement of each bar
        index[0] = 2*element1D[i * 2] - 1 - 1;
        index[1] = 2*element1D[i * 2] - 1;
        index[2] = 2*element1D[i * 2 + 1] - 1 - 1;
        index[3] = 2*element1D[i * 2 + 1] - 1;
        disp_e.pMatrix[0 * disp_e.cols] = U.pMatrix[index[0] * U.cols];
        disp_e.pMatrix[1 * disp_e.cols] = U.pMatrix[index[1] * U.cols];
        disp_e.pMatrix[2 * disp_e.cols] = U.pMatrix[index[2] * U.cols];
        disp_e.pMatrix[3 * disp_e.cols] = U.pMatrix[index[3] * U.cols];

        multiplyMatrices1D(&Te, &disp_e, &de_o);
        //compute stress of element
        multiplyMatrices1D(&Be, &de_o, &productOfBe_de);
        multiplyScalarMatrix1D(E, &productOfBe_de, &temp); //1x1

        stress_e[i] = temp.pMatrix[0];
    }
#if 0
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
#endif

    /*********************** Check constraints violation ******************************/
#if 1
    double Cdisp[12], Cstress[10];
    double sumOfCdisp = 0, sumOfCtress = 0;

    //Displacement constraints
    for (int i = 0; i < NUM_OF_NODES*2; i++)
    {
        if ((minDisp <= U.pMatrix[i * U.cols]) && (U.pMatrix[i * U.cols] <= maxDisp))
        {
            Cdisp[i] = 0;
        }
        else
        {
            Cdisp[i] = fabs(((U.pMatrix[i * U.cols] - maxDisp)/maxDisp));
            //Cdisp[i] = U.pMatrix[i][0]; //aeDE paper
        }
        sumOfCdisp += Cdisp[i];
    }

    //Stress constraints
    for (int i = 0; i < NUM_OF_ELEMENTS; i++)
    {
        if ((minStress <= stress_e[i]) && (stress_e[i] <= maxStress))
        {
            Cstress[i] = 0;
        }
        else
        {
            Cstress[i] = fabs((stress_e[i] - maxStress)/maxStress);
            //Cstress[i] = stress_e[i];//aeDE paper
        }
        sumOfCtress += Cstress[i];
    }
#endif
    // calculate total weight
    for (int i = 0; i < (D-4); i++)
    {
        sum += getWeight(A[i], 360);
    }
    for (int i = (D-4) ; i < 10; i++)
    {
        sum += getWeight(A[i], 509.12);
    }
#if 0 //MATLAB
    sumOfCdisp = findMaxMemInArray(Cdisp, 10);
    sumOfCtress = findMaxMemInArray(Cstress, 10);
#endif
    /* Deallocate */
    deallocateMatrix1D(&temp);
    deallocateMatrix1D(&Te);
    deallocateMatrix1D(&Te_Transpose);
    deallocateMatrix1D(&invK);
    deallocateMatrix1D(&F);
    deallocateMatrix1D(&ke2x2);
    deallocateMatrix1D(&Be);
    deallocateMatrix1D(&U);
    deallocateMatrix1D(&disp_e);
    deallocateMatrix1D(&de_o);
    deallocateMatrix1D(&productOfBe_de);
    deallocateMatrix1D(&matrix2x2_Precomputed);
    deallocateMatrix1D(&output4x2);
    deallocateMatrix1D(&output4x4);

    return (sum * pow((sumOfCtress + sumOfCdisp + 1), 1));
}