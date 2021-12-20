#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "matrix.h"

#define NUM_OF_ELEMENTS 72
#define NUM_OF_NODES 20
#define DOF 3
#define TOTAL_DOF 60

const double standard_A[64] = {0.111, 0.141, 0.196, 0.25, 0.307, 0.391, 0.442, 0.563, 0.602, 0.766,
                      0.785, 0.994, 1.00, 1.228, 1.266, 1.457, 1.563,
                      1.62, 1.80, 1.99, 2.13, 2.38, 2.62, 2.63, 2.88, 2.93, 3.09, 3.13, 3.38,
                      3.47, 3.55, 3.63, 3.84, 3.87, 3.88, 4.18, 4.22, 4.49, 4.59, 4.80, 4.97,
                      5.12, 5.74, 7.22, 7.97, 8.53, 9.3, 10.85, 11.50, 13.50, 13.90, 14.20, 15.50, 16.00, 16.90,
                      18.80, 19.90, 22.00, 22.90, 24.5, 26.50, 28.0, 30.00, 33.50}; //Standard cross-sectional areas for design variable in^2

double Xl[NUM_OF_ELEMENTS] = {0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111,
                              0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111,
                              0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111,
                              0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111,
                              0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111,
                              0.111,0.111},

       Xu[NUM_OF_ELEMENTS] = {33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50,
                              33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50,
                              33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50,
                              33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50,
                              33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50,
                              33.50,33.50};

const int gCoord[DOF][NUM_OF_NODES] = { {0, 120, 120, 0, 0, 120, 120, 0, 0, 120, 120, 0, 0, 120, 120, 0, 0, 120, 120, 0},
                                        {0, 0, 120, 120, 0, 0, 120, 120, 0, 0, 120, 120, 0, 0, 120, 120, 0, 0, 120, 120},
                                        {0, 0, 0, 0, 60, 60, 60, 60, 120, 120, 120, 120, 180, 180, 180, 180, 240, 240, 240, 240} };

const int element[NUM_OF_ELEMENTS][2] = { {1, 5}, {2, 6}, {3, 7}, {4, 8}, {1, 6}, {2, 5}, {3, 6}, {2, 7}, {3, 8}, {4, 7},
                                          {4, 5}, {1, 8}, {5, 6}, {6 ,7}, {7, 8}, {8, 5}, {5, 7}, {6, 8}, {5, 9}, {6, 10},
                                          {7, 11}, {8, 12}, {5, 10}, {6, 9}, {7, 10}, {6, 11}, {7, 12}, {8, 11}, {8, 9}, {5, 12},
                                          {9, 10}, {10, 11}, {11, 12}, {12, 9}, {9, 11}, {10, 12}, {9, 13}, {10, 14}, {11, 15}, {12, 16},
                                          {9, 14}, {10, 13}, {11, 14}, {10, 15}, {11, 16}, {12, 15}, {12, 13}, {9, 16}, {13, 14}, {14, 15},
                                          {15, 16}, {16, 13}, {13, 15}, {14, 16}, {13, 17}, {14, 18}, {15, 19}, {16, 20}, {13, 18}, {14, 17},
                                          {15, 18}, {14, 19}, {15, 20}, {16, 19}, {16, 17}, {13, 20}, {17, 18}, {18, 19}, {19, 20}, {20, 17},
                                          {17, 19}, {18, 20} };

const int indexA[NUM_OF_ELEMENTS] = { 1, 1, 1, 1, 
                                      2, 2, 2, 2, 2, 2, 2, 2,
                                      3, 3, 3, 3,
                                      4, 4,
                                      5, 5, 5, 5,
                                      6, 6, 6, 6, 6, 6, 6, 6,
                                      7, 7, 7, 7,
                                      8, 8,
                                      9, 9, 9, 9,
                                      10, 10, 10, 10, 10, 10, 10, 10,
                                      11, 11, 11, 11,
                                      12, 12,
                                      13, 13, 13, 13,
                                      14, 14, 14, 14, 14, 14, 14, 14,
                                      15, 15, 15, 15,
                                      16, 16 };

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
        for (int j = 0; j < 64; j++)
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

void getTransposeOfTe(MatrixT* inputMat, MatrixT* outputMat)
{
    int i, j;
    allocateMatrix(outputMat, inputMat->cols, inputMat->rows);
    for (i = 0; i < inputMat->cols; i++)
        for (j = 0; j < inputMat->rows; j++)
            outputMat->pMatrix[i][j] = inputMat->pMatrix[j][i];
}
const int D = 16;
double func(double *A)
{
    const int E = 10000000;
    const int P = 5000;
    const double minDisp = -0.25, maxDisp = 0.25; //in
    const double minStress = -25000, maxStress = 25000;
    const double rho = 0.1;
    extern const int D;
    double sum = 0.0;
    double elementWeight[NUM_OF_ELEMENTS];
    int x[2], y[2], z[2];
    int index[6];
    double l_ij, m_ij, n_ij, le;
    double Ae[NUM_OF_ELEMENTS];
    const int bcDOF[12] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    double bcValue[12] = {0};
    double stress_e[NUM_OF_ELEMENTS] = {0};
    MatrixT K, F, Te, Te_Transpose, U, invK, Be, disp_e, de_o, temp;
    MatrixT matrix2x2_Precomputed, ke2x2, output6x2, output6x6, productOfBe_de;

    allocateMatrix(&K, TOTAL_DOF, TOTAL_DOF);
    allocateMatrix(&F, TOTAL_DOF, 1);
    allocateMatrix(&Te, 2, 6);
    allocateMatrix(&Te_Transpose, 6, 2);
    allocateMatrix(&ke2x2, 2, 2);
    allocateMatrix(&matrix2x2_Precomputed, 2, 2);
    allocateMatrix(&Be, 1, 2);
    allocateMatrix(&disp_e, 6, 1);

    initMatrix(&output6x2);
    initMatrix(&output6x6);
    initMatrix(&invK);
    initMatrix(&U);
    initMatrix(&de_o);
    initMatrix(&productOfBe_de);

    matrix2x2_Precomputed.pMatrix[0][0] = 1;
    matrix2x2_Precomputed.pMatrix[0][1] = -1;
    matrix2x2_Precomputed.pMatrix[1][0] = -1;
    matrix2x2_Precomputed.pMatrix[1][1] = 1;
    zerosMatrix(&K);
    zerosMatrix(&F);

    /* Convert the unit for A */
    //convertMilliMeterToMeter(A, meterSquare_A); //Pass

    /* Get A for each element */ //Pass
    for (int i = 0; i < NUM_OF_ELEMENTS; i++)
    {
        Ae[i] = A[indexA[i] - 1];
    }

    /* Compute stiffness matrix */
    for (int i = 0; i < NUM_OF_ELEMENTS; i++)
    {
        x[0] = gCoord[0][element[i][0] - 1];
        x[1] = gCoord[0][element[i][1] - 1];
        y[0] = gCoord[1][element[i][0] - 1];
        y[1] = gCoord[1][element[i][1] - 1];
        z[0] = gCoord[2][element[i][0] - 1];
        z[1] = gCoord[2][element[i][1] - 1];

        le = sqrt( pow((x[1] - x[0]), 2) + pow((y[1] - y[0]), 2) +  pow((z[1] - z[0]), 2));

        elementWeight[i] =  (Ae[i] * rho * le);
        //Compute direction cosin
        l_ij = (x[1] - x[0])/le;
        m_ij = (y[1] - y[0])/le;
        n_ij = (z[1] - z[0])/le;

        //Compute transform matrix
        Te.pMatrix[0][0] = l_ij; Te.pMatrix[0][1] = m_ij; Te.pMatrix[0][2] = n_ij; Te.pMatrix[0][3] = 0; Te.pMatrix[0][4] = 0; Te.pMatrix[0][5] = 0;
        Te.pMatrix[1][0] = 0; Te.pMatrix[1][1] = 0; Te.pMatrix[1][2] = 0; Te.pMatrix[1][3] = l_ij; Te.pMatrix[1][4] = m_ij; Te.pMatrix[1][5] = n_ij;
        
        // Compute stiffness martix of element
        multiplyScalarMatrix((Ae[i]*E/le), &matrix2x2_Precomputed, &ke2x2);
        getTransposeOfTe(&Te, &Te_Transpose);
        multiplyMatrices(&Te_Transpose, &ke2x2, &output6x2); //line 59
        multiplyMatrices(&output6x2, &Te, &output6x6); // -> ok here

        //Find index assemble - Index - OK
        index[0] = 3*element[i][0] - 1 - 1 - 1;
        index[1] = 3*element[i][0] - 1 - 1;
        index[2] = 3*element[i][0] - 1;
        index[3] = 3*element[i][1] - 1 - 1 - 1;
        index[4] = 3*element[i][1] - 1 - 1;
        index[5] = 3*element[i][1] - 1;

        for (int row_i = 0; row_i < 6; row_i++)
        {
            for (int col_i = 0; col_i < 6; col_i++)
                K.pMatrix[index[row_i]][index[col_i]] =  K.pMatrix[index[row_i]][index[col_i]] + output6x6.pMatrix[row_i][col_i];
        }
    }

    F.pMatrix[50][0] = -P;
    F.pMatrix[53][0] = -P;
    F.pMatrix[56][0] = -P;
    F.pMatrix[59][0] = -P;

    for (int bc_i = 0; bc_i < 12; bc_i++)
    {
        int temp = bcDOF[bc_i];
        for (int zeros_i = 0; zeros_i < TOTAL_DOF; zeros_i++)
            K.pMatrix[temp][zeros_i] = 0;

        K.pMatrix[temp][temp] = 1;
        F.pMatrix[temp][0] = bcValue[bc_i];
    } // F - OK
    //Calculate U = K\F. inv(K)*F
    LU_getInverseMatrix(&K, &invK);
    multiplyMatrices(&invK, &F, &U);
    //printMatrix(&U);

    /* Compute stress for each element */
    for (int i = 0; i < NUM_OF_ELEMENTS; i++)
    {
        x[0] = gCoord[0][element[i][0] - 1];
        x[1] = gCoord[0][element[i][1] - 1];
        y[0] = gCoord[1][element[i][0] - 1];
        y[1] = gCoord[1][element[i][1] - 1];
        z[0] = gCoord[2][element[i][0] - 1];
        z[1] = gCoord[2][element[i][1] - 1];

        le = sqrt( pow((x[1] - x[0]), 2) + pow((y[1] - y[0]), 2) +  pow((z[1] - z[0]), 2));

        //Compute direction cosin
        l_ij = (x[1] - x[0])/le;
        m_ij = (y[1] - y[0])/le;
        n_ij = (z[1] - z[0])/le;

        //Compute transform matrix
        Te.pMatrix[0][0] = l_ij; Te.pMatrix[0][1] = m_ij; Te.pMatrix[0][2] = n_ij; Te.pMatrix[0][3] = 0; Te.pMatrix[0][4] = 0; Te.pMatrix[0][5] = 0;
        Te.pMatrix[1][0] = 0; Te.pMatrix[1][1] = 0; Te.pMatrix[1][2] = 0; Te.pMatrix[1][3] = l_ij; Te.pMatrix[1][4] = m_ij; Te.pMatrix[1][5] = n_ij;

        //compute strain matrix
        Be.pMatrix[0][0] = -1/le;
        Be.pMatrix[0][1] = 1/le;
        
        //Compute displacement of each bar
        index[0] = 3*element[i][0] - 1 - 1 - 1;
        index[1] = 3*element[i][0] - 1 - 1;
        index[2] = 3*element[i][0] - 1;
        index[3] = 3*element[i][1] - 1 - 1 - 1;
        index[4] = 3*element[i][1] - 1 - 1;
        index[5] = 3*element[i][1] - 1;
        disp_e.pMatrix[0][0] = U.pMatrix[index[0]][0];
        disp_e.pMatrix[1][0] = U.pMatrix[index[1]][0];
        disp_e.pMatrix[2][0] = U.pMatrix[index[2]][0];
        disp_e.pMatrix[3][0] = U.pMatrix[index[3]][0];
        disp_e.pMatrix[4][0] = U.pMatrix[index[4]][0];
        disp_e.pMatrix[5][0] = U.pMatrix[index[5]][0];

        multiplyMatrices(&Te, &disp_e, &de_o);
        //compute stress of element
        multiplyMatrices(&Be, &de_o, &productOfBe_de);

        multiplyScalarMatrix(E, &productOfBe_de, &temp); //1x1
        stress_e[i] = temp.pMatrix[0][0];
    }
    /*********************** Check constraints violation ******************************/
    double Cdisp[NUM_OF_NODES*3], Cstress[NUM_OF_ELEMENTS];
    double sumOfCdisp = 0, sumOfCtress = 0;

    //Displacement constraints
    for (int i = 0; i < NUM_OF_NODES*3; i++)
    {
        if ((minDisp <= U.pMatrix[i][0]) && (U.pMatrix[i][0] <= maxDisp))
        {
            Cdisp[i] = 0;
        }
        else
        {
            Cdisp[i] = fabs(((U.pMatrix[i][0] - maxDisp)/maxDisp));
            //printf("Disp %d\n", i);
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
            //printf("Stress %f\n", stress_e[i]);
            //Cstress[i] = stress_e[i];//aeDE paper
        }
        sumOfCtress += Cstress[i];
    }
    
    // TODO: calculate total weight
    for(int i = 0; i < NUM_OF_ELEMENTS; i++)
    {
        sum += elementWeight[i];
    }
    printf("%f\n",sum);

    /* Deallocate */
    deallocateMatrix(&temp);
    deallocateMatrix(&Te);
    deallocateMatrix(&Te_Transpose);
    deallocateMatrix(&invK);
    deallocateMatrix(&F);
    deallocateMatrix(&ke2x2);
    deallocateMatrix(&Be);
    deallocateMatrix(&U);
    deallocateMatrix(&disp_e);
    deallocateMatrix(&de_o);
    deallocateMatrix(&productOfBe_de);
    deallocateMatrix(&matrix2x2_Precomputed);
    deallocateMatrix(&output6x2);
    deallocateMatrix(&output6x6);

    return (sum * pow((sumOfCtress + sumOfCdisp + 1), 1));
}