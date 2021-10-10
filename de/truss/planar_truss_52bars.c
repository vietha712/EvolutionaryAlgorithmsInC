#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "matrix.h"

#define NUM_OF_ELEMENTS 52
#define NUM_OF_NODES 20
#define DOF 2
#define TOTAL_DOF 40

#if 1

const double standard_A[64] = {71.613, 90.968, 126.451, 161.29, 198.064, 252.258, 285.161, 363.225, 388.386, 494.193,
                               506.451, 641.289, 645.16, 792.256, 816.773, 939.998, 1008.385,
                               1045.159, 1161.288, 1283.868, 1374.191, 1535.481, 1690.319, 1696.771, 1858.061,
                               1890.319, 1993.544, 2019.351, 2180.641, 2238.705, 2290.318, 2341.931, 2477.414,
                               2496.769, 2503.221, 2696.769, 2722.575, 2896.768, 2961.284, 3096.768, 3206.445,
                               3303.219, 3703.218, 4658.055, 5141.925, 5503.215, 5999.988, 6999.986, 7419.340, 8709.660, 8967.724, 9161.272,
                               9999.980, 10322.560, 10903.204, 12129.008, 12838.684, 14193.520, 14774.164, 15806.420, 17096.740, 18064.480, 19354.800, 21612.860};


double Xl[NUM_OF_ELEMENTS] = {71.613,71.613,71.613,71.613,71.613,71.613,71.613,71.613,71.613,71.613,
                              71.613,71.613,71.613,71.613,71.613,71.613,71.613,71.613,71.613,71.613,
                              71.613,71.613,71.613,71.613,71.613,71.613,71.613,71.613,71.613,71.613,
                              71.613,71.613,71.613,71.613,71.613,71.613,71.613,71.613,71.613,71.613,
                              71.613,71.613,71.613,71.613,71.613,71.613,71.613,71.613,71.613,71.613,
                              71.613,71.613},

       Xu[NUM_OF_ELEMENTS] = {21612.860,21612.860,21612.860,21612.860,21612.860,21612.860,21612.860,21612.860,21612.860,21612.860,
                              21612.860,21612.860,21612.860,21612.860,21612.860,21612.860,21612.860,21612.860,21612.860,21612.860,
                              21612.860,21612.860,21612.860,21612.860,21612.860,21612.860,21612.860,21612.860,21612.860,21612.860,
                              21612.860,21612.860,21612.860,21612.860,21612.860,21612.860,21612.860,21612.860,21612.860,21612.860,
                              21612.860,21612.860,21612.860,21612.860,21612.860,21612.860,21612.860,21612.860,21612.860,21612.860,
                              21612.860,21612.860};
#endif

#if 0
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
#endif

const int D = 12;

const int gCoord[2][20] = { {0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6},
                            {0, 0, 0, 0, 3, 3, 3, 3, 6, 6, 6, 6, 9, 9, 9, 9, 12, 12, 12, 12} };

const int element[NUM_OF_ELEMENTS][2] = 
{ {1, 5}, {2, 6}, {3, 7}, {4, 8}, {2, 5}, {1, 6}, {3, 6}, {2, 7}, {4, 7}, {3, 8}, {5, 6}, {6, 7}, 
  {7, 8}, {5, 9}, {6, 10}, {7, 11}, {8, 12}, {6, 9}, {5, 10}, {7, 10}, {6, 11}, {8, 11}, {7, 12}, {9, 10},
  {10, 11}, {11, 12}, {9, 13}, {10, 14}, {11, 15}, {12, 16}, {10, 13}, {9, 14}, {11, 14}, {10, 15}, {12, 15}, {11, 16},
  {13, 14}, {14, 15}, {15, 16}, {13, 17}, {14, 18}, {15, 19}, {16, 20}, {14, 17}, {13, 18}, {15, 18}, {14, 19}, {16, 19},
  {15, 20}, {17, 18}, {18, 19}, {19, 20} };

double stress_e[NUM_OF_ELEMENTS] = {0};

const int indexA[NUM_OF_ELEMENTS] = {1, 1, 1, 1, 
                                     2, 2, 2, 2, 2, 2, 
                                     3, 3, 3,
                                     4, 4, 4, 4,
                                     5, 5, 5, 5, 5, 5, 
                                     6, 6, 6, 
                                     7, 7, 7, 7, 
                                     8, 8, 8, 8, 8, 8,
                                     9, 9, 9,
                                     10, 10, 10, 10,
                                     11, 11, 11, 11, 11, 11,
                                     12, 12, 12 };

const double minDisp = -0.0508, maxDisp = 0.0508; //in
const double minStress = -180000000, maxStress = 180000000;
const double rho = 7860; //kg/m3
//const double rho = 0.284; //lb/in3
const double E = 207000000000.000; // N/m2
const double Px = 100000; //N
const double Py = 200000; //N

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

static void convertMilliMeterToMeter(double *A, double *meterA)
{
    for (int i = 0; i < D; i++)
    {
        meterA[i] = A[i] * 0.000001;
    }
}

/*
 * @in: A is in mm2
 * A is then converted to m2 for our problem calculation
 * @out: the weight corresponding to the given problem.
 */
double func(double *A)
{
    extern const int D;
    double sum = 0.0;
    int x[2], y[2];
    int index[4];
    double l_ij, m_ij, le;
    double Ae[NUM_OF_ELEMENTS];
    double meterSquare_A[12];
    const int bcDOF[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    double bcValue[8] = {0};
    MatrixT K, F, Te, Te_Transpose, U, invK, Be, disp_e, de_o, temp;
    MatrixT matrix2x2_Precomputed, ke2x2, output4x2, output4x4, productOfBe_de;

    allocateMatrix(&K, TOTAL_DOF, TOTAL_DOF);
    allocateMatrix(&F, TOTAL_DOF, 1);
    allocateMatrix(&Te, 2, 4);
    allocateMatrix(&Te_Transpose, 4, 2);
    allocateMatrix(&ke2x2, 2, 2);
    allocateMatrix(&matrix2x2_Precomputed, 2, 2);
    allocateMatrix(&Be, 1, 2);
    allocateMatrix(&disp_e, 4, 1);

    initMatrix(&output4x2);
    initMatrix(&output4x4);
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
    convertMilliMeterToMeter(A, meterSquare_A); //Pass

    /* Get A for each element */ //Pass
    for (int i = 0; i < NUM_OF_ELEMENTS; i++)
    {
        Ae[i] = meterSquare_A[indexA[i] - 1];
        printf("Ae[%d] A[%d] = %f\n", i,indexA[i], Ae[i]);
    }

    /* Compute stiffness matrix */
    for (int i = 0; i < NUM_OF_ELEMENTS; i++)
    {
        x[0] = gCoord[0][element[i][0] - 1];
        x[1] = gCoord[0][element[i][1] - 1];
        y[0] = gCoord[1][element[i][0] - 1];
        y[1] = gCoord[1][element[i][1] - 1];

        le = sqrt( pow((x[1] - x[0]), 2) + pow((y[1] - y[0]), 2) );

        //Compute direction cosin
        l_ij = (x[1] - x[0])/le;
        m_ij = (y[1] - y[0])/le;

        //Compute transform matrix
        Te.pMatrix[0][0] = l_ij; Te.pMatrix[0][1] = m_ij; Te.pMatrix[0][2] = 0; Te.pMatrix[0][3] = 0;
        Te.pMatrix[1][0] = 0; Te.pMatrix[1][1] = 0; Te.pMatrix[1][2] = l_ij; Te.pMatrix[1][3] = m_ij;
        
        // Compute stiffness martix of element line 56
        multiplyScalarMatrix((Ae[i]*E/le), &matrix2x2_Precomputed, &ke2x2);
        getTransposeOfTe(&Te, &Te_Transpose);
        multiplyMatrices(&Te_Transpose, &ke2x2, &output4x2); //line 59
        multiplyMatrices(&output4x2, &Te, &output4x4);

        //Find index assemble
        index[0] = 2*element[i][0] - 1 - 1;
        index[1] = 2*element[i][0] - 1;
        index[2] = 2*element[i][1] - 1 - 1;
        index[3] = 2*element[i][1] - 1;

        for (int row_i = 0; row_i < 4; row_i++)
        {
            for (int col_i = 0; col_i < 4; col_i++)
                K.pMatrix[index[row_i]][index[col_i]] =  K.pMatrix[index[row_i]][index[col_i]] + output4x4.pMatrix[row_i][col_i];
        }
    }

    F.pMatrix[32][0] = Px;
    F.pMatrix[33][0] = Py;
    F.pMatrix[34][0] = Px;
    F.pMatrix[35][0] = Py;
    F.pMatrix[36][0] = Px;
    F.pMatrix[37][0] = Py; 
    F.pMatrix[38][0] = Px;
    F.pMatrix[39][0] = Py;

    for (int bc_i = 0; bc_i < 8; bc_i++)
    {
        int temp = bcDOF[bc_i];
        for (int zeros_i = 0; zeros_i < TOTAL_DOF; zeros_i++)
            K.pMatrix[temp][zeros_i] = 0;

        K.pMatrix[temp][temp] = 1;
        F.pMatrix[temp][0] = bcValue[bc_i];
    }

    //Calculate U = K\F. inv(K)*F
    LU_getInverseMatrix(&K, &invK);
    multiplyMatrices(&invK, &F, &U); //U is nodal displacement of each element //Pass U

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

    /*********************** Check constraints violation ******************************/
    double Cdisp[NUM_OF_NODES*2], Cstress[NUM_OF_ELEMENTS];
    double sumOfCdisp = 0, sumOfCtress = 0;

    //Displacement constraints
    for (int i = 0; i < NUM_OF_NODES*2; i++)
    {
        if ((minDisp <= U.pMatrix[i][0]) && (U.pMatrix[i][0] <= maxDisp))
        {
            Cdisp[i] = 0;
        }
        else
        {
            Cdisp[i] = fabs(((U.pMatrix[i][0] - maxDisp)/maxDisp));
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
    
    // TODO: calculate total weight
    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum3 = 0.0;
    double sum4 = 0.0;
    for (int i = 0; i < 4; i++)
    {
        sum1 = sum1 + (Ae[i] * rho * 3.0);
        sum2 = sum2 + (Ae[13+i] * rho * 3.0);
        sum3 = sum3 + (Ae[26+i] * rho * 3.0);
        sum4 = sum4 + (Ae[39+i] * rho * 3.0);
    }

    for (int i = 0; i < 6; i++)
    {
        sum1 = sum1 + (Ae[4+i] * rho * 3.6);
        sum2 = sum2 + (Ae[17+i] * rho * 3.6);
        sum3 = sum3 + (Ae[30+i] * rho * 3.6);
        sum4 = sum4 + (Ae[43+i] * rho * 3.6);
    }

    for (int i = 0; i < 3; i++)
    {
        sum1 = sum1 + (Ae[10+i] * rho * 2.0);
        sum2 = sum2 + (Ae[23+i] * rho * 2.0);
        sum3 = sum3 + (Ae[36+i] * rho * 2.0);
        sum4 = sum4 + (Ae[49+i] * rho * 2.0);
    }
    sum = sum1 +sum2+sum3+sum4;
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
    deallocateMatrix(&output4x2);
    deallocateMatrix(&output4x4);

    return (sum * pow((sumOfCtress + sumOfCdisp + 1), 1));
}