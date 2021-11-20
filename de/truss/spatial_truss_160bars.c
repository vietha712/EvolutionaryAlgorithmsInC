#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "matrix.h"

#define NUM_OF_ELEMENTS 160
#define NUM_OF_NODES 52
#define DOF 3
#define TOTAL_DOF (NUM_OF_NODES * DOF)
#define STANDARD_SET_LENGTH 42

#define P_MINUS_1091 (-1091)
#define P_MINUS_868  (-868)
#define P_MINUS_996  (-996)
#define P_MINUS_493  (-493)
#define P_MINUS_917  (-917)
#define P_MINUS_951  (-951)
#define P_MINUS_1015 (-1015)
#define P_MINUS_572  (-572)
#define P_MINUS_546  (-546)
#define P_MINUS_428  (-428)
#define P_MINUS_363  (-363)
#define P_MINUS_636  (-636)
#define P_MINUS_498  (-498)
#define P_MINUS_491  (-491)
#define P_1245       (1245)
#define P_1259       (1259)
#define P_1303       (1303)
#define P_1460       (1460)

const double standard_A[STANDARD_SET_LENGTH] = {1.84, 2.26, 2.66,
                               3.07, 3.47, 3.88, 4.79, 5.27, 5.75, 6.25, 6.84, 7.44, 8.06, 8.66, 9.40,
                               10.47, 11.38, 12.21, 13.79, 15.39, 17.03, 19.03, 21.12, 23.20,
                               25.12, 27.50, 29.88, 32.76, 33.90, 34.77, 39.16, 43.00, 45.65,
                               46.94, 51.00, 52.10, 61.82, 61.90, 68.30, 76.38, 90.60, 94.13}; //cm2

const double r[STANDARD_SET_LENGTH] = {0.47, 0.57, 0.67, 0.77, 0.87, 0.97, 0.97, 1.06, 1.16, 1.26,
                                       1.15, 1.26, 1.36, 1.46, 1.35, 1.36, 1.45, 1.55, 1.75, 1.95, 1.74, 1.94,
                                       2.16, 2.36, 2.57, 2.35, 2.56, 2.14, 2.33, 2.97, 2.54, 2.93, 2.94, 2.94,
                                       2.92, 3.54, 3.96, 3.52, 3.51, 3.93, 3.92, 3.92};

const double Xl = 1.84, Xu = 94.13;

const double gCoord[DOF][NUM_OF_NODES] = {{-105, 105, 105, -105, -93.929, 93.929, 93.929, -93.929, -82.859, 82.859, 82.859, -82.859, -71.156, 71.156, 71.156, -71.156, -60.085, 60.085, 60.085, -60.085, -49.805, 49.805, 49.805, -49.805, -214, -40, 40, 214, 40, -40, -40, 40, 40, -40, -40, 40, -207, 40, -40, -40, 40, 40, -40, -26.592, 26.592, 26.592, -26.592, -12.737, 12.737, 12.737, -12.737, 0},
                                       {-105, -105, 105, 105, -93.929, -93.929, 93.929, 93.929, -82.859, -82.859, 82.859, 82.859, -71.156, -71.156, 71.156, 71.156, -60.085, -60.085, 60.085, 60.085, -49.805, -49.805, 49.805, 49.805, 0, -40, -40, 0, 40, 40, -40, -40, 40, 40, -40, -40, 0, 40, 40, -40, -40, 40, 40, -26.592, -26.592, 26.5920, 26.592, -12.737, -12.737, 12.737, 12.737, 0},
                                       {0, 0, 0, 0, 175, 175, 175, 175, 350, 350, 350, 350, 535, 535, 535, 535, 710, 710, 710, 710, 872.50, 872.50, 872.50, 872.50, 1027.5, 1027.5, 1027.5, 1027.5, 1027.5, 1027.5, 1105.5, 1105.5, 1105.5, 1105.5, 1256.5, 1256.5, 1256.5, 1256.5, 1256.5, 1346.5, 1346.5, 1346.5, 1346.5, 1436.5, 1436.5, 1436.5, 1436.5, 1526.5, 1526.5, 1526.5, 1526.5, 1615}};

const int element[NUM_OF_ELEMENTS][2] = {{1, 5}, {2, 6}, {3, 7}, {4, 8}, {1, 6}, {2, 5}, {2, 7}, {3, 6}, {3, 8}, {4, 7}, {4, 5}, {1, 8}, {5, 9}, {6, 10}, {7, 11}, {8, 12}, {5, 10}, {6, 9}, {6, 11}, {7, 10}, {7, 12}, {8, 11}, {8, 9}, {5, 12}, {9, 13}, {10, 14}, {11, 15}, {12, 16}, {9, 14}, {10, 13}, {10, 15}, {11, 14}, {11, 16}, {12, 15}, {12, 13}, {9, 16}, {13, 17}, {14, 18}, {15, 19}, {16, 20}, {13, 18}, {14, 17}, {14, 19}, {15, 18},
                                         {15, 20}, {16, 19}, {16, 17}, {13, 20}, {17, 21}, {18, 22}, {19, 23}, {20, 24}, {17, 22}, {18, 21}, {19, 24}, {20, 23}, {18, 23}, {19, 22}, {20, 21}, {17, 24}, {21, 26}, {22, 27}, {23, 29}, {24, 30}, {21, 27}, {22, 26}, {23, 30}, {24, 29}, {22, 29}, {23, 27}, {24, 26}, {21, 30}, {26, 27}, {27, 29}, {29, 30}, {30, 26}, {25, 26}, {27, 28}, {25, 30}, {29, 28}, {25, 31}, {28, 32}, {28, 33}, {25, 34},
                                         {26, 31}, {27, 32}, {29, 33}, {30, 34}, {26, 32}, {27, 31}, {29, 34}, {30, 33}, {27, 33}, {29, 32}, {30, 31}, {26, 34}, {26, 29}, {27, 30}, {31, 35}, {32, 36}, {33, 38}, {34, 39}, {33, 39}, {32, 35}, {31, 36}, {34, 38}, {32, 38}, {33, 36}, {34, 35}, {31, 39}, {37, 35}, {37, 39}, {37, 40}, {37, 43}, {35, 40}, {36, 41}, {38, 42}, {39, 43}, {35, 38}, {36, 39}, {36, 40}, {38, 41}, {39, 42}, {35, 43},
                                         {40, 41}, {41, 42}, {42, 43}, {43, 40}, {35, 36}, {36, 38}, {38, 39}, {39, 35}, {40, 44}, {41, 45}, {42, 46}, {43, 47}, {40, 45}, {41, 46}, {42, 47}, {43, 44}, {44, 45}, {45, 46}, {46, 47}, {44, 47}, {44, 48}, {45, 49}, {46, 50}, {47, 51}, {45, 48}, {46, 49}, {47, 50}, {44, 51}, {48, 49}, {49, 50}, {50, 51}, {48, 51}, {48, 52}, {49, 52}, {50, 52}, {51, 52}};

const int indexA[NUM_OF_ELEMENTS] = { 1, 1, 1, 1,
                                      2, 2, 2, 2, 2, 2, 2, 2,
                                      3, 3, 3, 3,
                                      4, 4, 4, 4, 4, 4, 4, 4,
                                      5, 5, 5, 5,
                                      6, 6, 6, 6, 6, 6, 6, 6,
                                      7, 7, 7, 7,
                                      8, 8, 8, 8, 8, 8, 8, 8,
                                      9, 9, 9, 9,
                                      10, 10, 10, 10,
                                      11, 11, 11, 11,
                                      12, 12, 12, 12,
                                      13, 13, 13, 13,
                                      14, 14, 14, 14,
                                      15, 15, 15, 15,
                                      16, 16, 16, 16,
                                      17, 17, 17, 17,
                                      18, 18, 18, 18,
                                      19, 19, 19, 19,
                                      20, 20, 20, 20,
                                      21, 21,
                                      22, 22, 22, 22,
                                      23, 23, 23, 23,
                                      24, 24, 24, 24,
                                      25, 25,
                                      26, 26,
                                      27, 27, 27, 27,
                                      28, 28,
                                      29, 29, 29, 29,
                                      30, 30, 30, 30,
                                      31, 31, 31, 31,
                                      32, 32, 32, 32,
                                      33, 33, 33, 33,
                                      34, 34, 34, 34,
                                      35, 35, 35, 35,
                                      36, 36, 36, 36,
                                      37, 37, 37, 37,
                                      38, 38, 38, 38 };

void fix(double *X, int length)
{
    double temp1, temp2;

    for (int i=0; i<length; i++)
    {
        while (X[i] < Xl || X[i] > Xu)
        {
            if (X[i] < Xl) X[i] = 2.0*Xl - X[i];
            if (X[i] > Xu) X[i] = 2.0*Xu - X[i];
        }
    }

    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < STANDARD_SET_LENGTH; j++)
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

double getR(double A)
{
    int index = 0;
    for (int i = 0; i < STANDARD_SET_LENGTH; i++)
    {
        if (A == standard_A[i])
        {
            index = i;
            break;
        }
    }

    return r[index];
}
const int D = 38;
double func(double *A)
{
    const int E = 2047000; //kgf/cm2
    const double rho = 0.00785; //kg/cm3
    extern const int D;
    double sum = 0.0;
    double elementWeight[NUM_OF_ELEMENTS] = {0};
    double x[2], y[2], z[2];
    int index[6];
    double l_ij, m_ij, n_ij, le;
    double Ae[NUM_OF_ELEMENTS];
    const int bcDOF[12] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    double bcValue[12] = {0};
    double r;
    double stress_e[NUM_OF_ELEMENTS] = {0};
    double bucklingStress[NUM_OF_ELEMENTS] = {0};
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

    /* Get A for each element */
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
    // boundary condition 
    for (int bc_i = 0; bc_i < 12; bc_i++)
    {
        int temp = bcDOF[bc_i];
        for (int zeros_i = 0; zeros_i < TOTAL_DOF; zeros_i++)
            K.pMatrix[temp][zeros_i] = 0;

        K.pMatrix[temp][temp] = 1;
        F.pMatrix[temp][0] = bcValue[bc_i];
    } // F - OK

    LU_getInverseMatrix(&K, &invK);

    double sumOfCtress = 0;
    //for(int loadcase_i = 0; loadcase_i < 8; loadcase_i++)
    //{}
    int loadcase_i = 1;
        switch(loadcase_i)
        {
            case 0:
                F.pMatrix[153][0] = P_MINUS_868;
                F.pMatrix[155][0] = P_MINUS_491;
                F.pMatrix[108][0] = P_MINUS_996;
                F.pMatrix[110][0] = P_MINUS_546;
                F.pMatrix[72][0] = P_MINUS_1091;
                F.pMatrix[74][0] = P_MINUS_546;
                F.pMatrix[81][0] = P_MINUS_1091;
                F.pMatrix[83][0] = P_MINUS_546;
                //printf("case 1\n");
                break;
            case 1:
                F.pMatrix[153][0] = P_MINUS_493;
                F.pMatrix[154][0] = P_1245;
                F.pMatrix[155][0] = P_MINUS_363;
                F.pMatrix[108][0] = P_MINUS_996;
                F.pMatrix[110][0] = P_MINUS_546;
                F.pMatrix[72][0] = P_MINUS_1091;
                F.pMatrix[74][0] = P_MINUS_546;
                F.pMatrix[81][0] = P_MINUS_1091;
                F.pMatrix[83][0] = P_MINUS_546;
                //printf("case 2\n");
                break;
            case 2:
                F.pMatrix[153][0] = P_MINUS_917;
                F.pMatrix[154][0] = 0;
                F.pMatrix[155][0] = P_MINUS_491;
                F.pMatrix[108][0] = P_MINUS_951;
                F.pMatrix[110][0] = P_MINUS_546;
                F.pMatrix[72][0] = P_MINUS_1015;
                F.pMatrix[74][0] = P_MINUS_546;
                F.pMatrix[81][0] = P_MINUS_1015;
                F.pMatrix[83][0] = P_MINUS_546;
                //printf("case 3\n");
                break;
            case 3:
                F.pMatrix[153][0] = P_MINUS_917;
                F.pMatrix[155][0] = P_MINUS_546;
                F.pMatrix[108][0] = P_MINUS_572;
                F.pMatrix[109][0] = P_1259;
                F.pMatrix[110][0] = P_MINUS_428;
                F.pMatrix[72][0] = P_MINUS_1015;
                F.pMatrix[74][0] = P_MINUS_546;
                F.pMatrix[81][0] = P_MINUS_1015;
                F.pMatrix[83][0] = P_MINUS_546;
                //printf("case 4\n");
                break;
            case 4:
                F.pMatrix[153][0] = P_MINUS_917;
                F.pMatrix[155][0] = P_MINUS_491;
                F.pMatrix[108][0] = P_MINUS_951;
                F.pMatrix[109][0] = 0;
                F.pMatrix[110][0] = P_MINUS_546;
                F.pMatrix[72][0] = P_MINUS_1015;
                F.pMatrix[74][0] = P_MINUS_546;
                F.pMatrix[81][0] = P_MINUS_636;
                F.pMatrix[82][0] = P_1259;
                F.pMatrix[83][0] = P_MINUS_428;
                //printf("case 5\n");
                break;
            case 5:
                F.pMatrix[153][0] = P_MINUS_917;
                F.pMatrix[155][0] = P_MINUS_491;
                F.pMatrix[108][0] = P_MINUS_572;
                F.pMatrix[109][0] = P_1303;
                F.pMatrix[110][0] = P_MINUS_428;
                F.pMatrix[72][0] = P_MINUS_1015;
                F.pMatrix[74][0] = P_MINUS_546;
                F.pMatrix[81][0] = P_MINUS_1015;
                F.pMatrix[82][0] = 0;
                F.pMatrix[83][0] = P_MINUS_546;
                //printf("case 6\n");
                break;
            case 6:
                F.pMatrix[153][0] = P_MINUS_917;
                F.pMatrix[155][0] = P_MINUS_491;
                F.pMatrix[108][0] = P_MINUS_951;
                F.pMatrix[109][0] = 0;
                F.pMatrix[110][0] = P_MINUS_546;
                F.pMatrix[72][0] = P_MINUS_1015;
                F.pMatrix[74][0] = P_MINUS_546;
                F.pMatrix[81][0] = P_MINUS_636;
                F.pMatrix[82][0] = P_1303;
                F.pMatrix[83][0] = P_MINUS_428;
                //printf("case 7\n");
                break;
            case 7:
                F.pMatrix[153][0] = P_MINUS_498;
                F.pMatrix[154][0] = P_1460;
                F.pMatrix[155][0] = P_MINUS_363;
                F.pMatrix[108][0] = P_MINUS_951;
                F.pMatrix[110][0] = P_MINUS_546;
                F.pMatrix[72][0] = P_MINUS_1015;
                F.pMatrix[74][0] = P_MINUS_546;
                F.pMatrix[81][0] = P_MINUS_1015;
                F.pMatrix[82][0] = 0;
                F.pMatrix[83][0] = P_MINUS_546;
                //printf("case 8\n");
                break;
            default:
            break;
        }

        //Calculate U = K\F. inv(K)*F
        multiplyMatrices(&invK, &F, &U);

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

            r = getR(Ae[i]);
            //Calculate buckling stress
            double le_over_r = le/r;
            if (stress_e[i] < 0)
            {
                if (le_over_r <= 120)
                {
                    bucklingStress[i] = 1300 - ( pow(le_over_r, 2) / 24);
                }
                else
                {
                    bucklingStress[i] = pow(10, 7) / pow(le_over_r, 2);
                }
            }
            else
            {
                bucklingStress[i] = 0;
            }
        }
        /*********************** Check constraints violation ******************************/
        double Cstress[NUM_OF_ELEMENTS] = {0};
        //Stress constraints
        for (int i = 0; i < NUM_OF_ELEMENTS; i++)
        {
            if (stress_e[i] <= 0)
            {
                if (bucklingStress[i] >= abs(stress_e[i]))
                {
                    Cstress[i] = 0;
                }
                else
                {
                    Cstress[i] = fabs((stress_e[i] - 100)/100);
                }
            }
            sumOfCtress += Cstress[i];
        }
    

    //calculate total weight
    for(int i = 0; i < NUM_OF_ELEMENTS; i++)
    {
        sum += elementWeight[i];
    }
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

    return (sum * pow(( sumOfCtress + 1), 1));
}