#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "matrix.h"

#define NUM_OF_ELEMENTS 200
#define NUM_OF_NODES 77
#define DOF 2
#define TOTAL_DOF 154

//#define LOAD_CASE_1
#define LOAD_CASE_2


const double standard_A[30] = {0.100, 0.347, 0.440, 0.539, 0.954, 1.081, 1.174, 1.333, 1.488, 1.764, 2.142, 2.697, 2.800,
                               3.131, 3.565, 3.813, 4.805, 5.952, 6.572, 7.192, 8.525, 9.300, 10.850,
                               13.330, 14.290, 17.170, 19.180, 23.680, 28.080, 33.700}; //Standard cross-sectional areas for design variable in^2

double Xl = 0.100; 
double Xu = 33.700;

const int D = 29;

const int gCoord[2][NUM_OF_NODES] = { {0   , 240 , 480 , 720 , 960 , 0   , 120 , 240 , 360 , 480 , 600 , 720 , 840 , 960 , 0   , 240 , 480 , 720 , 960 , 0   , 120 , 240 , 360 , 480 , 600 , 720 , 840 , 960 , 0   , 240 , 480 , 720 , 960 , 0   , 120 , 240 , 360 , 480 , 600 , 720 , 840 , 960 , 0  , 240, 480, 720, 960, 0  , 120, 240, 360, 480, 600, 720, 840, 960, 0  , 240, 480, 720, 960, 0  , 120, 240, 360, 480, 600, 720, 840, 960, 0  , 240, 480, 720, 960, 240, 720},
                                      {1800, 1800, 1800, 1800, 1800, 1656, 1656, 1656, 1656, 1656, 1656, 1656, 1656, 1656, 1512, 1512, 1512, 1512, 1512, 1368, 1368, 1368, 1368, 1368, 1368, 1368, 1368, 1368, 1224, 1224, 1224, 1224, 1224, 1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080, 936, 936, 936, 936, 936, 792, 792, 792, 792, 792, 792, 792, 792, 792, 648, 648, 648, 648, 648, 504, 504, 504, 504, 504, 504, 504, 504, 504, 360, 360, 360, 360, 360, 0  , 0  } };
#if 0
const int element[NUM_OF_ELEMENTS][2] = 
{ {1, 2}, {2, 3}, {3, 4}, {4, 5}, {6, 1}, {7, 1}, {7, 2}, {8, 2}, {9, 2}, {9, 3}, {10, 3}, {11, 3}, 
  {11, 4}, {12, 4}, {13, 4}, {13, 5}, {14, 5} /*E17*/, {6, 7}, {7, 8}, {8, 9}, {9, 10}, {10, 11}, {11, 12}, {12, 13}, {13, 14}, //E25
  {15, 6}, {15, 7}, {16, 7}, {16, 8}, {16, 9}, {17, 9}, {17, 10}, {17, 11}, {18, 11}, {18, 12}, {18, 13}, {19, 13}, {19, 14}, //E38
  {15, 16}, {16, 17}, {17, 18}, {18, 19}, {20, 15}, {21, 15}, {21, 16}, {22, 16}, {23, 16}, {23, 17}, {24, 17}, {25, 17}, {25, 18}, {26, 18}, {27, 18}, {27, 19}, {28, 19}, //E55
  {20, 21}, {21, 22}, {22, 23}, {23, 24}, {24, 25}, {25, 26}, {26, 27}, {27, 28} /*E63*/, {29, 20}, {29, 21}, {30, 21}, {30, 22}, {30, 23}, {31, 23}, {31, 24}, {31, 25},
  {32, 25}, {32, 26}, {32, 27}, {33,27}, {33, 28} /*E76*/, {29, 30}, {30, 31}, {31, 32}, {32, 33} /*E80*/, {34, 29}, {35, 29}, {35, 30}, {36, 30}, {37, 30}, {37, 31}, {38, 31}, {39, 31},
  {39, 32}, {40, 32}, {41, 32}, {41, 33}, {42, 33} /*93*/, {34, 35}, {35, 36}, {36, 37}, {37, 38}, {38, 39}, {39, 40}, {40, 41}, {41, 42} /*101*/, {43, 34}, {43, 35},
  {44, 35}, {44, 36}, {44, 37}, {45, 37}, {45, 38}, {45, 39}, {46, 39}, {46, 40}, {46, 41}, {47, 41}, {47, 42} /*114*/, {43, 44}, {44, 45}, {45, 46}, {46, 47}, //118
  {48, 43}, {49, 43}, {49, 44}, {50, 44}, {51, 44}, {51, 45}, {52, 45}, {53, 45}, {53, 46}, {54, 46}, {55, 46}, {55, 47}, {56, 47} /* 131 */,
  {48, 49}, {49, 50}, {50, 51}, {51, 52}, {52, 53}, {53, 54}, {54, 55}, {55, 56}/*139*/, {57, 48}, {57, 49}, {58, 49}, {58, 50}, {58, 51}, {59, 51}, {59, 52}, {59, 53},
  {60, 53}, {60, 54}, {60, 55}, {61, 55}, {61, 56} /*152*/, {57, 58}, {58, 59}, {59, 60}, {60, 61} /*156*/, {62, 57}, {63, 57}, {63, 58}, {64, 58}, {65, 58},
  {65, 59}, {66, 59}, {67, 59}, {67, 60}, {68, 60}, {69, 60}, {69, 61}, {70, 61} /*169*/, {62, 63}, {63, 64}, {64, 65}, {65, 66}, {66, 67}, {67, 68}, {68, 69}, {69, 70}, //177
  {71, 62}, {71, 63}, {72, 63}, {72, 64}, {72, 65}, {73, 65}, {73, 66}, {73, 67}, {74, 67}, {74, 68}, {74, 69}, {75, 69}, {75, 70} /*190*/, {71, 72}, {72, 73}, {73, 74}, {74, 75},
  {76, 71}, {76, 72}, {76, 73}, {77, 73}, {77, 74}, {77, 75}}; 
#else
const int element[NUM_OF_ELEMENTS][2] = 
{ {1, 2}, {2, 3}, {3, 4}, {4, 5}, {1, 6}, {1, 7}, {2, 7}, {2, 8}, {2, 9}, {3, 9}, {3, 10}, {3, 11}, 
  {4, 11}, {4, 12}, {4, 13}, {5, 13}, {5, 14} /*E17*/, {6, 7}, {7, 8}, {8, 9}, {9, 10}, {10, 11}, {11, 12}, {12, 13}, {13, 14}, //E25
  {6, 15}, {7, 15}, {7, 16}, {8, 16}, {9, 16}, {9, 17}, {10, 17}, {11, 17}, {11, 18}, {12, 18}, {13, 18}, {13, 19}, {14, 19}, //E38
  {15, 16}, {16, 17}, {17, 18}, {18, 19}, {15, 20}, {15, 21}, {16, 21}, {16, 22}, {16, 23}, {17, 23}, {17, 24}, {17, 25}, {18, 25}, {18, 26}, {18, 27}, {19, 27}, {19, 28}, //E55
  {20, 21}, {21, 22}, {22, 23}, {23, 24}, {24, 25}, {25, 26}, {26, 27}, {27, 28} /*E63*/, {20, 29}, {21, 29}, {21, 30}, {22, 30}, {23, 30}, {23, 31}, {24, 31}, {25, 31},
  {25, 32}, {26, 32}, {27, 32}, {27, 33}, {28, 33} /*E76*/, {29, 30}, {30, 31}, {31, 32}, {32, 33} /*E80*/, {29, 34}, {29, 35}, {30, 35}, {30, 36}, {30, 37}, {31, 37}, {31, 38}, {31, 39},
  {32, 39}, {32, 40}, {32, 41}, {33, 41}, {33, 42} /*93*/, {34, 35}, {35, 36}, {36, 37}, {37, 38}, {38, 39}, {39, 40}, {40, 41}, {41, 42} /*101*/, {34, 43}, {35, 43},
  {35, 44}, {36, 44}, {37, 44}, {37, 45}, {38, 45}, {39, 45}, {39, 46}, {40, 46}, {41, 46}, {41, 47}, {42, 47} /*114*/, {43, 44}, {44, 45}, {45, 46}, {46, 47}, //118
  {43, 48}, {43, 49}, {44, 49}, {44, 50}, {44, 51}, {45, 51}, {45, 52}, {45, 53}, {46, 53}, {46, 54}, {46, 55}, {47, 55}, {47, 56} /* 131 */,
  {48, 49}, {49, 50}, {50, 51}, {51, 52}, {52, 53}, {53, 54}, {54, 55}, {55, 56}/*139*/, {48, 57}, {49, 57}, {49, 58}, {50, 58}, {51, 58}, {51, 59}, {52, 59}, {53, 59},
  {53, 60}, {54, 60}, {55, 60}, {55, 61}, {56, 61} /*152*/, {57, 58}, {58, 59}, {59, 60}, {60, 61} /*156*/, {57, 62}, {57, 63}, {58, 63}, {58, 64}, {58, 65},
  {59, 65}, {59, 66}, {59, 67}, {60, 67}, {60, 68}, {60, 69}, {61, 69}, {61, 70} /*169*/, {62, 63}, {63, 64}, {64, 65}, {65, 66}, {66, 67}, {67, 68}, {68, 69}, {69, 70}, //177
  {62, 71}, {63, 71}, {63, 72}, {64, 72}, {65, 72}, {65, 73}, {66, 73}, {67, 73}, {67, 74}, {68, 74}, {69, 74}, {69, 75}, {70, 75} /*190*/, {71, 72}, {72, 73}, {73, 74}, {74, 75},
  {71, 76}, {72, 76}, {73, 76}, {73, 77}, {74, 77}, {75, 77}}; 
#endif
double stress_e[NUM_OF_ELEMENTS] = {0};

const int index_A1[4] = {1, 2, 3, 4};
const int index_A2[5] = {5, 8, 11, 14, 17};
const int index_A3[6] = {19, 20, 21, 22, 23, 24};
const int index_A4[10] = {18, 25, 56, 63, 94, 101, 132, 139, 170, 177};
const int index_A5[5] = {26, 29, 32, 35, 38};
const int index_A6[16] = {6, 7, 9, 10, 12, 13, 15, 16, 27, 28, 30, 31, 33, 34, 36, 37};
const int index_A7[4] = {39, 40, 41, 42};
const int index_A8[5] = {43, 46, 49, 52, 55};
const int index_A9[6] = {57, 58, 59, 60, 61, 62};
const int index_A10[5] = {64, 67, 70, 73, 76};
const int index_A11[16] = {44, 45, 47, 48, 50, 51, 53, 54, 65, 66, 68, 69, 71, 72, 74, 75};
const int index_A12[4] = {77, 78, 79, 80};
const int index_A13[5] = {81, 84, 87, 90, 93};
const int index_A14[6] = {95, 96, 97, 98, 99, 100};
const int index_A15[5] = {102, 105, 108, 111, 114};
const int index_A16[16] = {82, 83, 85, 86, 88, 89, 91, 92, 103, 104, 106, 107, 109, 110, 112, 113};
const int index_A17[4] = {115, 116, 117, 118};
const int index_A18[5] = {119, 122, 125, 128, 131};
const int index_A19[6] = {133, 134, 135, 136, 137, 138};
const int index_A20[5] = {140, 143, 146, 149, 152};
const int index_A21[16] = {120, 121, 123, 124, 126, 127, 129, 130, 141, 142, 144, 145, 147, 148, 150, 151};
const int index_A22[4] = {153, 154, 155, 156};
const int index_A23[5] = {157, 160, 163, 166, 169};
const int index_A24[6] = {171, 172, 173, 174, 175, 176};
const int index_A25[5] = {178, 181, 184, 187, 190};
const int index_A26[16] = {158, 159, 161, 162, 164, 165, 167, 168, 179, 180, 182, 183, 185, 186, 188, 189};
const int index_A27[4] = {191, 192, 193, 194};
const int index_A28[4] = {195, 197, 198, 200};
const int index_A29[2] = {196, 199};


//const double minStress = -68948000, maxStress = 68948000;
const double minStress = -10.0, maxStress = 10.0;
const double rho = 0.283; //lb/in3
//const double E = 206842800000.0; // N/m2
const double E = 30000;
//const double Px = 4448.2216; //N
//const double Py = 44482.216; //N
const double Px = 1.0; //N
const double Py = (-10.0); //N

inline double getWeight(double A, int length)
{
    return (A * rho * length);
}

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
        for (int j = 0; j < 30; j++)
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

#define BCDOF 4
double func(double *A)
{
    extern const int D;
    double sum = 0.0;
    int x[2], y[2];
    int index[4];
    double l_ij, m_ij, le;
    double Ae[NUM_OF_ELEMENTS];
    const int bcDOF[BCDOF] = {150, 151, 152, 153};
    double bcValue[BCDOF] = {0, 0, 0, 0};
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
    //convertMilliMeterToMeter(A, meterSquare_A); //Pass

    /* Get A for each element */ //Pass
    for (int i = 0; i < 4; i++)
    {
        Ae[index_A1[i] - 1]  = A[0];
        Ae[index_A7[i] - 1]  = A[6];
        Ae[index_A12[i] - 1] = A[11];
        Ae[index_A17[i] - 1] = A[16];
        Ae[index_A22[i] - 1] = A[21];
        Ae[index_A27[i] - 1] = A[26];
        Ae[index_A28[i] - 1] = A[27];
    }

    for (int i = 0; i < 16; i++)
    {
        Ae[index_A6[i] - 1]  = A[5];
        Ae[index_A11[i] - 1] = A[10];
        Ae[index_A16[i] - 1] = A[15];
        Ae[index_A21[i] - 1] = A[20];
        Ae[index_A26[i] - 1] = A[25];
    }

    for (int i = 0; i < 5; i++)
    {
        Ae[index_A2[i] - 1]  = A[1];
        Ae[index_A5[i] - 1] = A[4];
        Ae[index_A8[i] - 1] = A[7];
        Ae[index_A10[i] - 1] = A[9];
        Ae[index_A13[i] - 1] = A[12];
        Ae[index_A15[i] - 1]  = A[14];
        Ae[index_A18[i] - 1] = A[17];
        Ae[index_A20[i] - 1] = A[19];
        Ae[index_A23[i] - 1] = A[22];
        Ae[index_A25[i] - 1] = A[24];
    }

    for (int i = 0; i < 6; i++)
    {
        Ae[index_A3[i] - 1]  = A[2];
        Ae[index_A9[i] - 1] = A[8];
        Ae[index_A14[i] - 1] = A[13];
        Ae[index_A19[i] - 1] = A[18];
        Ae[index_A24[i] - 1] = A[23];
    }

    for (int i = 0; i < 2; i++)
    {
        Ae[index_A29[i] - 1]  = A[28];
    }

    for (int i = 0; i < 10; i++)
    {
        Ae[index_A4[i] - 1]  = A[3];
    }

    //for (int i = 0; i < NUM_OF_ELEMENTS; i++)
    //{
    //    printf("Ae[%d] A[%d] = %f\n", i,indexA[i], Ae[i]);
    //}

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

    // Applying force
#ifdef LOAD_CASE_1
    //Load case 1: x - direction
    F.pMatrix[0][0] = Px; //Node 1
    F.pMatrix[10][0] = Px; //Node 6
    F.pMatrix[28][0] = Px; //Node 15
    F.pMatrix[38][0] = Px; //Node 20
    F.pMatrix[56][0] = Px; //Node 29
    F.pMatrix[66][0] = Px; //Node 34
    F.pMatrix[84][0] = Px; //Node 43
    F.pMatrix[94][0] = Px; //Node 48
    F.pMatrix[112][0] = Px; //Node 57
    F.pMatrix[122][0] = Px; //Node 62
    F.pMatrix[140][0] = Px; //Node 71
#endif
    //Load case 2: y - direction: 55 nodes being applied
#ifdef LOAD_CASE_2
    F.pMatrix[1][0] = Py; //Node 1
    F.pMatrix[3][0] = Py; //Node 2
    F.pMatrix[5][0] = Py; //Node 3
    F.pMatrix[7][0] = Py; //Node 4
    F.pMatrix[9][0] = Py; //Node 5
    F.pMatrix[11][0] = Py; //Node 6
    F.pMatrix[15][0] = Py; //Node 8
    F.pMatrix[19][0] = Py; //Node 10
    F.pMatrix[23][0] = Py; //Node 12
    F.pMatrix[27][0] = Py; //Node 14
    F.pMatrix[29][0] = Py; //Node 15
    F.pMatrix[31][0] = Py; //Node 16
    F.pMatrix[33][0] = Py; //Node 17
    F.pMatrix[35][0] = Py; //Node 18
    F.pMatrix[37][0] = Py; //Node 19
    F.pMatrix[39][0] = Py; //Node 20
    F.pMatrix[43][0] = Py; //Node 22
    F.pMatrix[47][0] = Py; //Node 24
    F.pMatrix[51][0] = Py; //Node 26
    F.pMatrix[55][0] = Py; //Node 28
    F.pMatrix[57][0] = Py; //Node 29
    F.pMatrix[59][0] = Py; //Node 30
    F.pMatrix[61][0] = Py; //Node 31
    F.pMatrix[63][0] = Py; //Node 32
    F.pMatrix[65][0] = Py; //Node 33
    F.pMatrix[67][0] = Py; //Node 34
    F.pMatrix[71][0] = Py; //Node 36
    F.pMatrix[75][0] = Py; //Node 38
    F.pMatrix[79][0] = Py; //Node 40
    F.pMatrix[83][0] = Py; //Node 42
    F.pMatrix[85][0] = Py; //Node 43
    F.pMatrix[87][0] = Py; //Node 44
    F.pMatrix[89][0] = Py; //Node 45
    F.pMatrix[91][0] = Py; //Node 46
    F.pMatrix[93][0] = Py; //Node 47
    F.pMatrix[95][0] = Py; //Node 48
    F.pMatrix[99][0] = Py; //Node 50
    F.pMatrix[103][0] = Py; //Node 52
    F.pMatrix[107][0] = Py; //Node 54
    F.pMatrix[111][0] = Py; //Node 56
    F.pMatrix[113][0] = Py; //Node 57
    F.pMatrix[115][0] = Py; //Node 58
    F.pMatrix[117][0] = Py; //Node 59
    F.pMatrix[119][0] = Py; //Node 60
    F.pMatrix[121][0] = Py; //Node 61
    F.pMatrix[123][0] = Py; //Node 62
    F.pMatrix[127][0] = Py; //Node 64
    F.pMatrix[131][0] = Py; //Node 66
    F.pMatrix[135][0] = Py; //Node 68
    F.pMatrix[139][0] = Py; //Node 70
    F.pMatrix[141][0] = Py; //Node 71
    F.pMatrix[143][0] = Py; //Node 72
    F.pMatrix[145][0] = Py; //Node 73
    F.pMatrix[147][0] = Py; //Node 74
    F.pMatrix[149][0] = Py; //Node 75
#endif //LOAD_CASE_2
    for (int bc_i = 0; bc_i < BCDOF; bc_i++)
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
    double Cstress[NUM_OF_ELEMENTS];
    double sumOfCdisp = 0, sumOfCtress = 0;

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
            //printf("%d - %f\n", i, stress_e[i]);
        }
        sumOfCtress += Cstress[i];
    }
    
    // TODO: calculate total weight
    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum3 = 0.0;
    double sum4 = 0.0;
    double sum5 = 0.0;
    double sum6 = 0.0;
    double sum7 = 0.0;

    for (int i = 0; i < 4; i++) //length (in)
    {
        sum1 = sum1 + (Ae[index_A1[i] - 1] * rho * 240.0);
        sum2 = sum2 + (Ae[index_A7[i] - 1] * rho * 240.0);
        sum3 = sum3 + (Ae[index_A12[i] - 1] * rho * 240.0);
        sum4 = sum4 + (Ae[index_A17[i] - 1] * rho * 240.0);
        sum5 = sum5 + (Ae[index_A22[i] - 1] * rho * 240.0);
        sum6 = sum6 + (Ae[index_A27[i] - 1] * rho * 240.0);
        sum7 = sum7 + (Ae[index_A28[i] - 1] * rho * 432.67);
    }

    for (int i = 0; i < 5; i++)
    {
        sum1 = sum1 + (Ae[index_A2[i] - 1] * rho * 144.0);
        sum2 = sum2 + (Ae[index_A5[i] - 1] * rho * 144.0);
        sum3 = sum3 + (Ae[index_A8[i] - 1] * rho * 144.0);
        sum4 = sum4 + (Ae[index_A10[i] - 1] * rho * 144.0);
        sum5 = sum5 + (Ae[index_A13[i] - 1] * rho * 144.0);
        sum6 = sum6 + (Ae[index_A15[i] - 1] * rho * 144.0);
        sum7 = sum7 + (Ae[index_A18[i] - 1] * rho * 144.0);
        sum1 = sum1 + (Ae[index_A20[i] - 1] * rho * 144.0);
        sum2 = sum2 + (Ae[index_A23[i] - 1] * rho * 144.0);
        sum3 = sum3 + (Ae[index_A25[i] - 1] * rho * 144.0);
    }

    for (int i = 0; i < 6; i++)
    {
        sum1 = sum1 + (Ae[index_A3[i] - 1] * rho * 120.0);
        sum2 = sum2 + (Ae[index_A9[i] - 1] * rho * 120.0);
        sum3 = sum3 + (Ae[index_A14[i] - 1] * rho * 120.0);
        sum4 = sum4 + (Ae[index_A19[i] - 1] * rho * 120.0);
        sum5 = sum5 + (Ae[index_A24[i] - 1] * rho * 120.0);
    }

    for (int i = 0; i < 16; i++)
    {
        sum1 = sum1 + (Ae[index_A6[i] - 1] * rho * 187.45);
        sum2 = sum2 + (Ae[index_A11[i] - 1] * rho * 187.45);
        sum3 = sum3 + (Ae[index_A16[i] - 1] * rho * 187.45);
        sum4 = sum4 + (Ae[index_A21[i] - 1] * rho * 187.45);
        sum5 = sum5 + (Ae[index_A26[i] - 1] * rho * 187.45);
    }

    for (int i = 0; i < 10; i++)
    {
        sum1 = sum1 + (Ae[index_A4[i] - 1] * rho * 120.0);
    }

    for (int i = 0; i < 2; i++)
    {
        sum1 = sum1 + (Ae[index_A29[i] - 1] * rho * 360);
    }
    sum = sum1+sum2+sum3+sum4+sum5+sum6+sum7;

    

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