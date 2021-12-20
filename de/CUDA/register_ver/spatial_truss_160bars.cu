#include "spatial_truss_160bars.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "matrix_improved.cuh"

#define NUM_OF_ELEMENTS 160
#define NUM_OF_NODES 52
#define DOF 3
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


__host__ __device__ static void getTransposeOfTe1D(Matrix1DT* inputMat, Matrix1DT* outputMat);

__host__ __device__ static void getTransposeOfTe1D(Matrix1DT* inputMat, Matrix1DT* outputMat)
{
    int i, j;
    //allocateMatrix1D(outputMat, inputMat->cols, inputMat->rows);
    outputMat->rows = inputMat->cols; outputMat->cols = inputMat->rows; outputMat->isInit = 1;
    for (i = 0; i < inputMat->cols; i++)
        for (j = 0; j < inputMat->rows; j++)
            outputMat->pMatrix[i * outputMat->cols + j] = inputMat->pMatrix[j * inputMat->cols + i];
}

__host__ __device__ static float getR(float A)
{
    const float standard_A[STANDARD_SET_LENGTH] = {1.84, 2.26, 2.66,
                               3.07, 3.47, 3.88, 4.79, 5.27, 5.75, 6.25, 6.84, 7.44, 8.06, 8.66, 9.40,
                               10.47, 11.38, 12.21, 13.79, 15.39, 17.03, 19.03, 21.12, 23.20,
                               25.12, 27.50, 29.88, 32.76, 33.90, 34.77, 39.16, 43.00, 45.65,
                               46.94, 51.00, 52.10, 61.82, 61.90, 68.30, 76.38, 90.60, 94.13}; //cm2

    const float r[STANDARD_SET_LENGTH] = {0.47, 0.57, 0.67, 0.77, 0.87, 0.97, 0.97, 1.06, 1.16, 1.26,
                                       1.15, 1.26, 1.36, 1.46, 1.35, 1.36, 1.45, 1.55, 1.75, 1.95, 1.74, 1.94,
                                       2.16, 2.36, 2.57, 2.35, 2.56, 2.14, 2.33, 2.97, 2.54, 2.93, 2.94, 2.94,
                                       2.92, 3.54, 3.96, 3.52, 3.51, 3.93, 3.92, 3.92};
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

__host__ __device__ float functional(const float * __restrict A, const int D, float * d_invK, float * d_localLU, float * d_s)
{
    const float gCoord[DOF * NUM_OF_NODES] = {-105, 105, 105, -105, -93.929, 93.929, 93.929, -93.929, -82.859, 82.859, 82.859, -82.859, -71.156, 71.156, 71.156, -71.156, -60.085, 60.085, 60.085, -60.085, -49.805, 49.805, 49.805, -49.805, -214, -40, 40, 214, 40, -40, -40, 40, 40, -40, -40, 40, -207, 40, -40, -40, 40, 40, -40, -26.592, 26.592, 26.592, -26.592, -12.737, 12.737, 12.737, -12.737, 0,
                                       -105, -105, 105, 105, -93.929, -93.929, 93.929, 93.929, -82.859, -82.859, 82.859, 82.859, -71.156, -71.156, 71.156, 71.156, -60.085, -60.085, 60.085, 60.085, -49.805, -49.805, 49.805, 49.805, 0, -40, -40, 0, 40, 40, -40, -40, 40, 40, -40, -40, 0, 40, 40, -40, -40, 40, 40, -26.592, -26.592, 26.5920, 26.592, -12.737, -12.737, 12.737, 12.737, 0,
                                       0, 0, 0, 0, 175, 175, 175, 175, 350, 350, 350, 350, 535, 535, 535, 535, 710, 710, 710, 710, 872.50, 872.50, 872.50, 872.50, 1027.5, 1027.5, 1027.5, 1027.5, 1027.5, 1027.5, 1105.5, 1105.5, 1105.5, 1105.5, 1256.5, 1256.5, 1256.5, 1256.5, 1256.5, 1346.5, 1346.5, 1346.5, 1346.5, 1436.5, 1436.5, 1436.5, 1436.5, 1526.5, 1526.5, 1526.5, 1526.5, 1615};

    const int element[NUM_OF_ELEMENTS * 2] = {1, 5, 2, 6, 3, 7, 4, 8, 1, 6, 2, 5, 2, 7, 3, 6, 3, 8, 4, 7, 4, 5, 1, 8, 5, 9, 6, 10, 7, 11, 8, 12, 5, 10, 6, 9, 6, 11, 7, 10, 7, 12, 8, 11, 8, 9, 5, 12, 9, 13, 10, 14, 11, 15, 12, 16, 9, 14, 10, 13, 10, 15, 11, 14, 11, 16, 12, 15, 12, 13, 9, 16, 13, 17, 14, 18, 15, 19, 16, 20, 13, 18, 14, 17, 14, 19, 15, 18,
                                         15, 20, 16, 19, 16, 17, 13, 20, 17, 21, 18, 22, 19, 23, 20, 24, 17, 22, 18, 21, 19, 24, 20, 23, 18, 23, 19, 22, 20, 21, 17, 24, 21, 26, 22, 27, 23, 29, 24, 30, 21, 27, 22, 26, 23, 30, 24, 29, 22, 29, 23, 27, 24, 26, 21, 30, 26, 27, 27, 29, 29, 30, 30, 26, 25, 26, 27, 28, 25, 30, 29, 28, 25, 31, 28, 32, 28, 33, 25, 34,
                                         26, 31, 27, 32, 29, 33, 30, 34, 26, 32, 27, 31, 29, 34, 30, 33, 27, 33, 29, 32, 30, 31, 26, 34, 26, 29, 27, 30, 31, 35, 32, 36, 33, 38, 34, 39, 33, 39, 32, 35, 31, 36, 34, 38, 32, 38, 33, 36, 34, 35, 31, 39, 37, 35, 37, 39, 37, 40, 37, 43, 35, 40, 36, 41, 38, 42, 39, 43, 35, 38, 36, 39, 36, 40, 38, 41, 39, 42, 35, 43,
                                         40, 41, 41, 42, 42, 43, 43, 40, 35, 36, 36, 38, 38, 39, 39, 35, 40, 44, 41, 45, 42, 46, 43, 47, 40, 45, 41, 46, 42, 47, 43, 44, 44, 45, 45, 46, 46, 47, 44, 47, 44, 48, 45, 49, 46, 50, 47, 51, 45, 48, 46, 49, 47, 50, 44, 51, 48, 49, 49, 50, 50, 51, 48, 51, 48, 52, 49, 52, 50, 52, 51, 52};

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
    
    /****************************************************************************************************************/
    const int E = 2047000;
    const float rho = 0.00785;
    float sum = 0.0, r;
    int x[2], y[2], z[2];
    int index[6];
    float l_ij, m_ij, n_ij, le;
    float Ae[NUM_OF_ELEMENTS];
    const int bcDOF[12] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    float bcValue[12] = {0};
    float stress_e[NUM_OF_ELEMENTS] = {0};
    float bucklingStress[NUM_OF_ELEMENTS] = {0};
    Matrix1DT K, F, Te, Te_Transpose, U, invK, Be, disp_e, de_o, temp;
    Matrix1DT matrix2x2_Precomputed, ke2x2, output6x2, output6x6, productOfBe_de;

    float K_array[TOTAL_DOF*TOTAL_DOF];
    allocateMatrix1D(&K, K_array, TOTAL_DOF, TOTAL_DOF);
    
    float F_array[TOTAL_DOF];
    allocateMatrix1D(&F, F_array, TOTAL_DOF, 1);

    //Te.rows = 2; Te.cols = 6; Te.isInit = 1;
    float Te_array[12]; //rows * cols
    allocateMatrix1D(&Te, Te_array, 2, 6);

    //Te_Transpose.rows = 6; Te_Transpose.cols = 2; Te_Transpose.isInit = 1;
    float Te_Transpose_array[12]; //rows * cols
    allocateMatrix1D(&Te_Transpose, Te_Transpose_array, 6, 2);

    float ke2x2_array[4]; //rows * cols
    allocateMatrix1D(&ke2x2, ke2x2_array, 2, 2);
    float matrix2x2_Precomputed_array[4];
    allocateMatrix1D(&matrix2x2_Precomputed, matrix2x2_Precomputed_array, 2, 2);
    
    float Be_array[2];
    allocateMatrix1D(&Be, Be_array, 1, 2);
    //disp_e.rows = 6; disp_e.cols = 1; disp_e.isInit = 1;
    float disp_e_array[6];
    allocateMatrix1D(&disp_e, disp_e_array, 6, 1);
    /*********************************/
    float output6x2_array[12];
    float output6x6_array[36];

    float U_array[TOTAL_DOF];
    float de_o_array[TOTAL_DOF/2];
    float productOfBe_de_array[TOTAL_DOF/2];
    initMatrix(&output6x2);
    initMatrix(&output6x6);
    initMatrix(&invK);
    initMatrix(&U);
    initMatrix(&de_o);
    initMatrix(&productOfBe_de);

    matrix2x2_Precomputed.pMatrix[0] = 1;
    matrix2x2_Precomputed.pMatrix[1] = -1;
    matrix2x2_Precomputed.pMatrix[2] = -1;
    matrix2x2_Precomputed.pMatrix[3] = 1;
    zerosMatrix1D(&K);
    zerosMatrix1D(&F);

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
        x[0] = gCoord[element[i * 2] - 1];
        x[1] = gCoord[element[i * 2 + 1] - 1];
        y[0] = gCoord[NUM_OF_NODES + element[i * 2] - 1];
        y[1] = gCoord[NUM_OF_NODES + element[i *2 + 1] - 1];
        z[0] = gCoord[NUM_OF_NODES * 2+ element[i * 2] - 1];
        z[1] = gCoord[NUM_OF_NODES * 2 + element[i * 2 + 1] - 1];

        le = sqrt( pow((x[1] - x[0]), 2) + pow((y[1] - y[0]), 2) +  pow((z[1] - z[0]), 2));

        sum += (Ae[i] * rho * le);

        //Compute direction cosin
        l_ij = (x[1] - x[0])/le;
        m_ij = (y[1] - y[0])/le;
        n_ij = (z[1] - z[0])/le;

        //Compute transform matrix
        Te.pMatrix[0] = l_ij; Te.pMatrix[1] = m_ij; Te.pMatrix[2] = n_ij; Te.pMatrix[3] = 0; Te.pMatrix[4] = 0; Te.pMatrix[5] = 0;
        Te.pMatrix[6] = 0; Te.pMatrix[7] = 0; Te.pMatrix[8] = 0; Te.pMatrix[9] = l_ij; Te.pMatrix[10] = m_ij; Te.pMatrix[11] = n_ij;
        
        // Compute stiffness martix of element
        multiplyScalarMatrix1D((Ae[i]*E/le), &matrix2x2_Precomputed, ke2x2_array, &ke2x2);
        getTransposeOfTe1D(&Te, &Te_Transpose);
        multiplyMatrices1D(&Te_Transpose, &ke2x2, output6x2_array, &output6x2); //line 59
        multiplyMatrices1D(&output6x2, &Te, output6x6_array, &output6x6); // -> ok here

        //Find index assemble - Index - OK
        index[0] = 3*element[i * 2] - 1 - 1 - 1;
        index[1] = 3*element[i * 2] - 1 - 1;
        index[2] = 3*element[i * 2] - 1;
        index[3] = 3*element[i * 2 + 1] - 1 - 1 - 1;
        index[4] = 3*element[i * 2 + 1] - 1 - 1;
        index[5] = 3*element[i * 2 + 1] - 1;

        for (int row_i = 0; row_i < 6; row_i++)
        {
            for (int col_i = 0; col_i < 6; col_i++)
                K.pMatrix[index[row_i] * K.cols + index[col_i]] =  K.pMatrix[index[row_i] * K.cols + index[col_i]] + output6x6.pMatrix[row_i * output6x6.cols + col_i];
        }
    }

    for (int bc_i = 0; bc_i < 12; bc_i++)
    {
        int temp = bcDOF[bc_i];
        for (int zeros_i = 0; zeros_i < TOTAL_DOF; zeros_i++)
            K.pMatrix[temp * K.cols + zeros_i] = 0;

        K.pMatrix[temp * K.cols + temp] = 1;
        F.pMatrix[temp * F.cols] = bcValue[bc_i];
    } // F - OK

    LU_getInverseMatrix1D(&K, d_invK, &invK, d_localLU, d_s);

    float sumOfCtress = 0;
    for(int loadcase_i = 0; loadcase_i < 8; loadcase_i++)
    {
        switch(loadcase_i)
        {
            case 0:
                F.pMatrix[153 * F.cols] = P_MINUS_868;
                F.pMatrix[155 * F.cols] = P_MINUS_491;
                F.pMatrix[108 * F.cols] = P_MINUS_996;
                F.pMatrix[110 * F.cols] = P_MINUS_546;
                F.pMatrix[72* F.cols] = P_MINUS_1091;
                F.pMatrix[74* F.cols] = P_MINUS_546;
                F.pMatrix[81* F.cols] = P_MINUS_1091;
                F.pMatrix[83* F.cols] = P_MINUS_546;
                //printf("case 1\n");
                break;
            case 1:
                F.pMatrix[153 * F.cols] = P_MINUS_493;
                F.pMatrix[154 * F.cols] = P_1245;
                F.pMatrix[155 * F.cols] = P_MINUS_363;
                F.pMatrix[108 * F.cols] = P_MINUS_996;
                F.pMatrix[110 * F.cols] = P_MINUS_546;
                F.pMatrix[72 * F.cols] = P_MINUS_1091;
                F.pMatrix[74 * F.cols] = P_MINUS_546;
                F.pMatrix[81 * F.cols] = P_MINUS_1091;
                F.pMatrix[83 * F.cols] = P_MINUS_546;
                //printf("case 2\n");
                break;
            case 2:
                F.pMatrix[153 * F.cols] = P_MINUS_917;
                F.pMatrix[154 * F.cols] = 0;
                F.pMatrix[155 * F.cols] = P_MINUS_491;
                F.pMatrix[108 * F.cols] = P_MINUS_951;
                F.pMatrix[110 * F.cols] = P_MINUS_546;
                F.pMatrix[72 * F.cols] = P_MINUS_1015;
                F.pMatrix[74 * F.cols] = P_MINUS_546;
                F.pMatrix[81 * F.cols] = P_MINUS_1015;
                F.pMatrix[83 * F.cols] = P_MINUS_546;
                //printf("case 3\n");
                break;
            case 3:
                F.pMatrix[153 * F.cols] = P_MINUS_917;
                F.pMatrix[155 * F.cols] = P_MINUS_546;
                F.pMatrix[108 * F.cols] = P_MINUS_572;
                F.pMatrix[109 * F.cols] = P_1259;
                F.pMatrix[110 * F.cols] = P_MINUS_428;
                F.pMatrix[72 * F.cols]= P_MINUS_1015;
                F.pMatrix[74 * F.cols]= P_MINUS_546;
                F.pMatrix[81 * F.cols]= P_MINUS_1015;
                F.pMatrix[83 * F.cols]= P_MINUS_546;
                //printf("case 4\n");
                break;
            case 4:
                F.pMatrix[153 * F.cols] = P_MINUS_917;
                F.pMatrix[155 * F.cols] = P_MINUS_491;
                F.pMatrix[108 * F.cols] = P_MINUS_951;
                F.pMatrix[109 * F.cols] = 0;
                F.pMatrix[110 * F.cols] = P_MINUS_546;
                F.pMatrix[72 * F.cols] = P_MINUS_1015;
                F.pMatrix[74 * F.cols] = P_MINUS_546;
                F.pMatrix[81 * F.cols] = P_MINUS_636;
                F.pMatrix[82 * F.cols] = P_1259;
                F.pMatrix[83 * F.cols] = P_MINUS_428;
                //printf("case 5\n");
                break;
            case 5:
                F.pMatrix[153 * F.cols] = P_MINUS_917;
                F.pMatrix[155 * F.cols] = P_MINUS_491;
                F.pMatrix[108 * F.cols] = P_MINUS_572;
                F.pMatrix[109 * F.cols] = P_1303;
                F.pMatrix[110 * F.cols] = P_MINUS_428;
                F.pMatrix[72 * F.cols] = P_MINUS_1015;
                F.pMatrix[74 * F.cols] = P_MINUS_546;
                F.pMatrix[81 * F.cols] = P_MINUS_1015;
                F.pMatrix[82 * F.cols] = 0;
                F.pMatrix[83 * F.cols] = P_MINUS_546;
                //printf("case 6\n");
                break;
            case 6:
                F.pMatrix[153 * F.cols] = P_MINUS_917;
                F.pMatrix[155 * F.cols] = P_MINUS_491;
                F.pMatrix[108 * F.cols] = P_MINUS_951;
                F.pMatrix[109 * F.cols] = 0;
                F.pMatrix[110 * F.cols] = P_MINUS_546;
                F.pMatrix[72 * F.cols] = P_MINUS_1015;
                F.pMatrix[74 * F.cols] = P_MINUS_546;
                F.pMatrix[81 * F.cols] = P_MINUS_636;
                F.pMatrix[82 * F.cols] = P_1303;
                F.pMatrix[83 * F.cols] = P_MINUS_428;
                //printf("case 7\n");
                break;
            case 7:
                F.pMatrix[153 * F.cols] = P_MINUS_498;
                F.pMatrix[154 * F.cols] = P_1460;
                F.pMatrix[155 * F.cols] = P_MINUS_363;
                F.pMatrix[108 * F.cols] = P_MINUS_951;
                F.pMatrix[110 * F.cols] = P_MINUS_546;
                F.pMatrix[72 * F.cols] = P_MINUS_1015;
                F.pMatrix[74 * F.cols] = P_MINUS_546;
                F.pMatrix[81 * F.cols] = P_MINUS_1015;
                F.pMatrix[82 * F.cols] = 0;
                F.pMatrix[83 * F.cols] = P_MINUS_546;
                //printf("case 8\n");
                break;
            default:
            break;
        }

        //Calculate U = K\F. inv(K)*F
        multiplyMatrices1D(&invK, &F, U_array, &U);

        /* Compute stress for each element */
        for (int i = 0; i < NUM_OF_ELEMENTS; i++)
        {
            x[0] = gCoord[element[i * 2] - 1];
            x[1] = gCoord[element[i * 2 + 1] - 1];
            y[0] = gCoord[NUM_OF_NODES + element[i * 2] - 1];
            y[1] = gCoord[NUM_OF_NODES + element[i *2 + 1] - 1];
            z[0] = gCoord[NUM_OF_NODES * 2+ element[i * 2] - 1];
            z[1] = gCoord[NUM_OF_NODES * 2 + element[i * 2 + 1] - 1];

            le = sqrt( pow((x[1] - x[0]), 2) + pow((y[1] - y[0]), 2) +  pow((z[1] - z[0]), 2));

            //Compute direction cosin
            l_ij = (x[1] - x[0])/le;
            m_ij = (y[1] - y[0])/le;
            n_ij = (z[1] - z[0])/le;

            //Compute transform matrix
            Te.pMatrix[0] = l_ij; Te.pMatrix[1] = m_ij; Te.pMatrix[2] = n_ij; Te.pMatrix[3] = 0; Te.pMatrix[4] = 0; Te.pMatrix[5] = 0;
            Te.pMatrix[6] = 0; Te.pMatrix[7] = 0; Te.pMatrix[8] = 0; Te.pMatrix[9] = l_ij; Te.pMatrix[10] = m_ij; Te.pMatrix[11] = n_ij;

            //compute strain matrix
            Be.pMatrix[0] = -1/le;
            Be.pMatrix[1] = 1/le;
            
            //Compute displacement of each bar
            index[0] = 3*element[i * 2] - 1 - 1 - 1;
            index[1] = 3*element[i * 2] - 1 - 1;
            index[2] = 3*element[i * 2] - 1;
            index[3] = 3*element[i * 2 + 1] - 1 - 1 - 1;
            index[4] = 3*element[i * 2 + 1] - 1 - 1;
            index[5] = 3*element[i * 2 + 1] - 1;
            disp_e.pMatrix[0] = U.pMatrix[index[0] * U.cols];
            disp_e.pMatrix[1] = U.pMatrix[index[1] * U.cols];
            disp_e.pMatrix[2] = U.pMatrix[index[2] * U.cols];
            disp_e.pMatrix[3] = U.pMatrix[index[3] * U.cols];
            disp_e.pMatrix[4] = U.pMatrix[index[4] * U.cols];
            disp_e.pMatrix[5] = U.pMatrix[index[5] * U.cols];

            multiplyMatrices1D(&Te, &disp_e, de_o_array, &de_o);
            //compute stress of element
            multiplyMatrices1D(&Be, &de_o, productOfBe_de_array, &productOfBe_de);
            float temp_array[1];
            multiplyScalarMatrix1D(E, &productOfBe_de, temp_array, &temp); //1x1
            stress_e[i] = temp.pMatrix[0];

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
        float Cstress[NUM_OF_ELEMENTS] = {0};
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
    }
    
    return (sum * pow((sumOfCtress + 1), 1));
}