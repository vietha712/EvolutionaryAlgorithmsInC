#include "planar_truss_72bars.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "matrix_improved.cuh"

#define NUM_OF_ELEMENTS 72
#define NUM_OF_NODES 20
#define DOF 3
#define TOTAL_DOF 60

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

__host__ __device__ float function(const float * __restrict A, const int D)
{
    const int gCoord[DOF * NUM_OF_NODES] = { 0, 120, 120, 0, 0, 120, 120, 0, 0, 120, 120, 0, 0, 120, 120, 0, 0, 120, 120, 0,
                                         0, 0, 120, 120, 0, 0, 120, 120, 0, 0, 120, 120, 0, 0, 120, 120, 0, 0, 120, 120,
                                         0, 0, 0, 0, 60, 60, 60, 60, 120, 120, 120, 120, 180, 180, 180, 180, 240, 240, 240, 240 };

    const int element[NUM_OF_ELEMENTS * 2] = { 1, 5, 2, 6, 3, 7, 4, 8, 1, 6, 2, 5, 3, 6, 2, 7, 3, 8, 4, 7,
                                          4, 5, 1, 8, 5, 6, 6 ,7, 7, 8, 8, 5, 5, 7, 6, 8, 5, 9, 6, 10,
                                          7, 11, 8, 12, 5, 10, 6, 9, 7, 10, 6, 11, 7, 12, 8, 11, 8, 9, 5, 12,
                                          9, 10, 10, 11, 11, 12, 12, 9, 9, 11, 10, 12, 9, 13, 10, 14, 11, 15, 12, 16,
                                          9, 14, 10, 13, 11, 14, 10, 15, 11, 16, 12, 15, 12, 13, 9, 16, 13, 14, 14, 15,
                                          15, 16, 16, 13, 13, 15, 14, 16, 13, 17, 14, 18, 15, 19, 16, 20, 13, 18, 14, 17,
                                          15, 18, 14, 19, 15, 20, 16, 19, 16, 17, 13, 20, 17, 18, 18, 19, 19, 20, 20, 17,
                                          17, 19, 18, 20 };

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
    
    /****************************************************************************************************************/
    const int E = 10;
    const int P = 5;
    const float minDisp = -0.25, maxDisp = 0.25; //in
    const float minStress = -25, maxStress = 25;
    const float rho = 0.1;
    float sum = 0.0;
    int x[2], y[2], z[2];
    int index[6];
    float l_ij, m_ij, n_ij, le;
    float Ae[NUM_OF_ELEMENTS];
    const int bcDOF[12] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    float bcValue[12] = {0};
    float stress_e[NUM_OF_ELEMENTS] = {0};
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
    float invK_array[MAX_ROWS*MAX_COLS];
    float U_array[MAX_ROWS];
    float de_o_array[MAX_ROWS];
    float productOfBe_de_array[MAX_ROWS];
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
        y[0] = gCoord[20 + element[i * 2] - 1];
        y[1] = gCoord[20 + element[i *2 + 1] - 1];
        z[0] = gCoord[40 + element[i * 2] - 1];
        z[1] = gCoord[40 + element[i * 2 + 1] - 1];

        le = sqrt( pow((x[1] - x[0]), 2) + pow((y[1] - y[0]), 2) +  pow((z[1] - z[0]), 2));


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

    F.pMatrix[50 * F.cols] = -P;
    F.pMatrix[53 * F.cols] = -P;
    F.pMatrix[56 * F.cols] = -P;
    F.pMatrix[59 * F.cols] = -P;

    for (int bc_i = 0; bc_i < 12; bc_i++)
    {
        int temp = bcDOF[bc_i];
        for (int zeros_i = 0; zeros_i < TOTAL_DOF; zeros_i++)
            K.pMatrix[temp * K.cols + zeros_i] = 0;

        K.pMatrix[temp * K.cols + temp] = 1;
        F.pMatrix[temp * F.cols] = bcValue[bc_i];
    } // F - OK
    //Calculate U = K\F. inv(K)*F
    LU_getInverseMatrix1D(&K, invK_array, &invK);
    multiplyMatrices1D(&invK, &F, U_array, &U);
    //printMatrix(&U);

    /* Compute stress for each element */
    for (int i = 0; i < NUM_OF_ELEMENTS; i++)
    {
        x[0] = gCoord[element[i * 2] - 1];
        x[1] = gCoord[element[i * 2 + 1] - 1];
        y[0] = gCoord[20 + element[i * 2] - 1];
        y[1] = gCoord[20 + element[i *2 + 1] - 1];
        z[0] = gCoord[40 + element[i * 2] - 1];
        z[1] = gCoord[40 + element[i * 2 + 1] - 1];

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
    }
    /*********************** Check constraints violation ******************************/
    float Cdisp[NUM_OF_NODES*2], Cstress[NUM_OF_ELEMENTS];
    float sumOfCdisp = 0, sumOfCtress = 0;

    //Displacement constraints
    
    for (int i = 0; i < NUM_OF_NODES*3; i++)
    {
        if ((minDisp <= U.pMatrix[i * U.cols]) && (U.pMatrix[i * U.cols] <= maxDisp))
        {
            Cdisp[i] = 0;
        }
        else
        {
            Cdisp[i] = fabs(((U.pMatrix[i * U.cols] - maxDisp)/maxDisp));
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
    float sum1 = 0.0;
    float sum2 = 0.0;
    float sum3 = 0.0;
    float sum4 = 0.0;
    for (int i = 0; i < 4; i++)
    {
        sum1 = sum1 + (Ae[i] * rho * 60.0);
        sum2 = sum2 + (Ae[18+i] * rho * 60.0);
        sum3 = sum3 + (Ae[36+i] * rho * 60.0);
        sum4 = sum4 + (Ae[54+i] * rho * 60.0);
    }

    for (int i = 0; i < 8; i++)
    {
        sum1 = sum1 + (Ae[4+i] * rho * 134.164);
        sum2 = sum2 + (Ae[22+i] * rho * 134.164);
        sum3 = sum3 + (Ae[40+i] * rho * 134.164);
        sum4 = sum4 + (Ae[58+i] * rho * 134.164);
    }

    for (int i = 0; i < 4; i++)
    {
        sum1 = sum1 + (Ae[12+i] * rho * 120.0);
        sum2 = sum2 + (Ae[30+i] * rho * 120.0);
        sum3 = sum3 + (Ae[48+i] * rho * 120.0);
        sum4 = sum4 + (Ae[66+i] * rho * 120.0);
    }

    for (int i = 0; i < 2; i++)
    {
        sum1 = sum1 + (Ae[16+i] * rho * 169.7056);
        sum2 = sum2 + (Ae[34+i] * rho * 169.7056);
        sum3 = sum3 + (Ae[52+i] * rho * 169.7056);
        sum4 = sum4 + (Ae[70+i] * rho * 169.7056);
    }
    sum = sum1 +sum2+sum3+sum4;

    return (sum * pow((sumOfCtress + sumOfCdisp + 1), 1));
}