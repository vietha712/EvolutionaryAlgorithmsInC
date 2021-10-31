#include "planar_truss_52bars.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "matrix_improved.cuh"

#define NUM_OF_ELEMENTS 52
#define NUM_OF_NODES 20
#define DOF 2

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

__host__ __device__ static void convertMilliMeterToMeter(const float *A, float *meterA, const int D)
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
__host__ __device__ float functional(const float * __restrict A, const int D, float * d_invK, float * d_localLU, float * d_s)
{
    float sum = 0.0f;
    int x[2], y[2];
    int index[4];
    float l_ij, m_ij, le;
    float Ae[NUM_OF_ELEMENTS];
    float meterSquare_A[12];
    const int bcDOF[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    float bcValue[8] = {0};
    Matrix1DT K, F, Te, Te_Transpose, U, invK, Be, disp_e, de_o, temp;
    Matrix1DT matrix2x2_Precomputed, ke2x2, output4x2, output4x4, productOfBe_de;

    const float minStress = -180000000, maxStress = 180000000;
    const float rho = 7860; //kg/m3
    //const float rho = 0.284; //lb/in3
    const float E = 207000000000.000; // N/m2
    const float Px = 100000; //N
    const float Py = 200000; //N

    static const int gCoord1D[2 * 20] = { 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6,
                                      0, 0, 0, 0, 3, 3, 3, 3, 6, 6, 6, 6, 9, 9, 9, 9, 12, 12, 12, 12 };

    static const int element1D[NUM_OF_ELEMENTS * 2] = 
    { 1, 5, 2, 6, 3, 7, 4, 8, 2, 5, 1, 6, 3, 6, 2, 7, 4, 7, 3, 8, 5, 6, 6, 7, 
      7, 8, 5, 9, 6, 10, 7, 11, 8, 12, 6, 9, 5, 10, 7, 10, 6, 11, 8, 11, 7, 12, 9, 10,
      10, 11, 11, 12, 9, 13, 10, 14, 11, 15, 12, 16, 10, 13, 9, 14, 11, 14, 10, 15, 12, 15, 11, 16,
      13, 14, 14, 15, 15, 16, 13, 17, 14, 18, 15, 19, 16, 20, 14, 17, 13, 18, 15, 18, 14, 19, 16, 19,
      15, 20, 17, 18, 18, 19, 19, 20};

    float stress_e[NUM_OF_ELEMENTS] = {0};

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

    /*************************************************************************************************/
    float K_array[TOTAL_DOF*TOTAL_DOF];
    allocateMatrix1D(&K, K_array, TOTAL_DOF, TOTAL_DOF);

    float F_array[TOTAL_DOF];
    allocateMatrix1D(&F, F_array, TOTAL_DOF, 1);

    float Te_array[8]; //rows * cols
    allocateMatrix1D(&Te, Te_array, 2, 4);

    float Te_Transpose_array[8]; //rows * cols
    allocateMatrix1D(&Te_Transpose, Te_Transpose_array, 4, 2);

    float ke2x2_array[4]; //rows * cols
    allocateMatrix1D(&ke2x2, ke2x2_array, 2, 2);

    float matrix2x2_Precomputed_array[4];
    allocateMatrix1D(&matrix2x2_Precomputed, matrix2x2_Precomputed_array, 2, 2);

    float Be_array[2];
    allocateMatrix1D(&Be, Be_array, 1, 2);

    float disp_e_array[4];
    allocateMatrix1D(&disp_e, disp_e_array, 4, 1);
    /*************************************************************************************************/
    float output4x2_array[8];
    float output4x4_array[16];
    //float invK_array[MAX_ROWS*MAX_COLS];
    float U_array[MAX_ROWS];
    float de_o_array[MAX_ROWS];
    float productOfBe_de_array[MAX_ROWS];
    initMatrix(&output4x2);
    initMatrix(&output4x4);
    initMatrix(&invK);
    initMatrix(&U);
    initMatrix(&de_o);
    initMatrix(&productOfBe_de);

    matrix2x2_Precomputed.pMatrix[0] = 1;
    matrix2x2_Precomputed.pMatrix[1] = -1;
    matrix2x2_Precomputed.pMatrix[2] = -1;
    matrix2x2_Precomputed.pMatrix[3] = 1;
    zerosMatrix1D(&F);
    zerosMatrix1D(&K);

    /* Convert the unit for A */
    convertMilliMeterToMeter(A, meterSquare_A, D); //Pass

    /* Get A for each element */ //Pass
    for (int i = 0; i < NUM_OF_ELEMENTS; i++)
    {
        Ae[i] = meterSquare_A[indexA[i] - 1];
    }

    /* Compute stiffness matrix */
    for (int i = 0; i < NUM_OF_ELEMENTS; i++)
    {
        x[0] = gCoord1D[element1D[i * 2] - 1];
        x[1] = gCoord1D[element1D[i * 2 + 1] - 1];
        y[0] = gCoord1D[20 + element1D[i * 2] - 1];
        y[1] = gCoord1D[20 + element1D[i * 2 + 1] - 1];

        le = sqrt( pow((x[1] - x[0]), 2) + pow((y[1] - y[0]), 2) );

        //Compute direction cosin
        l_ij = (x[1] - x[0])/le;
        m_ij = (y[1] - y[0])/le;

        //Compute transform matrix
        Te.pMatrix[0] = l_ij; Te.pMatrix[1] = m_ij; Te.pMatrix[2] = 0; Te.pMatrix[3] = 0;
        Te.pMatrix[4] = 0; Te.pMatrix[5] = 0; Te.pMatrix[6] = l_ij; Te.pMatrix[7] = m_ij;

        // Compute stiffness martix of element line 56
      
        multiplyScalarMatrix1D((Ae[i]*E/le), &matrix2x2_Precomputed, ke2x2_array, &ke2x2);
        getTransposeOfTe1D(&Te, &Te_Transpose);
        multiplyMatrices1D(&Te_Transpose, &ke2x2, output4x2_array, &output4x2); //line 59
        multiplyMatrices1D(&output4x2, &Te, output4x4_array, &output4x4);

        //Find index assemble
        index[0] = 2*element1D[i * 2] - 1 - 1;
        index[1] = 2*element1D[i * 2] - 1;
        index[2] = 2*element1D[i * 2 + 1] - 1 - 1;
        index[3] = 2*element1D[i * 2 + 1] - 1;

        for (int row_i = 0; row_i < 4; row_i++)
        {
            for (int col_i = 0; col_i < 4; col_i++)
                K.pMatrix[index[row_i] * K.cols + index[col_i]] =  K.pMatrix[index[row_i] * K.cols + index[col_i]] + output4x4.pMatrix[row_i * output4x4.cols + col_i];
        }
    }

    F.pMatrix[32 * F.cols] = Px;
    F.pMatrix[33 * F.cols] = Py;
    F.pMatrix[34 * F.cols] = Px;
    F.pMatrix[35 * F.cols] = Py;
    F.pMatrix[36 * F.cols] = Px;
    F.pMatrix[37 * F.cols] = Py; 
    F.pMatrix[38 * F.cols] = Px;
    F.pMatrix[39 * F.cols] = Py;

    for (int bc_i = 0; bc_i < 8; bc_i++)
    {
        int temp = bcDOF[bc_i];
        for (int zeros_i = 0; zeros_i < TOTAL_DOF; zeros_i++)
            K.pMatrix[temp * K.cols + zeros_i] = 0;

        K.pMatrix[temp * K.cols + temp] = 1;
        F.pMatrix[temp * F.cols] = bcValue[bc_i];
    }

    //Calculate U = K\F. inv(K)*F
    LU_getInverseMatrix1D(&K, d_invK, &invK, d_localLU, d_s);
    multiplyMatrices1D(&invK, &F, U_array, &U); //U is nodal displacement of each element //Pass U

    /* Compute stress for each element */
    for (int i = 0; i < NUM_OF_ELEMENTS; i++)
    {
        x[0] = gCoord1D[element1D[i * 2] - 1];
        x[1] = gCoord1D[element1D[i * 2 + 1] - 1];
        y[0] = gCoord1D[20 + element1D[i * 2] - 1];
        y[1] = gCoord1D[20 + element1D[i * 2 + 1] - 1];

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
#if 0
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
#endif
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
    float sum1 = 0.0;
    float sum2 = 0.0;
    float sum3 = 0.0;
    float sum4 = 0.0;
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

    return (sum * pow((sumOfCtress + sumOfCdisp + 1), 1));
}