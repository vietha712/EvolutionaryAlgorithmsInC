#include "planar_truss_10bars.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "matrix.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_OF_ELEMENTS 10
#define NUM_OF_NODES 6
#define DOF 2
#define TOTAL_DOF 12 // DOF * NUM_OF_NODES 

__host__ __device__ static void getTransposeOfTe1D(Matrix1DT* inputMat, Matrix1DT* outputMat);

__host__ __device__ static inline float getWeight(float A, int length)
{
    return (A * 0.1 * length); //rho = 0.1
}

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
    float sum = 0.0;
    float le;
    int x[2], y[2];
    float l_ij, m_ij;
    Matrix1DT Te, Te_Transpose, invK, F, K, temp;
    Matrix1DT ke2x2, Be, U, disp_e, de_o, productOfBe_de;
    Matrix1DT matrix2x2_Precomputed, output4x2, output4x4; //line 57 in 10 bars
    int index[4];
    int bcDOF[4] = {8, 9, 10, 11}; //reindex in C. Original 9 - 10 - 11 - 12
    float bcValue[4] = {0};
    float stress_e[NUM_OF_ELEMENTS] = {0};

    static const int E = 10000000; //N/m2
    static const int P = 100000;
    static const float minDisp = -2.0, maxDisp = 2.0;
    static const float minStress = -25000, maxStress = 25000;
    static const int element1D[NUM_OF_ELEMENTS * 2] = { 3, 5, 1, 3, 4, 6, 2, 4, 3, 4, 1, 2, 4, 5, 3, 6, 2, 3, 1, 4 };
    static const int gCoord1D[2 * 6] = {720, 720, 360, 360, 0, 0, 360, 0, 360, 0, 360, 0};

    //allocateMatrix1D(&Te, 2, 4);
    Te.rows = 2; Te.cols = 4; Te.isInit = 1;
    //allocateMatrix1D(&Te_Transpose, 4, 2);
    Te_Transpose.rows = 4; Te_Transpose.cols = 2; Te_Transpose.isInit = 1;
    //allocateMatrix1D(&ke2x2, 2, 2);
    ke2x2.rows = 2; ke2x2.cols = 2; ke2x2.isInit = 1;
    //allocateMatrix1D(&matrix2x2_Precomputed, 2, 2);
    matrix2x2_Precomputed.rows = 2; matrix2x2_Precomputed.cols = 2; matrix2x2_Precomputed.isInit = 1;
    //allocateMatrix1D(&F, TOTAL_DOF, 1); //12x1
    F.rows = 12; F.cols = 1; F.isInit = 1;
    //allocateMatrix1D(&Be, 1, 2);
    Be.rows = 1; Be.cols = 2; Be.isInit = 1;
    //allocateMatrix1D(&disp_e, 4, 1);
    disp_e.rows = 4; disp_e.cols = 1; disp_e.isInit = 1;
    //allocateMatrix1D(&K, TOTAL_DOF, TOTAL_DOF);
    K.rows = K.cols = TOTAL_DOF; K.isInit = 1;
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

    /*********************** Check constraints violation ******************************/
    float Cdisp[12], Cstress[10];
    float sumOfCdisp = 0, sumOfCtress = 0;

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

    // calculate total weight
    for (int i = 0; i < (D-4); i++)
    {
        sum += getWeight(A[i], 360);
    }
    for (int i = (D-4) ; i < 10; i++)
    {
        sum += getWeight(A[i], 509.12);
    }

    /* Deallocate */
    //deallocateMatrix1D(&temp);
    //deallocateMatrix1D(&Te);
    //deallocateMatrix1D(&Te_Transpose);
    //deallocateMatrix1D(&invK);
    //deallocateMatrix1D(&F);
    //deallocateMatrix1D(&ke2x2);
    //deallocateMatrix1D(&Be);
    //deallocateMatrix1D(&U);
    //deallocateMatrix1D(&disp_e);
    //deallocateMatrix1D(&de_o);
    //deallocateMatrix1D(&productOfBe_de);
    //deallocateMatrix1D(&matrix2x2_Precomputed);
    //deallocateMatrix1D(&output4x2);
    //deallocateMatrix1D(&output4x4);
    //printf("%f\n", (sum * pow((sumOfCtress + sumOfCdisp + 1), 1)));
    return (sum * pow((sumOfCtress + sumOfCdisp + 1), 1));
}