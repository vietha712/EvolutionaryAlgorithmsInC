#include "planar_truss_200bars.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "matrix_improved.cuh"

#define NUM_OF_ELEMENTS 200
#define NUM_OF_NODES 77
#define DOF 2
#define TOTAL_DOF 154

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
#define BCDOF 4
#define LOAD_CASE_2
__host__ __device__ float functional(const float * __restrict A, const int D, float * d_invK, float * d_localLU, float * d_s)
{
    float sum = 0.0f;
    int x[2], y[2];
    int index[4];
    float l_ij, m_ij, le;
    float Ae[NUM_OF_ELEMENTS];
    const int bcDOF[BCDOF] = {150, 151, 152, 153};
    float bcValue[BCDOF] = {0, 0, 0, 0};
    Matrix1DT K, F, Te, Te_Transpose, U, invK, Be, disp_e, de_o, temp;
    Matrix1DT matrix2x2_Precomputed, ke2x2, output4x2, output4x4, productOfBe_de;
    const float minStress = -10, maxStress = 10;
    const float rho = 0.283; //lb/in3
    const float E = 30000;
    //const float Px = 1.0; 
    const float Py = (-10.0); 

    static const int gCoord1D[2 * NUM_OF_NODES] = { 0   , 240 , 480 , 720 , 960 , 0   , 120 , 240 , 360 , 480 , 600 , 720 , 840 , 960 , 0   , 240 , 480 , 720 , 960 , 0   , 120 , 240 , 360 , 480 , 600 , 720 , 840 , 960 , 0   , 240 , 480 , 720 , 960 , 0   , 120 , 240 , 360 , 480 , 600 , 720 , 840 , 960 , 0  , 240, 480, 720, 960, 0  , 120, 240, 360, 480, 600, 720, 840, 960, 0  , 240, 480, 720, 960, 0  , 120, 240, 360, 480, 600, 720, 840, 960, 0  , 240, 480, 720, 960, 240, 720,
                                       1800, 1800, 1800, 1800, 1800, 1656, 1656, 1656, 1656, 1656, 1656, 1656, 1656, 1656, 1512, 1512, 1512, 1512, 1512, 1368, 1368, 1368, 1368, 1368, 1368, 1368, 1368, 1368, 1224, 1224, 1224, 1224, 1224, 1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080, 936, 936, 936, 936, 936, 792, 792, 792, 792, 792, 792, 792, 792, 792, 648, 648, 648, 648, 648, 504, 504, 504, 504, 504, 504, 504, 504, 504, 360, 360, 360, 360, 360, 0  , 0   };

    static const int element1D[NUM_OF_ELEMENTS * 2] = 
    {   1, 2, 2, 3, 3, 4, 4, 5, 1, 6, 1, 7, 2, 7, 2, 8, 2, 9, 3, 9, 3, 10, 3, 11, 
        4, 11, 4, 12, 4, 13, 5, 13, 5, 14 /*E17*/, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, //E25
        6, 15, 7, 15, 7, 16, 8, 16, 9, 16, 9, 17, 10, 17, 11, 17, 11, 18, 12, 18, 13, 18, 13, 19, 14, 19, //E38
        15, 16, 16, 17, 17, 18, 18, 19, 15, 20, 15, 21, 16, 21, 16, 22, 16, 23, 17, 23, 17, 24, 17, 25, 18, 25, 18, 26, 18, 27, 19, 27, 19, 28, //E55
        20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28 /*E63*/, 20, 29, 21, 29, 21, 30, 22, 30, 23, 30, 23, 31, 24, 31, 25, 31,
        25, 32, 26, 32, 27, 32, 27, 33, 28, 33 /*E76*/, 29, 30, 30, 31, 31, 32, 32, 33 /*E80*/, 29, 34, 29, 35, 30, 35, 30, 36, 30, 37, 31, 37, 31, 38, 31, 39,
        32, 39, 32, 40, 32, 41, 33, 41, 33, 42 /*93*/, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41, 42 /*101*/, 34, 43, 35, 43,
        35, 44, 36, 44, 37, 44, 37, 45, 38, 45, 39, 45, 39, 46, 40, 46, 41, 46, 41, 47, 42, 47 /*114*/, 43, 44, 44, 45, 45, 46, 46, 47, //118
        43, 48, 43, 49, 44, 49, 44, 50, 44, 51, 45, 51, 45, 52, 45, 53, 46, 53, 46, 54, 46, 55, 47, 55, 47, 56 /* 131 */,
        48, 49, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 55, 56/*139*/, 48, 57, 49, 57, 49, 58, 50, 58, 51, 58, 51, 59, 52, 59, 53, 59,
        53, 60, 54, 60, 55, 60, 55, 61, 56, 61 /*152*/, 57, 58, 58, 59, 59, 60, 60, 61 /*156*/, 57, 62, 57, 63, 58, 63, 58, 64, 58, 65,
        59, 65, 59, 66, 59, 67, 60, 67, 60, 68, 60, 69, 61, 69, 61, 70 /*169*/, 62, 63, 63, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68, 69, 69, 70, //177
        62, 71, 63, 71, 63, 72, 64, 72, 65, 72, 65, 73, 66, 73, 67, 73, 67, 74, 68, 74, 69, 74, 69, 75, 70, 75 /*190*/, 71, 72, 72, 73, 73, 74, 74, 75,
        71, 76, 72, 76, 73, 76, 73, 77, 74, 77, 75, 77 };

    float stress_e[NUM_OF_ELEMENTS] = {0};

    static const int index_A1[4] = {1, 2, 3, 4};
    static const int index_A2[5] = {5, 8, 11, 14, 17};
    static const int index_A3[6] = {19, 20, 21, 22, 23, 24};
    static const int index_A4[10] = {18, 25, 56, 63, 94, 101, 132, 139, 170, 177};
    static const int index_A5[5] = {26, 29, 32, 35, 38};
    static const int index_A6[16] = {6, 7, 9, 10, 12, 13, 15, 16, 27, 28, 30, 31, 33, 34, 36, 37};
    static const int index_A7[4] = {39, 40, 41, 42};
    static const int index_A8[5] = {43, 46, 49, 52, 55};
    static const int index_A9[6] = {57, 58, 59, 60, 61, 62};
    static const int index_A10[5] = {64, 67, 70, 73, 76};
    static const int index_A11[16] = {44, 45, 47, 48, 50, 51, 53, 54, 65, 66, 68, 69, 71, 72, 74, 75};
    static const int index_A12[4] = {77, 78, 79, 80};
    static const int index_A13[5] = {81, 84, 87, 90, 93};
    static const int index_A14[6] = {95, 96, 97, 98, 99, 100};
    static const int index_A15[5] = {102, 105, 108, 111, 114};
    static const int index_A16[16] = {82, 83, 85, 86, 88, 89, 91, 92, 103, 104, 106, 107, 109, 110, 112, 113};
    static const int index_A17[4] = {115, 116, 117, 118};
    static const int index_A18[5] = {119, 122, 125, 128, 131};
    static const int index_A19[6] = {133, 134, 135, 136, 137, 138};
    static const int index_A20[5] = {140, 143, 146, 149, 152};
    static const int index_A21[16] = {120, 121, 123, 124, 126, 127, 129, 130, 141, 142, 144, 145, 147, 148, 150, 151};
    static const int index_A22[4] = {153, 154, 155, 156};
    static const int index_A23[5] = {157, 160, 163, 166, 169};
    static const int index_A24[6] = {171, 172, 173, 174, 175, 176};
    static const int index_A25[5] = {178, 181, 184, 187, 190};
    static const int index_A26[16] = {158, 159, 161, 162, 164, 165, 167, 168, 179, 180, 182, 183, 185, 186, 188, 189};
    static const int index_A27[4] = {191, 192, 193, 194};
    static const int index_A28[4] = {195, 197, 198, 200};
    static const int index_A29[2] = {196, 199};

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
    float U_array[TOTAL_DOF];
    float de_o_array[500];
    float productOfBe_de_array[500];
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

    /* Compute stiffness matrix */
    for (int i = 0; i < NUM_OF_ELEMENTS; i++)
    {
        x[0] = gCoord1D[element1D[i * 2] - 1];
        x[1] = gCoord1D[element1D[i * 2 + 1] - 1];
        y[0] = gCoord1D[77 + element1D[i * 2] - 1];
        y[1] = gCoord1D[77 + element1D[i * 2 + 1] - 1];

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

    // Applying force
#ifdef LOAD_CASE_1
    //Load case 1: x - direction
    F.pMatrix[0] = Px; //Node 1
    F.pMatrix[10] = Px; //Node 6
    F.pMatrix[28] = Px; //Node 15
    F.pMatrix[38] = Px; //Node 20
    F.pMatrix[56] = Px; //Node 29
    F.pMatrix[66] = Px; //Node 34
    F.pMatrix[84] = Px; //Node 43
    F.pMatrix[94] = Px; //Node 48
    F.pMatrix[112] = Px; //Node 57
    F.pMatrix[122] = Px; //Node 62
#endif
    //Load case 2: y - direction: 55 nodes being applied
#ifdef LOAD_CASE_2
    F.pMatrix[1] = Py; //Node 1
    F.pMatrix[3] = Py; //Node 2
    F.pMatrix[5] = Py; //Node 3
    F.pMatrix[7] = Py; //Node 4
    F.pMatrix[9] = Py; //Node 5
    F.pMatrix[11] = Py; //Node 6
    F.pMatrix[15] = Py; //Node 8
    F.pMatrix[19] = Py; //Node 10
    F.pMatrix[23] = Py; //Node 12
    F.pMatrix[27] = Py; //Node 14
    F.pMatrix[29] = Py; //Node 15
    F.pMatrix[31] = Py; //Node 16
    F.pMatrix[33] = Py; //Node 17
    F.pMatrix[35] = Py; //Node 18
    F.pMatrix[37] = Py; //Node 19
    F.pMatrix[39] = Py; //Node 20
    F.pMatrix[43] = Py; //Node 22
    F.pMatrix[47] = Py; //Node 24
    F.pMatrix[51] = Py; //Node 26
    F.pMatrix[55] = Py; //Node 28
    F.pMatrix[57] = Py; //Node 29
    F.pMatrix[59] = Py; //Node 30
    F.pMatrix[61] = Py; //Node 31
    F.pMatrix[63] = Py; //Node 32
    F.pMatrix[65] = Py; //Node 33
    F.pMatrix[67] = Py; //Node 34
    F.pMatrix[71] = Py; //Node 36
    F.pMatrix[75] = Py; //Node 38
    F.pMatrix[79] = Py; //Node 40
    F.pMatrix[83] = Py; //Node 42
    F.pMatrix[85] = Py; //Node 43
    F.pMatrix[87] = Py; //Node 44
    F.pMatrix[89] = Py; //Node 45
    F.pMatrix[91] = Py; //Node 46
    F.pMatrix[93] = Py; //Node 47
    F.pMatrix[95] = Py; //Node 48
    F.pMatrix[99] = Py; //Node 50
    F.pMatrix[103] = Py; //Node 52
    F.pMatrix[107] = Py; //Node 54
    F.pMatrix[111] = Py; //Node 56
    F.pMatrix[113] = Py; //Node 57
    F.pMatrix[115] = Py; //Node 58
    F.pMatrix[117] = Py; //Node 59
    F.pMatrix[119] = Py; //Node 60
    F.pMatrix[121] = Py; //Node 61
    F.pMatrix[123] = Py; //Node 62
    F.pMatrix[127] = Py; //Node 64
    F.pMatrix[131] = Py; //Node 66
    F.pMatrix[135] = Py; //Node 68
    F.pMatrix[139] = Py; //Node 70
    F.pMatrix[141] = Py; //Node 71
    F.pMatrix[143] = Py; //Node 72
    F.pMatrix[145] = Py; //Node 73
    F.pMatrix[147] = Py; //Node 74
    F.pMatrix[149] = Py; //Node 75
#endif //LOAD_CASE_2

    for (int bc_i = 0; bc_i < BCDOF; bc_i++)
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
        y[0] = gCoord1D[77 + element1D[i * 2] - 1];
        y[1] = gCoord1D[77 + element1D[i * 2 + 1] - 1];

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
    float Cstress[NUM_OF_ELEMENTS];
    float sumOfCtress = 0;
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
    float sum1 = 0.0;
    float sum2 = 0.0;
    float sum3 = 0.0;
    float sum4 = 0.0;
    float sum5 = 0.0;
    float sum6 = 0.0;
    float sum7 = 0.0;

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

    return (sum * pow((sumOfCtress + 1), 1));
}