#include "matrix.h"

double func(double *);
void fix(double *X, int length);

double preCpted_A[10] = {30, 1.62, 22.9, 13.5, 1.62, 1.62, 7.97, 26.5, 22, 1.8};

double preCpted_B[12] = {4658.055, 1161.288, 494.193, 3303.219, 939.998, 494.193, 2238.705, 1008.385, 494.193, 1283.868, 1161.288, 494.193};

double testVector_2[16] =  {1.990,
                            0.563,
                            0.111,
                            0.111,
                            1.228,
                            0.442,
                            0.111,
                            0.111,
                            0.563,
                            0.563,
                            0.111,
                            0.111,
                            0.196,
                            0.563,
                            0.391,
                            0.563};

double testVector_3[16] = {0.39, 0.14, 0.11, 0.11, 0.39, 0.14, 0.11, 0.11, 0.31, 0.11, 0.11, 0.14, 0.20, 0.11, 0.14, 0.14};

double testVector_4[16] = {26.5, 8.53, 28, 7.22, 11.5, 9.3, 0.307, 18.8, 24.5, 0.442, 9.3, 4.97, 22.9, 2.38, 4.49, 7.97};

double test_vector[10] = {1.62, 27, 35, 14.6, 0, 3, -4, 7.7, 32, 1.8};

int main(void)
{
    //fix(test_vector, 10);

    //for(int i = 0; i < 10; i++)
    //    printf(" i[%d] = %f\n", i, test_vector[i]);
    printf("obj value 72 bars: %.20f\n", func(testVector_4));
    //double sum = 0;
    //for (int i = 0; i < 10; i++)
    //    sum += 0.1 * 360 * testVector_2[i];
    //printf("sum: %f\n", sum);
    return 0;
}