#include "matrix.h"

double func(double *);
void fix(double *X, int length);

double preCpted_A[10] = {30, 1.62, 22.9, 13.5, 1.62, 1.62, 7.97, 26.5, 22, 1.8};

double testVector_2[10] =  {33.5,
    1.62, 
    22.9,
    14.2,
    1.62,
    1.62,
    7.97,
    22.9,
    22,
    1.62};

double test_vector[10] = {1.62, 27, 35, 14.6, 0, 3, -4, 7.7, 32, 1.8};

int main(void)
{
    //fix(test_vector, 10);

    //for(int i = 0; i < 10; i++)
    //    printf(" i[%d] = %f\n", i, test_vector[i]);
    printf("obj value: %.14f\n", func(testVector_2));
    double sum = 0;
    for (int i = 0; i < 10; i++)
        sum += 0.1 * 360 * testVector_2[i];
    printf("sum: %f\n", sum);
    return 0;
}