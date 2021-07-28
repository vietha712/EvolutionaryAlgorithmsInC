#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double
        Xl[12] = {-2.048,-2.048,-2.048,-2.048,-2.048,-2.048,-2.048,-2.048,-2.048,-2.048,
		-2.048,-2.048},
        Xu[12] = {2.048,2.048,2.048,2.048,2.048,2.048,2.048,2.048,2.048,2.048,
		2.048,2.048};

int D = 12;

double func(double *x)
{
    register int i;
	double sum = 0.f;
    extern int D;
	for (i = 1; i < D; i++)
    {
        sum = sum + 100.f * (x[i] - x[i - 1] * x[i - 1]) * (x[i] - x[i - 1] * x[i - 1]) + (x[i - 1] - 1.f) * (x[i - 1] - 1.f);
    }

    return sum;
}