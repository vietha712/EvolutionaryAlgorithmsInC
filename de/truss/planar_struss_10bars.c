#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// ????? Do we need to perform finite element analysis of truss

const double A[42] = {1.62, 1.80, 1.99, 2.13, 2.38, 2.62, 2.63, 2.88, 2.93, 3.09, 3.13, 3.38,
                      3.47, 3.55, 3.63, 3.84, 3.87, 3.88, 4.18, 4.22, 4.49, 4.59, 4.80, 4.97,
                      5.12, 5.74, 7.22, 7.97, 11.50, 13.50, 13.90, 14.20, 15.50, 16.00, 16.90,
                      18.80, 19.90, 22.00, 22.90, 26.50, 30.00, 33.50}; //Standard cross-sectional areas for design variable


double Xl[10] = {1.62,1.62,1.62,1.62,1.62,1.62,1.62,1.62,1.62,1.62},
       Xu[10] = {33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50,33.50};


inline double getWeight(double A, double matDen, double len)
{
    return (A * matDen * len);
}

void fix(double *X, int len)
{
    for (int i = 0; i < len; i++)
    {
        for (int i = 0; i < 42; i++)
        {
            if (X[i] <= A[i])
            {
                X[i] = A[i];
            }
        }
    }
}

const double epsilon_1 = 1.0;
double epsilon_2 = 20.0;
int youngModulus = 100000;
double materialDensity = 0.1;

int D = 10;

double func(double *X)
{
    extern int D;
    double sum;



    return
}