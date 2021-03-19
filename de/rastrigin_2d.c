/* Rastrigin's function, 2D	*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define M_PI 3.14159265358979323846

/* Dimension of problem and initialization boundaries for variables
   Xl defines lower limit
   Xu defines upper limit       */

int D = 2;

double
        Xl[2] = {-5.12,-5.12},
        Xu[2] = {5.12,5.12};

double func(double *X)
{
   register int i;
   double sum, A = 10;
   extern int D;

/* Correction of boundary constraint violations, violating variable
   values are reflected back from the violated boundary        */

   for (i=0; i<D; i++)
   {
      while (X[i] < Xl[i] || X[i] > Xu[i])
      {
         if (X[i] < Xl[i]) X[i] = 2.0*Xl[i] - X[i];
         if (X[i] > Xu[i]) X[i] = 2.0*Xu[i] - X[i];
      }
   }


   sum = 0.0;
   for (i=0; i<D; i++)
   {
      sum += X[i]*X[i] - A*cos(2.0*M_PI*X[i]);
   }

   return (double)D*A + sum;
}

