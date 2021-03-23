/* Rastrigin's function, 20 D, skewed initialization	*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define M_PI 3.14159265358979323846

/* Dimension of problem and initialization boundaries for variables
   Xl defines lower limit
   Xu defines upper limit       */

int D = 20;

double
        Xl[20] = {-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,
		-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0},
        Xu[20] = {-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,
		-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0};


double func(double *X)
{
   register int i;
   double sum, A = 10;
   extern int D;


   sum = 0.0;
   for (i=0; i<D; i++)
   {
      sum += X[i]*X[i] - A*cos(2.0*M_PI*X[i]);
   }

   return ((double)D*A + sum);
}

