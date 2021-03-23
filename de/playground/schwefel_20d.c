/* Normalized Schwefel's function, 20D	*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* Dimension of problem and initialization boundaries for variables
   Xl defines lower limit
   Xu defines upper limit       */

int D = 20;

double
	Xl[20] = {-500.0,-500.0,-500.0,-500.0,-500.0,-500.0,-500.0,-500.0,
                -500.0,-500.0,-500.0,-500.0,-500.0,-500.0,-500.0,-500.0,
                -500.0,-500.0,-500.0,-500.0},
        Xu[20] = {500.0,500.0,500.0,500.0,500.0,500.0,500.0,500.0,500.0,
                500.0,500.0,500.0,500.0,500.0,500.0,500.0,500.0,500.0,
                500.0,500.0};


double func(double *X)
{
   register int i;
   double sum;
   extern int D;
   extern double Xl[], Xu[];


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
      sum += X[i]*sin(sqrt(fabs(X[i])));
   }

   return 418.9828872724337998*(double)D - sum;
}

