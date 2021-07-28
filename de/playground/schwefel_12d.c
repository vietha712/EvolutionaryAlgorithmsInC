/* Normalized Schwefel's function, 20D	*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* Dimension of problem and initialization boundaries for variables
   Xl defines lower limit
   Xu defines upper limit       */

int D = 12;

double
	Xl[12] = {-512.03,-512.03,-512.03,-512.03,-512.03,-512.03,-512.03,-512.03,
                -512.03,-512.03,-512.03,-512.03},
        Xu[12] = {511.97,511.97,511.97,511.97,511.97,511.97,511.97,511.97,511.97,
                511.97,511.97,511.97};


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


	sum = 418.9829 * D;
	for ( i = 1; i <= D; i++) sum = sum + X[i - 1] * sin(sqrt(fabs(X[i - 1])));
	
	return sum;
}

