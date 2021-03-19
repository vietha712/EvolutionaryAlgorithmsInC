#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <float.h>

/* Function definitions		*/

double func(double *);
int usage(char *);

/* Random number generator defined by URAND should return
   double-precision floating-point values uniformly distributed
   over the interval [0.0, 1.0)					*/

#define URAND	((double)rand()/((double)RAND_MAX + 1.0))

/* Definition for random number generator initialization	*/

#define INITRAND srand(time(0))

/* Usage for the program	*/

int usage(char *str)
{
   fprintf(stderr,"Usage: %s [-h] [-u] [-s] [-N NP (20*D)] ", str);
   fprintf(stderr,"[-G Gmax (1000)]\n");
   fprintf(stderr,"\t[-C crossover constant, CR (0.9)]\n");
   fprintf(stderr,"\t[-F mutation scaling factor, F (0.9)]\n");
   fprintf(stderr,"\t[-o <outputfile>]\n\n");
   fprintf(stderr,"\t-s does not initialize random number generator\n");
   exit(-1);
}


int main(int argc, char **argv)
{
   register int i, j, k, r1, r2, best, jrand, numOfFuncEvals = 0;
   extern int D;
   extern double Xl[], Xu[];
   int NP = 20*D, Gmax = 1000, c, index = -1, s = 1;
   double **pPop, **pNext, **ptr, *iptr, *U;
   double CR = 0.9, Fxc = 0.9, Fcr = 0.6, minValue = DBL_MAX, totaltime = 0.0;
   char *ofile = NULL;
   FILE *fid;
   clock_t startTime, endTime;


   /* Parse command line arguments given by user	*/
   for (i = 1; i < argc; i++)
   {
      if (argv[i][0] != '-')
         usage(argv[0]);

      c = argv[i][1];

      switch (c)
      {
         case 'N':
                if (++i >= argc)
                   usage(argv[0]);

		NP = atoi(argv[i]);
                break;
         case 'G':
                if (++i >= argc)
                   usage(argv[0]);

                Gmax = atoi(argv[i]);
                break;
         case 'C':
                if (++i >= argc)
                   usage(argv[0]);

                CR = atof(argv[i]);
                break;
         case 'F':
                if (++i >= argc)
                   usage(argv[0]);

                F = atof(argv[i]);
                break;
         case 'o':
                if (++i >= argc)
                   usage(argv[0]);

		ofile = argv[i];
                break;
         case 's':	/* Flag for using same seeds for		*/
                s = 0;	/* different runs				*/
                break;
         case 'h':
         case 'u':
         default:
		usage(argv[0]);
      }
   }

   if (s) INITRAND;

   /* Printing out information about optimization process for the user	*/
   printf("Program parameters: ");
   printf("NP = %d, Gmax = %d, CR = %.2f, F = %.2f\n",
	NP, Gmax, CR, F);

   printf("Dimension of the problem: %d\n", D);

   /* Starting timer    */
   startTime = clock();

   /* Allocating memory for current and next population, intializing
      current population with uniformly distributed random values and
      calculating value for the objective function	*/

   pPop = (double **)malloc(NP * sizeof(double *));
   if (NULL == pPop) perror("malloc");

   pNext = (double **)malloc(NP*sizeof(double *));
   if (NULL == pNext) perror("malloc");

   for (i = 0; i < NP; i++)
   {
      pPop[i] = (double *)malloc((D+1)*sizeof(double));
      if (NULL == pPop[i]) perror("malloc");

      for (j = 0; j < D; j++)
         pPop[i][j] = Xl[j] + (Xu[j] - Xl[j])*URAND;

      pPop[i][D] = func(pPop[i]);
      numOfFuncEvals++;

      pNext[i] = (double *)malloc((D+1)*sizeof(double));
      if (NULL == pNext[i]) perror("malloc");
   }

   /* Allocating memory for a trial vector U	*/
   U = (double *)malloc((D+1)*sizeof(double));
   if (NULL == U) perror("malloc");


   /* The main loop of the algorithm	*/
   for (k = 0; k < Gmax; k++)
   {
      for (i = 0; i < NP; i++)	/* Going through whole population	*/
      {

        /* Selecting random r1, r2, and r3 to individuals of
           the population such that i != r1 != r2 != r3	*/
        do
        {
           r1 = (int)(NP*URAND);
        } while(r1 == i);

        do
        {
           r2 = (int)(NP*URAND);
        } while(r2 == i || r2 == r1);

        /* Find best value */
        for (best = 0; best < NP; best++)
        {
            if (pPop[best][D] < minValue)
            {
                minValue = pPop[best][D];
            }
        }

        jrand = (int)(D*URAND);

        /* Mutation and crossover	*/
        for (j = 0; j < D; j++)
        {
           if (URAND < CR || j == jrand)
           {
              U[j] = pPop[i][j] + F*(pPop[r1][j] - pPop[r2][j]) + F*(pPop[best][j] - pPop[i][j]);
           }
           else
              U[j] = pPop[i][j];
        }

        U[D] = func(U);
        numOfFuncEvals++;

        /* Comparing the trial vector 'U' and the old individual
           'pNext[i]' and selecting better one to continue in the
           Next Population.	*/
        if (U[D] <= pPop[i][D])
        {
           iptr = U;
           U = pNext[i];
           pNext[i] = iptr;
        }
        else
        {
            for (j = 0; j <= D; j++)
                pNext[i][j] = pPop[i][j];
        }

      }	/* End of the going through whole population	*/


      /* Pointers of old and new population are swapped	*/

      ptr = pPop;
      pPop = pNext;
      pNext = ptr;

   }	/* End of the main loop		*/


   /* Stopping timer	*/

   endTime = clock();
   totaltime = (double)(endTime - startTime);


   /* If user has defined output file, the whole final pPopation is
      saved to the file						*/

   if (ofile != NULL)
   {
      if ((fid=(FILE *)fopen(ofile,"a")) == NULL)
      {
         fprintf(stderr,"Error in opening file %s\n\n",ofile);
         usage(argv[0]);
      }

      for (i=0; i < NP; i++)
      {
         for (j=0; j <= D; j++)
            fprintf(fid, "%.15e ", pPop[i][j]);
         fprintf(fid, "\n");
      }
      fclose(fid);
   }

   /* Finding best individual	*/

   for (i=0; i < NP; i++)
   {
      if (pPop[i][D] < minValue)
      {
         minValue = pPop[i][D];
         index = i;
      }
   }

   /* Printing out information about optimization process for the user	*/

   printf("Execution time: %.3f s\n", totaltime / (double)CLOCKS_PER_SEC);
   printf("Number of objective function evaluations: %d\n", numOfFuncEvals);

   printf("Solution:\nValues of variables: ");
   for (i=0; i < D; i++)
      printf("%.15f ", pPop[index][i]);

   printf("\nObjective function value: ");
   printf("%.15f\n", pPop[index][D]);


   /* Freeing dynamically allocated memory	*/

   for (i=0; i < NP; i++)
   {
      free(pPop[i]);
      free(pNext[i]);
   }
   free(pPop);
   free(pNext);
   free(U);

   return(0);
}

