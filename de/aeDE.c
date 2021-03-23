#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <float.h>
#include <math.h>

/* Function definitions		*/

double func(double *);
int usage(char *);
inline double getCurrentToBest(double *, double *, double *, double *, double *);
inline double getRand1(double *, double *, double *, double *);

/* Random number generator defined by URAND should return
   double-precision floating-point values uniformly distributed
   over the interval [0.0, 1.0)					*/

#define URAND	((double)rand()/((double)RAND_MAX + 1.0))

float RandomBetween(float smallNumber, float bigNumber)
{
    float diff = bigNumber - smallNumber;
    return (((float) rand() / RAND_MAX) * diff) + smallNumber;
}

/* Definition for random number generator initialization	*/

#define INITRAND srand(time(0))

/* Definition for a threshold of mutation scheme */
#define THRESHOLD (double)0.00001

#define FALSE 0
#define TRUE 1

/* Usage for the program	*/

int usage(char *str)
{
   fprintf(stderr,"Usage: %s [-h] [-u] [-s] [-N NP (20*D)] ", str);
   fprintf(stderr,"[-G maxIter (2000)]\n");
   fprintf(stderr,"\t[-C crossover constant, CR (0.9)]\n");
   fprintf(stderr,"\t[-F mutation scaling factor, F (0.9)]\n");
   fprintf(stderr,"\t[-o <outputfile>]\n\n");
   fprintf(stderr,"\t-s does not initialize random number generator\n");
   exit(-1);
}

inline double getCurrentToBest(double *pCurrent, double *pBest, double *pRand1, double *pRand2, double *pF)
{
   return (*pCurrent + (*pF)*(*pRand1 - *pRand2) + (*pF)*(*pBest - *pCurrent));
}

inline double getRand1(double *pRand1, double *pRand2, double *pRand3, double *pF)
{
   return (*pRand1 + (*pF)*(*pRand2 - *pRand3));
}

// Function to swap position of elements
void swap(double *a, double *b) {
  double t = *a;
  *a = *b;
  *b = t;
}

// Function to partition the array on the basis of pivot element
double partition(double array[], int low, int high) {
  
  // Select the pivot element
  double pivot = array[high];
  int i = (low - 1);

  // Put the elements smaller than pivot on the left 
  // and greater than pivot on the right of pivot
  for (int j = low; j < high; j++) {
    if (array[j] <= pivot) {
      i++;
      swap(&array[i], &array[j]);
    }
  }

  swap(&array[i + 1], &array[high]);
  return (i + 1);
}

void quickSort(double array[], int low, int high) {
   if (low < high) 
   {
    // Select pivot position and put all the elements smaller 
    // than pivot on left and greater than pivot on right
    double pi = partition(array, low, high);

    // Sort the elements on the left of pivot
    quickSort(array, low, pi - 1);

    // Sort the elements on the right of pivot
    quickSort(array, pi + 1, high);
   }
}

int main(int argc, char **argv)
{
   register int i, j, l, k, m, r1, r2, r3, best, jrand, numOfFuncEvals = 0;
   extern int D;
   extern double Xl[], Xu[];
   int NP = 20*D, maxIter = 2000, lenOfUnionSet = NP*2, c, index = -1, s = 1;
   double **pPop, **pNext, **ptr, **U, **unionSet, *sortedArray;
   double CR = 0.7, F = 0.5, delta = 0.0, tolerance = 0.0000001, minValue = DBL_MAX, totaltime = 0.0,
          fMean = 0.0;
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

                maxIter = atoi(argv[i]);
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
   printf("NP = %d, maxIter = %d, CR = %.2f, F = %.2f, tolerance = %.6f, threshold = %.4f\n",
	NP, maxIter, CR, F, tolerance, THRESHOLD);

   printf("Dimension of the problem: %d\n", D);

   /* Starting timer    */
   startTime = clock();

   /* Allocating memory for current and next population, intializing
      current population with uniformly distributed random values and
      calculating value for the objective function	*/

   pPop = (double **)malloc(NP * sizeof(double *));
   if (NULL == pPop) perror("malloc");

   pNext = (double **)malloc(NP * sizeof(double *));
   if (NULL == pNext) perror("malloc");

   /* Allocating memory for a trial vector U	*/
   U = (double **)malloc(NP * sizeof(double *));
   if (NULL == U) perror("malloc");

   for (i = 0; i < NP; i++)
   {
      U[i] = (double *)malloc((D+1)*sizeof(double));
      if (NULL == U[i]) perror("malloc");

      pPop[i] = (double *)malloc((D+1)*sizeof(double));
      if (NULL == pPop[i]) perror("malloc");

      for (j = 0; j < D; j++)
         pPop[i][j] = Xl[j] + (Xu[j] - Xl[j])*URAND;

      /* Evaluate the fitness for each individual */
      pPop[i][D] = func(pPop[i]);
      numOfFuncEvals++;

      pNext[i] = (double *)malloc((D+1)*sizeof(double));
      if (NULL == pNext[i]) perror("malloc");
   } /*   for (i = 0; i < NP; i++) */

   /* The main loop of the algorithm	*/
   k = 0;
   do
   {
      for (i = 0; i < NP; i++)	/* Going through whole population	*/
      {

         /* Selecting random r1, r2 to individuals of
            the population such that i != r1 != r2	*/
         do
         {
            r1 = (int)(NP*URAND);
         } while(r1 == i);
 
         do
         {
            r2 = (int)(NP*URAND);
         } while((r2 == i) || (r2 == r1));

         jrand = (int)(D*URAND);

         /* Crossover */
         for (j = 0; j < D; j++)
         {
            if ((URAND < CR) || (j == jrand))
            {
               /* Mutation schemes */
               if (delta > THRESHOLD)
               {
                  do
                  {
                     r3 = (int)(NP*URAND);
                  } while((r3 == i) || (r3 == r1) || (r3 == r2));

                  U[i][j] = getRand1(&pPop[r1][j], &pPop[r2][j], &pPop[r3][j], &F);
               }
               else
               {
                  /* Find best individual */
                  for (minValue = DBL_MAX, l = 0; l < NP; l++)
                  {
                     if (pPop[l][D] < minValue)
                     {
                        minValue = pPop[l][D];
                        best = l; /* best individual to */
                     }
                  }

                  U[i][j] = getCurrentToBest(&pPop[i][j], &pPop[best][j], &pPop[r1][j], &pPop[r2][j], &F);
               }
            }
            else
            {
               U[i][j] = pPop[i][j];
            }
         }

         U[i][D] = func(&U[i][0]);
         numOfFuncEvals++;
      }	/* End of the going through whole population	*/

      /* Comparing the trial vector 'U' and the old individual
         'pNext[i]' and selecting better one to continue in the
         Next Population.	*/
      /* Selection process according alogorithm 1.
         Q = C U P to search for best individual */

      /* Allocating memory for an union vector Q	*/
      unionSet = (double **)malloc((NP+NP)*sizeof(double));
      if (NULL == unionSet) perror("malloc"); // size of trial vector 20 and pPop 20

      for (m = 0; m < (NP+NP); m++)
      {
         unionSet[m] = (double *)malloc((D+1)*sizeof(double));
         if (NULL == unionSet[m]) perror("malloc");
      }

      /* Copy trial vectors to unionSet */
      for (m = 0; m < NP; m++)
      {
         for (int n = 0; n <= D; n++)
            unionSet[m][n] = U[m][n];
      }

      /* Creating union set with target vectors */
      for (int pos = 0; pos < NP; pos++)
      {
         fMean += pPop[pos][D]; //Calculate mean value of objective functions

         if(unionSet[pos][D] == pPop[pos][D])
         {
            continue;
         }
         else
         {
            for (int n = 0; n <= D; n++)
               unionSet[m][n] = pPop[pos][n];
            m++;
         }
      }

      lenOfUnionSet = m;
      /* Do sorting over unionSet to find NP best individuals, in this case NP is 400 */
      /* Copy evaluated output for each set of design variable to new array for sorting */
      /* Allocating memory for a trial vector U	*/
      sortedArray = (double *)malloc((lenOfUnionSet)*sizeof(double));
      if (NULL == sortedArray) perror("malloc");

      for (int copyIndex = 0; copyIndex < lenOfUnionSet; copyIndex++)
         sortedArray[copyIndex] = unionSet[copyIndex][D];

      quickSort(sortedArray, 0, (lenOfUnionSet - 1));

      /* Matching and copying best individuals to next generation */
      for (int sortedIndex = 0; sortedIndex < NP; sortedIndex++)
      {
         for (int marchingIndex = 0; marchingIndex < lenOfUnionSet; marchingIndex++)
         {
            if (sortedArray[sortedIndex] == unionSet[marchingIndex][D])
            {
               for (int copyingIndex = 0; copyingIndex <= D; copyingIndex++)
                  pNext[sortedIndex][copyingIndex] = unionSet[marchingIndex][copyingIndex];
               break;
            }
         }
      }

      /* Calculating fMean */
      fMean /= NP;
      delta = fabs(((fMean/sortedArray[0]) - 1));

      /* Pointers of old and new population are swapped	*/
      ptr = pPop;
      pPop = pNext;
      pNext = ptr;
      k++;
   } while((delta > tolerance) && (k < maxIter)); /* main loop of aeDE */


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
   for (minValue = DBL_MAX, i = 0; i < NP; i++)
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
      free(U[i]);
   }
   for (i = 0; i < (NP+NP); i++)
      free(unionSet[i]);

   free(pPop);
   free(pNext);
   free(unionSet);
   free(U);
   free(sortedArray);

   return(0);
}

