#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include "ae_de.h"

/* Function definitions		*/

//double func(double *);
static void fix(double *, int);
inline double getCurrentToBest(double *, double *, double *, double *, double *);
inline double getRand1(double *, double *, double *, double *);
static void swap(double *a, double *b);
static double partition(double array[], int low, int high);
static void quickSort(double array[], int low, int high);
static int isArrayIdentical(double array1[], double array2[], int length);

/* Definition for random number generator initialization	*/

#define INITRAND srand(time(0))

inline double getCurrentToBest(double *pCurrent, double *pBest, double *pRand1, double *pRand2, double *pF)
{
   return (*pCurrent + (*pF)*(*pRand1 - *pRand2) + (*pF)*(*pBest - *pCurrent));
}

inline double getRand1(double *pRand1, double *pRand2, double *pRand3, double *pF)
{
   return (*pRand1 + (*pF)*(*pRand2 - *pRand3));
}

// Function to swap position of elements
static void swap(double *a, double *b) {
  double t = *a;
  *a = *b;
  *b = t;
}

// Function to partition the array on the basis of pivot element
static double partition(double array[], int low, int high) {
  
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

static void quickSort(double array[], int low, int high) {
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

static int isArrayIdentical(double array1[], double array2[], int length)
{
   int isIndentical = TRUE;

   for (int i = 0; i < length; i++)
   {
      if (array1[i] == array2[i])
      {
         continue;
      }
      else
      {
         isIndentical = FALSE;
         break;
      }
   }

   return isIndentical;
}

/********************** Exported interface implementation ***************************/

void run_aeDE(int numOfPop,
              int iter,
              float threshHold, 
              float tolerance,
              int varDimension,
              problemT *problemCtx,
              int isMinimized)
{
   register int i, j, l, k, m, r1, r2, r3, best, jrand, numOfFuncEvals = 0;
   //extern double Xl[], Xu[];
   int lenOfUnionSet = numOfPop*2, index = -1, s = 1;
   double **pPop, **pNext, **ptr, **U, **unionSet = NULL, *sortedArray = NULL;
   double CR = 0.7, F = 0.7, delta = 0.0, minValue = DBL_MAX, totaltime = 0.0,
          fMean = 0.0;
   char *ofile = NULL;
   FILE *fid;
   clock_t startTime, endTime;

   if (s) INITRAND;

   /* Printing out information about optimization process for the user	*/
   printf("Program parameters: ");
   printf("numOfPop = %d, maxIter = %d, CR = %.2f, F = %.2f, tolerance = %.6f, threshold = %.6f\n",
	numOfPop, iter, CR, F, tolerance, threshHold);

   printf("Dimension of the problem: %d\n", varDimension);

   /* Starting timer    */
   startTime = clock();

   /* Allocating memory for current and next population, intializing
      current population with uniformly distributed random values and
      calculating value for the objective function	*/

   pPop = (double **)malloc(numOfPop * sizeof(double *));
   if (NULL == pPop) perror("malloc");

   pNext = (double **)malloc(numOfPop * sizeof(double *));
   if (NULL == pNext) perror("malloc");

   /* Allocating memory for a trial vector U	*/
   U = (double **)malloc(numOfPop * sizeof(double *));
   if (NULL == U) perror("malloc");

   for (i = 0; i < numOfPop; i++)
   {
      U[i] = (double *)malloc((varDimension+1)*sizeof(double));
      if (NULL == U[i]) perror("malloc");

      pPop[i] = (double *)malloc((varDimension+1)*sizeof(double));
      if (NULL == pPop[i]) perror("malloc");

      /* Initialization */
      for (j = 0; j < varDimension; j++)
         //pPop[i][j] = Xl[j] + (Xu[j] - Xl[j])*URAND;
         pPop[i][j] = problemCtx->lowerConstraints[j] + 
                     (problemCtx->upperConstraints[j] - problemCtx->lowerConstraints[j])*URAND;

      /* Evaluate the fitness for each individual */
      pPop[i][varDimension] = problemCtx->penaltyFunc(pPop[i]); //func(pPop[i]);
      numOfFuncEvals++;

      pNext[i] = (double *)malloc((varDimension+1)*sizeof(double));
      if (NULL == pNext[i]) perror("malloc");
   } /*   for (i = 0; i < numOfPop; i++) */

   /* The main loop of the algorithm	*/
   k = 0;
   do
   {
      for (i = 0; i < numOfPop; i++)	/* Going through whole population	*/
      {
         F = FRAND; // line 5
         CR = CRRAND; // line 6
         jrand = (int)(varDimension*URAND); // line 7

         /* Selecting random r1, r2 individuals of
            the population such that i != r1 != r2	
            line 11 and 14 */
         do
         {
            r1 = (int)(numOfPop*URAND);
         } while(r1 == i);
 
         do
         {
            r2 = (int)(numOfPop*URAND);
         } while((r2 == i) || (r2 == r1));

         /* Crossover */
         for (j = 0; j < varDimension; j++)
         {
            if ((URAND < CR) || (j == jrand)) // line 9 
            {
               /* Mutation schemes */
               if (delta > threshHold) // line 10
               {
                  do
                  {
                     r3 = (int)(numOfPop*URAND);
                  } while((r3 == i) || (r3 == r1) || (r3 == r2)); // line 11

                  U[i][j] = getRand1(&pPop[r1][j], &pPop[r2][j], &pPop[r3][j], &F); // line 12
               }
               else
               {
                  /* Find best individual | line 14 */
                  for (minValue = DBL_MAX, l = 0; l < numOfPop; l++)
                  {
                     if (pPop[l][varDimension] < minValue)
                     {
                        minValue = pPop[l][varDimension];
                        best = l; /* best individual to */
                     }
                  }

                  U[i][j] = getCurrentToBest(&pPop[i][j], &pPop[best][j], &pPop[r1][j], &pPop[r2][j], &F); // line 15
               }
            }
            else
            {
               U[i][j] = pPop[i][j]; // line 18
            }
         }

         U[i][varDimension] = problemCtx->penaltyFunc(&U[i][0]); // Evaluate trial vectors | line 21
         numOfFuncEvals++;
      }	/* End of the going through whole population	*/

      /* Selection process according alogorithm 1.
         Q = C U P to search for best individual */

      /* Allocating memory for an union vector Q	*/
      if (NULL == unionSet)
      {
         unionSet = (double **)malloc((numOfPop+numOfPop)*sizeof(double));
         if (NULL == unionSet) perror("malloc"); // size of trial vector 20 and pPop 20

         for (m = 0; m < (numOfPop+numOfPop); m++)
         {
            unionSet[m] = (double *)malloc((varDimension+1)*sizeof(double));
            if (NULL == unionSet[m]) perror("malloc");
         }
      }

      /* Copy trial vectors U to unionSet */
      for (m = 0; m < numOfPop; m++)
      {
         for (int n = 0; n <= varDimension; n++)
            unionSet[m][n] = U[m][n];
      }

      /* Creating union set with target vectors */
      for (int pos = 0; pos < numOfPop; pos++)
      {
         fMean += pPop[pos][varDimension]; // To calculate mean value of objective functions

         if(isArrayIdentical(&unionSet[pos][0], &pPop[pos][0], varDimension+1))
         {
            continue;
         }
         else
         {
            for (int n = 0; n <= varDimension; n++)
               unionSet[m][n] = pPop[pos][n];
            m++;
         }
      }

      lenOfUnionSet = m;
      /* Do sorting over unionSet to find numOfPop best individuals */
      /* Copy evaluated output for each set of design variables to new array for sorting */
      /* Allocating memory	*/
      if (NULL == sortedArray)
      {
         sortedArray = (double *)malloc((lenOfUnionSet)*sizeof(double));
         if (NULL == sortedArray) perror("malloc");
      }

      for (int copyIndex = 0; copyIndex < lenOfUnionSet; copyIndex++)
         sortedArray[copyIndex] = unionSet[copyIndex][varDimension]; //Sort with the ascending direction. Smallest/best fitness value is the first member.

      quickSort(sortedArray, 0, (lenOfUnionSet - 1));

      /* Matching and copying best individuals to next generation */
      for (int sortedIndex = 0; sortedIndex < numOfPop; sortedIndex++)
      {
         for (int marchingIndex = 0; marchingIndex < lenOfUnionSet; marchingIndex++)
         {
            if (sortedArray[sortedIndex] == unionSet[marchingIndex][varDimension])
            {
               for (int copyingIndex = 0; copyingIndex <= varDimension; copyingIndex++)
                  pNext[sortedIndex][copyingIndex] = unionSet[marchingIndex][copyingIndex];
               break;
            }
         }
      }

      /* Calculating fMean */
      fMean /= numOfPop;
      delta = fabs(((fMean/sortedArray[0]) - 1));

      /* Pointers of old and new population are swapped	*/
      ptr = pPop;
      pPop = pNext;
      pNext = ptr;
      k++;
   } while((delta > tolerance) && (k < iter)); /* main loop of aeDE */

   /* Post aeDE processing */
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
      }

      for (i=0; i < numOfPop; i++)
      {
         for (j=0; j <= varDimension; j++)
            fprintf(fid, "%.15e ", pPop[i][j]);
         fprintf(fid, "\n");
      }
      fclose(fid);
   }

   /* Finding best individual	*/
   for (minValue = DBL_MAX, i = 0; i < numOfPop; i++)
   {
      if (pPop[i][varDimension] < minValue)
      {
         minValue = pPop[i][varDimension];
         index = i;
      }
   }

   /* Printing out information about optimization process for the user	*/
   printf("Execution time: %.3f s\n", totaltime / (double)CLOCKS_PER_SEC);
   printf("Number of objective function evaluations: %d\n", numOfFuncEvals);

   printf("Solution:\nValues of variables: ");
   for (i=0; i < varDimension; i++)
      printf("%.15f ", pPop[index][i]);
   printf("\nObjective function value: ");
   printf("%.15f\n", pPop[index][varDimension]);


   /* Freeing dynamically allocated memory	*/

   for (i=0; i < numOfPop; i++)
   {
      free(pPop[i]);
      free(pNext[i]);
      free(U[i]);
   }
   for (i = 0; i < (numOfPop+numOfPop); i++)
      free(unionSet[i]);

   free(pPop);
   free(pNext);
   free(unionSet);
   free(U);
   free(sortedArray);
}

