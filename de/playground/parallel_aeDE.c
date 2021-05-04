#include "ae_de.h"
#include <omp.h>

/* Function definitions		*/
//static void fix(double *, int);
inline double getCurrentToBest(double *, double *, double *, double *, double *);
inline double getRand1(double *, double *, double *, double *);
static void swap(double *a, double *b);
static double partition(double array[], int low, int high);
static void quickSort(double array[], int low, int high);

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
/********************** Exported interface implementation ***************************/

void run_parallel_aeDE(int numOfPop,
                       int iter,
                       int varDimension,
                       problemT *problemCtx,
                       resultT *result,
                       int isMinimized)
{
   int popIndex, popIndex1, popIndex2, bestIndex, mainIndex, best, jrand1, jrand2, numOfFuncEvals = 0, numOfFuncEvals1 = 0, numOfFuncEvals2 = 0;
   int sub1_r1, sub1_r2, sub1_r3, sub2_r1, sub2_r2, sub2_r3;
   int sub1_j, sub2_j, j;
   int index = -1, s = 1;
   double **pPop, **pSubPop2, **pSubNext2, **ptrSub2, *U2, *iptrSub2, *sortedArray;
   double **pSubPop1, **pSubNext1, **ptrSub1, *U1, *iptrSub1;
   double CR = 0.0, F = 0.5, minValue = DBL_MAX;
   clock_t startTime, endTime;

   if (s) INITRAND;

   /* Printing out information about optimization process for the user	*/
   printf("Program parameters: ");
   printf("numOfPop = %d, maxIter = %d, CR = %.2f, F = %.2f\n",
	   numOfPop, iter, CR, F);
   printf("Dimension of the problem: %d\n", varDimension);

   /* Starting timer    */
   startTime = clock();

   /* Allocating memory for current and next population, intializing
      current population with uniformly distributed random values and
      calculating value for the objective function	*/
   pPop = (double **)malloc(numOfPop * sizeof(double *));
   if (NULL == pPop) perror("malloc");
   pSubPop1 = (double **)malloc((numOfPop/2) * sizeof(double *));
   if (NULL == pSubPop1) perror("malloc");
   pSubNext1 = (double **)malloc((numOfPop/2) * sizeof(double *));
   if (NULL == pSubNext1) perror("malloc");

   pSubPop2 = (double **)malloc((numOfPop/2) * sizeof(double *));
   if (NULL == pSubPop2) perror("malloc");
   pSubNext2 = (double **)malloc((numOfPop/2) * sizeof(double *));
   if (NULL == pSubNext2) perror("malloc");

   /* Allocating memory for a trial vector U	*/
   U1 = (double *)malloc((varDimension+1)*sizeof(double));
   if (NULL == U1) perror("malloc");
   U2 = (double *)malloc((varDimension+1)*sizeof(double));
   if (NULL == U2) perror("malloc");
   
   #pragma omp parallel for
   for (popIndex = 0; popIndex < numOfPop; popIndex++)
   {
      pPop[popIndex] = (double *)malloc((varDimension+1)*sizeof(double));
      if (NULL == pPop[popIndex]) perror("malloc");

      /* Initialization */
      #pragma omp parallel for
      for (j = 0; j < varDimension; j++)
         pPop[popIndex][j] = problemCtx->lowerConstraints[j] + 
                     (problemCtx->upperConstraints[j] - problemCtx->lowerConstraints[j])*URAND;

      /* Evaluate the fitness for each individual */
      pPop[popIndex][varDimension] = problemCtx->penaltyFunc(pPop[popIndex]);
      numOfFuncEvals++;
   } /* for (popIndex = 0; popIndex < numOfPop; popIndex++) */

   /* Split to two sub-pops for co-evolution */
   /* Allocate mem for sub-pops */
   #pragma omp parallel for
   for (popIndex = 0; popIndex < (numOfPop/2); popIndex++)
   {
      pSubPop1[popIndex] = (double *)malloc((varDimension+1)*sizeof(double));
      if (NULL == pSubPop1[popIndex]) perror("malloc");
      pSubPop2[popIndex] = (double *)malloc((varDimension+1)*sizeof(double));
      if (NULL == pSubPop2[popIndex]) perror("malloc");

      pSubNext1[popIndex] = (double *)malloc((varDimension+1)*sizeof(double));
      if (NULL == pSubNext1[popIndex]) perror("malloc");
      pSubNext2[popIndex] = (double *)malloc((varDimension+1)*sizeof(double));
      if (NULL == pSubNext2[popIndex]) perror("malloc");
   }

   sortedArray = (double *)malloc((numOfPop)*sizeof(double));
   if (NULL == sortedArray) perror("malloc");

   for (int copyingIndex = 0; copyingIndex < numOfPop; copyingIndex++)
   {
      sortedArray[copyingIndex] = pPop[copyingIndex][varDimension];
   }

   quickSort(sortedArray, 0, (numOfPop - 1));

   /* Good fitness member go to first sub-pop for current-to-best mutation */
   for (int sortedIndex = 0; sortedIndex < (numOfPop/2); sortedIndex++)
   {
      for (int marchingIndex = 0; marchingIndex < numOfPop; marchingIndex++)
      {
         if (sortedArray[sortedIndex] == pPop[marchingIndex][varDimension])
         {
            for (int copyingIndex = 0; copyingIndex <= varDimension; copyingIndex++)
               pSubPop1[sortedIndex][copyingIndex] = pPop[marchingIndex][copyingIndex];
            break;
         }
      }
   }

   /* This sub-pop for rand/1 mutation scheme */
   for (int copyingIndex = 0; copyingIndex < (numOfPop/2); copyingIndex++)
   {
      for(int varIndex = 0; varIndex <= varDimension; varIndex++)
         pSubPop2[copyingIndex][varIndex] = pPop[copyingIndex+10][varIndex];
   }


   /* The main loop of the algorithm	*/
   for (mainIndex = 0; mainIndex < iter; mainIndex++)
   {
      /* Population 1 */
      for (popIndex1 = 0; popIndex1 < (numOfPop/2); popIndex1++)	/* Going through whole population	*/
      {
         jrand1 = (int)(varDimension*URAND); // line 7
   
         /* Selecting random sub1_r1, sub1_r2 individuals of
            the population such that popIndex != sub1_r1 != sub1_r2	
            line 11 and 14 */
         do
         {
            sub1_r1 = (int)((numOfPop/2)*URAND);
         } while(sub1_r1 == popIndex1);
   
         do
         {
            sub1_r2 = (int)((numOfPop/2)*URAND);
         } while((sub1_r2 == popIndex1) || (sub1_r2 == sub1_r1));

         do
         {
            sub1_r3 = (int)((numOfPop/2)*URAND);
         } while((sub1_r3 == popIndex1) || (sub1_r3 == sub1_r1) || (sub1_r3 == sub1_r2));

         /* Crossover */
         for (sub1_j = 0; sub1_j < varDimension; sub1_j++)
         {
            if ((URAND < CR) || (sub1_j == jrand1))
            {
               /* Mutation schemes */
               U1[sub1_j] = getRand1(&pSubPop1[sub1_r1][sub1_j], 
                                     &pSubPop1[sub1_r2][sub1_j], 
                                     &pSubPop1[sub1_r3][sub1_j], 
                                     &F);
            }
            else
            {
               U1[sub1_j] = pSubPop1[popIndex1][sub1_j];
            }
         }
   
         U1[varDimension] = problemCtx->penaltyFunc(&U1[0]); // Evaluate trial vectors | line 21
         numOfFuncEvals1++;

         /* Comparing the trial vector 'U' and the old individual
            'pNext[popIndex]' and selecting better one to continue in the
            Next Population.	*/
         if (U1[varDimension] <= pSubPop1[popIndex1][varDimension])
         {
            iptrSub1 = U1;
            U1 = pSubNext1[popIndex1];
            pSubNext1[popIndex1] = iptrSub1;
         }
         else
         {
            for (sub1_j = 0; sub1_j <= varDimension; sub1_j++)
               pSubNext1[popIndex1][sub1_j] = pSubPop1[popIndex1][sub1_j];
         }
         /* elite strategy to pSubNext populations */
         ptrSub1 = pSubPop1;
         pSubPop1 = pSubNext1;
         pSubNext1 = ptrSub1;
      }	/* End of the going through sub-population 1*/

      /* Population 2 */
      for (popIndex2 = 0; popIndex2 < (numOfPop/2); popIndex2++)	/* Going through whole population	*/
      {
         jrand2 = (int)(varDimension*URAND); // line 7
   
         /* Selecting random sub2_r1, sub2_r2 individuals of
            the population such that popIndex2 != sub2_r1 != sub2_r2	
            line 11 and 14 */
         do
         {
            sub2_r1 = (int)((numOfPop/2)*URAND);
         } while(sub2_r1 == popIndex2);
   
         do
         {
            sub2_r2 = (int)((numOfPop/2)*URAND);
         } while((sub2_r2 == popIndex2) || (sub2_r2 == sub2_r1));
   
         /* Crossover */
         for (sub2_j = 0; sub2_j < varDimension; sub2_j++)
         {
            if ((URAND < CR) || (sub2_j == jrand2))
            {
               /* Mutation schemes */
               /* Find best individual */
               for (minValue = DBL_MAX, bestIndex = 0; bestIndex < numOfPop; bestIndex++)
               {
                  if (pSubPop2[bestIndex][varDimension] < minValue)
                  {
                     minValue = pSubPop2[bestIndex][varDimension];
                     best = bestIndex; /* best individual to */
                  }

                  U2[sub2_j] = getCurrentToBest(&pSubPop2[popIndex2][sub2_j], 
                                                &pPop[best][sub2_j], 
                                                &pPop[sub2_r1][sub2_j], 
                                                &pPop[sub2_r2][sub2_j], 
                                                &F);
               }
            }
            else
            {
               U2[sub2_j] = pSubPop2[popIndex2][sub2_j];
            }
         }
   
         U2[varDimension] = problemCtx->penaltyFunc(&U2[0]); // Evaluate trial vectors | line 21
         numOfFuncEvals2++;

         /* Comparing the trial vector 'U' and the old individual
            'pNext[popIndex]' and selecting better one to continue in the
            Next Population.	*/
         if (U2[varDimension] <= pSubPop2[popIndex2][varDimension])
         {
            iptrSub2 = U2;
            U2 = pSubNext2[popIndex2];
            pSubNext2[popIndex2] = iptrSub2;
         }
         else
         {
            for (sub2_j = 0; sub2_j <= varDimension; sub2_j++)
               pSubNext2[popIndex2][sub2_j] = pSubPop2[popIndex2][sub2_j];
         }

         /* elite strategy to pSubNext populations */
         ptrSub2 = pSubPop2;
         pSubPop2 = pSubNext2;
         pSubNext2 = ptrSub2;
      }	/* End of the going through sub-population 2 */
   }

   /* Stopping timer	*/
   endTime = clock();
   result->executionTime = (double)(endTime - startTime)/(double)CLOCKS_PER_SEC;

   /* Finding best individual	*/
   for (minValue = DBL_MAX, popIndex = 0; popIndex < numOfPop; popIndex++)
   {
      if (pPop[popIndex][varDimension] < minValue)
      {
         minValue = pPop[popIndex][varDimension];
         index = popIndex;
      }
   }

   for (int copyingInx = 0; copyingInx <= numOfPop; copyingInx++)
      result->optimizedVars[copyingInx] = pPop[index][copyingInx];

   result->fitnessVal = pPop[index][varDimension];
   result->numOfEvals = numOfFuncEvals + numOfFuncEvals1 + numOfFuncEvals2;

   /* Freeing dynamically allocated memory	*/
   for (popIndex = 0; popIndex < numOfPop; popIndex++)
   {
      free(pPop[popIndex]);
   }

   for (popIndex = 0; popIndex < (numOfPop/2); popIndex++)
   {
      free(pSubPop1[popIndex]);
      free(pSubPop2[popIndex]);
      free(pSubNext1[popIndex]);
      free(pSubNext2[popIndex]);
   }

   free(pPop);
   free(U1);
   free(U2);
   free(pSubNext1);
   free(pSubNext2);
}

