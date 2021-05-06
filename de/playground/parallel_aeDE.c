#include "ae_de.h"
#include <string.h>
//#include <omp.h>

/* Function definitions		*/
//static void fix(double *, int);
static double getCurrentToBest(double *, double *, double *, double *, double *);
static double getRand1(double *, double *, double *, double *);
static void swap(double *a, double *b);
static double partition(double array[], int low, int high);
static void quickSort(double array[], int low, int high);
static void executeEvolutionOverOnePop(problemT *ctx,
                                       mutationSchemeT mutationOp,
                                       double **pPop,
                                       double **pNext,
                                       double *U,
                                       double *F,
                                       int *numOfEvalsOut,
                                       double CR,
                                       int numOfPop, 
                                       int varDimension);

/* Definition for random number generator initialization	*/

#define INITRAND srand(time(0))

static double getCurrentToBest(double *pCurrent, double *pBest, double *pRand1, double *pRand2, double *pF)
{
   return (*pCurrent + (*pF)*(*pRand1 - *pRand2) + (*pF)*(*pBest - *pCurrent));
}

static double getRand1(double *pRand1, double *pRand2, double *pRand3, double *pF)
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

static void swapArray(double a[], double b[], int size)
{
   for (int i = 0; i < size; i++)
   {
      double tmp = a[i];
      a[i] = b[i];
      b[i] = tmp;
   }
}

static void swap2DArray(double **a, double **b, int outerSize, int innerSize)
{
   for (int i = 0; i < outerSize; i++)
   {
      for(int j = 0; j < innerSize; j++)
      {
         double tmp = a[i][j];
         a[i][j] = b[i][j];
         b[i][j] = tmp;
      }

   }
}

static void executeEvolutionOverOnePop(problemT *ctx,
                                       mutationSchemeT mutationOp,
                                       double **pPop,
                                       double **pNext,
                                       double *U,
                                       double *F,
                                       int *numOfEvalsOut,
                                       double CR,
                                       int numOfPop, 
                                       int varDimension)
{
   int popIndex, jrand, numOfEvals = 0;
   int r1, r2, r3, j;
   int bestIndex, bestPop;
   double minVal;

   for (popIndex = 0; popIndex < numOfPop; popIndex++)
   {
      jrand = (int)(varDimension*URAND);

      /* Selecting random sub1_r1, sub1_r2 individuals of
         the population such that popIndex != sub1_r1 != sub1_r2 */
      do
      {
         r1 = (int)(numOfPop*URAND);
      } while(r1 == popIndex);

      do
      {
         r2 = (int)(numOfPop*URAND);
      } while((r2 == popIndex) || (r2 == r1));

      /* Crossover */
      for (j = 0; j < varDimension; j++)
      {
         if ((URAND < CR) || (j == jrand))
         {
            /* Mutation schemes */
            switch (mutationOp)
            {
               case RAND_1:
               {
                  do
                  {
                     r3 = (int)(numOfPop*URAND);
                  } while((r3 == popIndex) || (r3 == r1) || (r3 == r2));

                  U[j] = getRand1(&pPop[r1][j], 
                                 &pPop[r2][j], 
                                 &pPop[r3][j], 
                                 F);

                  break;
               }

               case CURRENT_TO_BEST:
               {
                  /* Find best individual */
                  for (minVal = DBL_MAX, bestIndex = 0; bestIndex < numOfPop; bestIndex++)
                  {
                     if (pPop[bestIndex][varDimension] < minVal)
                     {
                        minVal = pPop[bestIndex][varDimension];
                        bestPop = bestIndex; /* best individual to */
                     }
                  }

                  U[j] = getCurrentToBest(&pPop[popIndex][j], 
                                          &pPop[bestPop][j], 
                                          &pPop[r1][j], 
                                          &pPop[r2][j], 
                                          F);
                  break;
               }
               default:
                  break;
            }
         }
         else
         {
            U[j] = pPop[popIndex][j];
         }
      }

      U[varDimension] = ctx->penaltyFunc(&U[0]); // Evaluate trial vectors
      numOfEvals++;

      /* Comparing the trial vector 'U' and the old individual
         'pNext[popIndex]' and selecting better one to continue in the
         Next Population.	*/
      if (U[varDimension] <= pPop[popIndex][varDimension])
      {
         swapArray(&U[0], &pNext[popIndex][0], varDimension+1);
      }
      else
      {
         for (j = 0; j <= varDimension; j++)
            pNext[popIndex][j] = pPop[popIndex][j];
      }
   }

   swap2DArray(pPop, pNext, numOfPop, varDimension+1);
   *numOfEvalsOut = numOfEvals;
}

/*
 * In: base population / Out: two sub-pops raking by their fitness values
 */
static void applyEliteStrategy(double **pBasePop, 
                               double **pSubPop1, 
                               double **pSubPop2,
                               int varDimension,
                               int numOfBasePop)
{
   double *sortedArray;
   int subPopSize = numOfBasePop / 2;

   sortedArray = (double *)malloc(numOfBasePop*sizeof(double));
   if (NULL == sortedArray) perror("malloc");

   for (int copyingIndex = 0; copyingIndex < numOfBasePop; copyingIndex++)
   {
      sortedArray[copyingIndex] = pBasePop[copyingIndex][varDimension];
   }

   quickSort(sortedArray, 0, (numOfBasePop - 1));

   /* Good fitness member go to first sub-pop for current-to-best mutation */
   for (int sortedIndex = 0; sortedIndex < subPopSize; sortedIndex++)
   {
      for (int marchingIndex = 0; marchingIndex < numOfBasePop; marchingIndex++)
      {
         if (sortedArray[sortedIndex] == pBasePop[marchingIndex][varDimension])
         {
            for (int copyingIndex = 0; copyingIndex <= varDimension; copyingIndex++)
               pSubPop1[sortedIndex][copyingIndex] = pBasePop[marchingIndex][copyingIndex];
            break;
         }
      }
   }

   /* This sub-pop for rand/1 mutation scheme */
   for (int copyingIndex = 0; copyingIndex < subPopSize; copyingIndex++)
   {
      for(int varIndex = 0; varIndex <= varDimension; varIndex++)
         pSubPop2[copyingIndex][varIndex] = pBasePop[copyingIndex+subPopSize][varIndex];
   }

   free(sortedArray);
}

static int isSatisfiedRastrigin(double **pPop, int numOfPop, int varDimension)
{
   int isMinimized = 0;
   /* Checking exit condition */
   for (int checkIndex = 0; checkIndex < numOfPop; checkIndex++)
   {
      if (0.0 == pPop[checkIndex][varDimension])
      {
         isMinimized = 1;
         break;
      }   
   }

   return isMinimized;
}
/********************** Exported interface implementation ***************************/

void run_parallel_aeDE(int numOfPop,
                       int iter,
                       int varDimension,
                       problemT *problemCtx,
                       resultT *result,
                       int isMinimized)
{
   int mainIndex, popIndex, numOfFuncEvals = 0, numOfFuncEvals1 = 0, numOfFuncEvals2 = 0;
   int j;
   int index = -1, s = 1;
   int subPopSize = numOfPop / 2;
   double **pPop, **pElitePop, **pSubPop2, **pSubNext2, *U2;
   double **pSubPop1, **pSubNext1, *U1;
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
   pElitePop = (double **)malloc(subPopSize * sizeof(double *));
   if (NULL == pElitePop) perror("malloc");
   pSubPop1 = (double **)malloc(subPopSize * sizeof(double *));
   if (NULL == pSubPop1) perror("malloc");
   pSubNext1 = (double **)malloc(subPopSize * sizeof(double *));
   if (NULL == pSubNext1) perror("malloc");

   pSubPop2 = (double **)malloc(subPopSize * sizeof(double *));
   if (NULL == pSubPop2) perror("malloc");
   pSubNext2 = (double **)malloc(subPopSize * sizeof(double *));
   if (NULL == pSubNext2) perror("malloc");

   /* Allocating memory for a trial vector U	*/
   U1 = (double *)malloc((varDimension+1)*sizeof(double));
   if (NULL == U1) perror("malloc");
   U2 = (double *)malloc((varDimension+1)*sizeof(double));
   if (NULL == U2) perror("malloc");
   
   for (popIndex = 0; popIndex < numOfPop; popIndex++)
   {
      pPop[popIndex] = (double *)malloc((varDimension+1)*sizeof(double));
      if (NULL == pPop[popIndex]) perror("malloc");

      /* Initialization */
      for (j = 0; j < varDimension; j++)
         pPop[popIndex][j] = problemCtx->lowerConstraints[j] + 
                     (problemCtx->upperConstraints[j] - problemCtx->lowerConstraints[j])*URAND;

      /* Evaluate the fitness for each individual */
      pPop[popIndex][varDimension] = problemCtx->penaltyFunc(pPop[popIndex]);
      numOfFuncEvals++;
   } /* for (popIndex = 0; popIndex < numOfPop; popIndex++) */

   /* Split to two sub-pops for co-evolution */
   /* Allocate mem for sub-pops */
   for (popIndex = 0; popIndex < subPopSize; popIndex++)
   {
      pElitePop[popIndex] = (double *)malloc((varDimension+1)*sizeof(double));
      if (NULL == pElitePop[popIndex]) perror("malloc");

      pSubPop1[popIndex] = (double *)malloc((varDimension+1)*sizeof(double));
      if (NULL == pSubPop1[popIndex]) perror("malloc");
      pSubPop2[popIndex] = (double *)malloc((varDimension+1)*sizeof(double));
      if (NULL == pSubPop2[popIndex]) perror("malloc");

      pSubNext1[popIndex] = (double *)malloc((varDimension+1)*sizeof(double));
      if (NULL == pSubNext1[popIndex]) perror("malloc");
      pSubNext2[popIndex] = (double *)malloc((varDimension+1)*sizeof(double));
      if (NULL == pSubNext2[popIndex]) perror("malloc");
   }

   applyEliteStrategy(pPop, pSubPop1, pSubPop2, varDimension, numOfPop);

   /* The main loop of the algorithm	*/
   for (mainIndex = 0; mainIndex < iter; mainIndex++)
   {
      /* Population 1 */
      executeEvolutionOverOnePop(problemCtx,
                                 CURRENT_TO_BEST,
                                 pSubPop1,
                                 pSubNext1,
                                 U1,
                                 &F,
                                 &numOfFuncEvals1,
                                 CR,
                                 subPopSize,
                                 varDimension);

      executeEvolutionOverOnePop(problemCtx,
                                 RAND_1,
                                 pSubPop2,
                                 pSubNext2,
                                 U2,
                                 &F,
                                 &numOfFuncEvals2,
                                 CR,
                                 subPopSize,
                                 varDimension);  

      numOfFuncEvals += (numOfFuncEvals1 + numOfFuncEvals2);

      /**** Construct elite pop and update sub-pops ****/

      /* Merge two sub-pops */
      for (int copyingIndex = 0; copyingIndex < subPopSize; copyingIndex++)
      {
         for (int copyingIndex2 = 0; copyingIndex2 <= varDimension; copyingIndex2++)
            pPop[copyingIndex][copyingIndex2] = pSubPop1[copyingIndex][copyingIndex2];                         
      }

      for (int copyingIndex = 0; copyingIndex < subPopSize; copyingIndex++)
      {
         for (int copyingIndex2 = 0; copyingIndex2 <= varDimension; copyingIndex2++)
            pPop[(copyingIndex+subPopSize)][copyingIndex2] = pSubPop2[copyingIndex][copyingIndex2];
      }

      applyEliteStrategy(pPop, pSubPop1, pSubPop2, varDimension, numOfPop);

      if(1 == isSatisfiedRastrigin(pPop, numOfPop, varDimension))
         break;
   } /* Main loop */

   /* Calculating and output results */

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
   {
      result->optimizedVars[copyingInx] = pPop[index][copyingInx];
   }


   result->fitnessVal = pPop[index][varDimension];
   result->numOfEvals = numOfFuncEvals;

   /* Freeing dynamically allocated memory	*/
   for (popIndex = 0; popIndex < numOfPop; popIndex++)
   {
      free(pPop[popIndex]);
   }
      
   for (popIndex = 0; popIndex < subPopSize; popIndex++)
   {
      free(pSubPop1[popIndex]);
      free(pSubPop2[popIndex]);
      free(pSubNext1[popIndex]);
      free(pSubNext2[popIndex]);
   }

   free(pPop);
   free(pSubNext1);
   free(pSubNext2);
   free(pSubPop1);
   free(pSubPop2);
   free(U1);
   free(U2);
}

