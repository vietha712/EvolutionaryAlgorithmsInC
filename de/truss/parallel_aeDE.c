#include "ae_de.h"
#include <string.h>
#include <omp.h>

#define SUB_POP_POS(current_pos,pop_size) ((current_pos - pop_size))

/* Function definitions		*/
//void fix(double *X, int length);
static double getCurrentToBest(double *, double *, double *, double *, double );
static double getRand1(double *, double *, double *, double );
static double getBest1(double *, double *, double *, double );
static void swap(double *a, double *b);
static double partition(double array[], int low, int high);
static void quickSort(double array[], int low, int high);
void fix(double *, int );
static void executeEvolutionOverOnePop(problemT *ctx,
                                       mutationSchemeT mutationOp,
                                       double **pPop,
                                       double **pNext,
                                       double *U,
                                       double F,
                                       double CR,
                                       int numOfPop, 
                                       int varDimension);

/* Definition for random number generator initialization	*/

#define INITRAND srand(time(0))

static inline double getCurrentToBest(double *pCurrent, double *pBest, double *pRand1, double *pRand2, double F)
{
   return (*pCurrent + (F)*(*pRand1 - *pRand2) + (F)*(*pBest - *pCurrent));
}

static inline double getRand1(double *pRand1, double *pRand2, double *pRand3, double F)
{
   return (*pRand1 + (F)*(*pRand2 - *pRand3));
}

static inline double getBest1(double *pBest, double *pRand1, double *pRand2, double F)
{
   return (*pBest + (F)*(*pRand1 - *pRand2));
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
                                       double F,
                                       double CR,
                                       int numOfPop, 
                                       int varDimension)
{
   int popIndex, jrand;
   int r1, r2, r3, j;
   int bestIndex, bestPop = 0;
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
               case BEST_1:
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

                  U[j] = getBest1(&pPop[bestPop][j], &pPop[r1][j], &pPop[r2][j], F);                
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
      
      fix(U, varDimension); // make the value to be in discrete type
      U[varDimension] = ctx->penaltyFunc(&U[0]); // Evaluate trial vectors

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
}

/*
 * In: base population / Out: two sub-pops raking by their fitness values
 * TODO: 
 */
static void applyEliteStrategy(double **pBasePop, 
                               double **pSubPop1, 
                               double **pSubPop2,
                               double **pSubPop3,
                               int varDimension,
                               int numOfBasePop)
{
   double *sortedArray;
   int subPopSize = numOfBasePop / 3;
   int sortedIndex = 0;

   sortedArray = (double *)malloc(numOfBasePop*sizeof(double));
   if (NULL == sortedArray) perror("malloc");

   for (int copyingIndex = 0; copyingIndex < numOfBasePop; copyingIndex++)
   {
      sortedArray[copyingIndex] = pBasePop[copyingIndex][varDimension];
   }

   quickSort(sortedArray, 0, (numOfBasePop - 1));

   /* Good fitness member go to superior sub-pop for current-to-best mutation */
   for (sortedIndex = 0; sortedIndex < subPopSize; sortedIndex++)
   {
      for (int marchingIndex = 0; marchingIndex < numOfBasePop; marchingIndex++) //marching through the base pop to identify matching pop to sorted member
      {
         if (sortedArray[sortedIndex] == pBasePop[marchingIndex][varDimension])
         {
            for (int copyingIndex = 0; copyingIndex <= varDimension; copyingIndex++)
               pSubPop1[sortedIndex][copyingIndex] = pBasePop[marchingIndex][copyingIndex];
            break;
         }
      }
   }

   for (; sortedIndex < (subPopSize+subPopSize); sortedIndex++)
   {
      for (int marchingIndex = 0; marchingIndex < numOfBasePop; marchingIndex++)
      {
         if (sortedArray[sortedIndex] == pBasePop[marchingIndex][varDimension])
         {
            for (int copyingIndex = 0; copyingIndex <= varDimension; copyingIndex++)
               pSubPop2[SUB_POP_POS(sortedIndex, subPopSize)][copyingIndex] = pBasePop[marchingIndex][copyingIndex];
            break;
         }
      }
   }

   /* This sub-pop for rand/1 mutation scheme */
   for (; sortedIndex < (subPopSize+subPopSize+subPopSize); sortedIndex++)
   {
      for (int marchingIndex = 0; marchingIndex < numOfBasePop; marchingIndex++)
      {
         if (sortedArray[sortedIndex] == pBasePop[marchingIndex][varDimension])
         {
            for (int copyingIndex = 0; copyingIndex <= varDimension; copyingIndex++)
               pSubPop3[SUB_POP_POS(sortedIndex, (subPopSize+subPopSize))][copyingIndex] = pBasePop[marchingIndex][copyingIndex];
            break;
         }
      }
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
   int mainIndex, popIndex;
   int j;
   int index = -1, s = 1;
   int subPopSize = numOfPop / 3;
   double **pPop, **pElitePop, **pSubPop2, **pSubNext2, *U2;
   double **pSubPop1, **pSubNext1, *U1;
   double **pSubPop3, **pSubNext3, *U3; 
   double CR = 0.5, F = 0.8, minValue = DBL_MAX;
   double startTime, endTime;

   if (s) INITRAND;

   /* Printing out information about optimization process for the user	*/
   printf("Program parameters: ");
   printf("numOfPop = %d, maxIter = %d, CR = %.2f, F = %.2f\n",
	   numOfPop, iter, CR, F);
   printf("Dimension of the problem: %d\n", varDimension);

   /* Starting timer    */
   startTime = omp_get_wtime();

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

   pSubPop3 = (double **)malloc(subPopSize * sizeof(double *));
   if (NULL == pSubPop3) perror("malloc");
   pSubNext3 = (double **)malloc(subPopSize * sizeof(double *));
   if (NULL == pSubNext3) perror("malloc");

   /* Allocating memory for a trial vector U	*/
   U1 = (double *)malloc((varDimension+1)*sizeof(double));
   if (NULL == U1) perror("malloc");
   U2 = (double *)malloc((varDimension+1)*sizeof(double));
   if (NULL == U2) perror("malloc");
   U3 = (double *)malloc((varDimension+1)*sizeof(double));
   if (NULL == U3) perror("malloc");
   
   for (popIndex = 0; popIndex < numOfPop; popIndex++)
   {
      pPop[popIndex] = (double *)malloc((varDimension+1)*sizeof(double));
      if (NULL == pPop[popIndex]) perror("malloc");

      /* Initialization */
      for (j = 0; j < varDimension; j++)
         pPop[popIndex][j] = problemCtx->lowerConstraints[j] + 
                     (problemCtx->upperConstraints[j] - problemCtx->lowerConstraints[j])*URAND;

      /* Evaluate the fitness for each individual */
      fix(pPop[popIndex], varDimension);
      pPop[popIndex][varDimension] = problemCtx->penaltyFunc(pPop[popIndex]);
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
      pSubPop3[popIndex] = (double *)malloc((varDimension+1)*sizeof(double));
      if (NULL == pSubPop3[popIndex]) perror("malloc");

      pSubNext1[popIndex] = (double *)malloc((varDimension+1)*sizeof(double));
      if (NULL == pSubNext1[popIndex]) perror("malloc");
      pSubNext2[popIndex] = (double *)malloc((varDimension+1)*sizeof(double));
      if (NULL == pSubNext2[popIndex]) perror("malloc");
      pSubNext3[popIndex] = (double *)malloc((varDimension+1)*sizeof(double));
      if (NULL == pSubNext3[popIndex]) perror("malloc");
   }

   applyEliteStrategy(pPop, pSubPop1, pSubPop2, pSubPop3, varDimension, numOfPop);

   /* The main loop of the algorithm	*/
   for (mainIndex = 0; mainIndex < iter; mainIndex++)
   {
      executeEvolutionOverOnePop(problemCtx,
                                 BEST_1,
                                 pSubPop1,
                                 pSubNext1,
                                 U1,
                                 F,
                                 CR,
                                 subPopSize,
                                 varDimension);

      executeEvolutionOverOnePop(problemCtx,
                                 CURRENT_TO_BEST,
                                 pSubPop2,
                                 pSubNext2,
                                 U2,
                                 F,
                                 CR,
                                 subPopSize,
                                 varDimension);            

      executeEvolutionOverOnePop(problemCtx,
                                 RAND_1,
                                 pSubPop3,
                                 pSubNext3,
                                 U3,
                                 F,
                                 CR,
                                 subPopSize,
                                 varDimension);

      /**** Construct elite pop and update sub-pops ****/
      if ((mainIndex%10) == 0)
      {
         /* Merge three sub-pops */
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
         for (int copyingIndex = 0; copyingIndex < subPopSize; copyingIndex++)
         {
            for (int copyingIndex2 = 0; copyingIndex2 <= varDimension; copyingIndex2++)
               pPop[(copyingIndex+subPopSize+subPopSize)][copyingIndex2] = pSubPop3[copyingIndex][copyingIndex2];
         }

         applyEliteStrategy(pPop, pSubPop1, pSubPop2, pSubPop3, varDimension, numOfPop);
         //CR -= 0.01;
         //F -= 0.01;
      }
      
   } /* Main loop */


   /* Calculating and output results */

   /* Stopping timer	*/
   endTime = omp_get_wtime();
   result->executionTime = (endTime - startTime);


   /* Finding best individual	*/
   for (minValue = DBL_MAX, popIndex = 0; popIndex < numOfPop; popIndex++)
   {
      if (pPop[popIndex][varDimension] < minValue)
      {
         minValue = pPop[popIndex][varDimension];
         index = popIndex;
      }
   }

   for (int copyingInx = 0; copyingInx <= varDimension; copyingInx++)
   {
      result->optimizedVars[copyingInx] = pPop[index][copyingInx];
   }
   printf("Solution:\nValues of variables: ");
   for (int i = 0; i < varDimension; i++)
       printf("%.3f\n", pPop[index][i]);


   result->fitnessVal = pPop[index][varDimension];
   result->iteration = mainIndex;

   /* Freeing dynamically allocated memory	*/
   for (popIndex = 0; popIndex < numOfPop; popIndex++)
   {
      free(pPop[popIndex]);
   }
      
   for (popIndex = 0; popIndex < subPopSize; popIndex++)
   {
      free(pSubPop1[popIndex]);
      free(pSubPop2[popIndex]);
      free(pSubPop3[popIndex]);
      free(pSubNext1[popIndex]);
      free(pSubNext2[popIndex]);
      free(pSubNext3[popIndex]);
   }

   free(pPop);
   free(pSubNext1);
   free(pSubNext2);
   free(pSubNext3);
   free(pSubPop1);
   free(pSubPop2);
   free(pSubPop3);
   free(U1);
   free(U2);
   free(U3);
}
