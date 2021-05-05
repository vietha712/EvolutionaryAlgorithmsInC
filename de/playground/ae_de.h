#ifndef _AE_DE_H_
#define _AE_DE_H_

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <float.h>
#include <math.h>

/*
 * Description: this is a implementation of an adaptive elitist differential evolution
 * for discrete variables and tailor to truss optimization.
 * 
 */

/***************************** Defines *************************************/
/* Random number generator defined by URAND should return
   double-precision floating-point values uniformly distributed
   over the interval [0.0, 1.0)					*/
#define MAX_DIMENSION 20
#define URAND	((double)rand()/((double)RAND_MAX + 1.0))

#define FRAND  (((double)rand()/(double)RAND_MAX) * 0.6 + 0.4) // [0.4 to 1]

#define CRRAND  (((double)rand()/(double)RAND_MAX) * 0.3 + 0.7) // [0.7 to 1]

#define FALSE 0

#define TRUE 1

/* Data structure */
typedef struct problemTag
{
    double upperConstraints[MAX_DIMENSION];
    double lowerConstraints[MAX_DIMENSION];
    double (*penaltyFunc)(double *);
}problemT;

typedef struct resultTag
{
    double optimizedVars[MAX_DIMENSION];
    int numOfEvals;
    double executionTime;
    double fitnessVal;
}resultT;

typedef enum mutationSchemeTag
{
    RAND_1,
    CURRENT_TO_BEST,
}mutationSchemeT;


/************************* Exported interface ******************************/
void run_aeDE(int numOfPop, 
              int iter, 
              float threshHold, 
              float tolerance,
              int varDimension,
              problemT *problemCtx,
              resultT *result,
              int isMinimized);

void run_parallel_aeDE(int numOfPop,
                       int iter,
                       int varDimension,
                       problemT *problemCtx,
                       resultT *result,
                       int isMinimized);

#endif