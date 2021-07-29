#include "ae_de.h"
/* Definition for user settings */
/* Definition for a threshold of mutation scheme */

#define NP (int)600
#define MAXITER (int)20000
#define THRESHOLD 0.001
#define TOLE 0.00001
#define UPDATE_RATE 2000
#define DECREMENT_VALUE 0.01f

double func(double *);

int main(void)
{
    extern double Xl[], Xu[];
    extern int D;
    problemT problemDefinitions;
    resultT resultStorage;

    for (int i = 0; i < D; i++)
    {
        problemDefinitions.lowerConstraints[i] = Xl[i];
        problemDefinitions.upperConstraints[i] = Xu[i];
        resultStorage.optimizedVars[i] = 0.0;
    }
    problemDefinitions.penaltyFunc = &func;

    resultStorage.executionTime = 0.0;
    resultStorage.fitnessVal = 0.0;
    resultStorage.iteration = 0;

#if 1
    run_parallel_aeDE(NP, MAXITER, D, &problemDefinitions, &resultStorage, UPDATE_RATE, DECREMENT_VALUE, TRUE);
#endif

#if 0
    run_aeDE(NP, MAXITER, THRESHOLD, TOLE, D, &problemDefinitions, &resultStorage);
#endif

    /* Printing out information about optimization process for the user	*/
    printf("Execution time: %.6f s\n", resultStorage.executionTime);
    printf("Stop at iteration: %d\n", resultStorage.iteration);

    printf("Solution:\nValues of variables: ");
    for (int i = 0; i < D; i++)
       printf("%.15f ", resultStorage.optimizedVars[i]);
    printf("\nObjective function value: ");
    printf("%.15f\n", resultStorage.fitnessVal);

    return 0;
}