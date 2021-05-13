#include "ae_de.h"
/* Definition for user settings */
/* Definition for a threshold of mutation scheme */

#define THRESHOLD (double)0.01
#define TOLERANCE (double)0.0001
#define NP (int)60
#define MAXITER (int)1500

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


    run_parallel_aeDE(NP, MAXITER, D, &problemDefinitions, &resultStorage, TRUE);

    /* Printing out information about optimization process for the user	*/
    printf("Execution time: %.4f s\n", resultStorage.executionTime);
    printf("Stop at iteration: %d\n", resultStorage.iteration);

    printf("Solution:\nValues of variables: ");
    for (int i = 0; i < D; i++)
       printf("%.15f ", resultStorage.optimizedVars[i]);
    printf("\nObjective function value: ");
    printf("%.15f\n", resultStorage.fitnessVal);

    return 0;
}