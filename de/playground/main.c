#include "ae_de.h"
/* Definition for user settings */
/* Definition for a threshold of mutation scheme */

#define THRESHOLD (double)0.00001
#define TOLERANCE (double)0.000001
#define NP (int)20
#define MAXITER (int)4000

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
    resultStorage.numOfEvals = 0;


    run_parallel_aeDE(NP, MAXITER, THRESHOLD, TOLERANCE, D, &problemDefinitions, &resultStorage, TRUE);

    /* Printing out information about optimization process for the user	*/
    printf("Execution time: %.4f s\n", resultStorage.executionTime);
    printf("Number of objective function evaluations: %d\n", resultStorage.numOfEvals);

    printf("Solution:\nValues of variables: ");
    for (int i = 0; i < D; i++)
       printf("%.15f ", resultStorage.optimizedVars[i]);
    printf("\nObjective function value: ");
    printf("%.15f\n", resultStorage.fitnessVal);

    return 0;
}