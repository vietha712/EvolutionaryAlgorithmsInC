#include "ae_de.h"
/* Definition for user settings */
/* Definition for a threshold of mutation scheme */

#define THRESHOLD (double)0.00001
#define TOLERANCE (double)0.000001
#define NP (int)20
#define MAXITER (int)4000


int main(void)
{
    extern double Xl[], Xu[];
    extern int D;
    problemT problemDefinitions;

    for (int i = 0; i < D; i++)
    {
        problemDefinitions.lowerConstraints[i] = Xl[i];
        problemDefinitions.upperConstraints[i] = Xu[i];
        problemDefinitions.upperConstraints[i] = 0;
    }

    problemDefinitions.penaltyFunc = &func();


    run_aeDE(NP, MAXITER, THRESHOLD, TOLERANCE, D, );

    return 0;
}