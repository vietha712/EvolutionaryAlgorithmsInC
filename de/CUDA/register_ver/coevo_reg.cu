#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include <curand.h>
#include <curand_kernel.h>

using namespace thrust;

#include <stdio.h>
#include <time.h>
#include <fstream>

#define ITERATION 150
#define NOP 300
#define DIVIDER_FOR_MIGRATION_POP 10
#define PRINT_FOR_DRAWING

#include "Utilities.cuh"
#include "TimingGPU.cuh"
#ifdef TRUSS_52BARS_PROBLEM
#include "planar_truss_52bars.cuh"
#define OP_DIMENSION 12
#endif //#ifdef TRUSS_52BARS_PROBLEM
#ifdef TRUSS_10BARS_PROBLEM
#include "planar_truss_10bars.cuh"
#define OP_DIMENSION 10
#endif //#ifdef TRUSS_10BARS_PROBLEM
#ifdef TRUSS_200BARS_PROBLEM
#include "planar_truss_200bars.cuh"
#define OP_DIMENSION 29
#endif
#ifdef TRUSS_72BARS_PROBLEM
#include "spatial_truss_72bars.cuh"
#define OP_DIMENSION 16
#endif //#ifdef TRUSS_72BARS_PROBLEM
#ifdef TRUSS_160BARS_PROBLEM
#include "spatial_truss_160bars.cuh"
#define OP_DIMENSION 38
#endif //#ifdef TRUSS_160BARS_PROBLEM
#define pi 3.14159265358979f

#define BLOCK_SIZE_POP		(16 )
#define BLOCK_SIZE_RAND1	(32 )
#define BLOCK_SIZE_RAND2	(32 )
#define BLOCK_SIZE_UNKN		(OP_DIMENSION	)
#define BLOCK_SIZE			(128)

#define PI_f				3.14159265358979f

#define TIMING
#define DEBUG

#define ELITIST_SELECTION

float createFloatRand( float min, float max )
{
    float scale = rand() / (float) RAND_MAX; /* [0, 1.0] */
    return min + scale * ( max - min );      /* [min, max] */
}



/****************************************/
/* EVALUATION OF THE OBJECTIVE FUNCTION */
/****************************************/
__global__ void curand_setup_kernel(curandState * __restrict state, const unsigned long int seed)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	curand_init(seed, tid, 0, &state[tid]);
}

/********************************/
/* INITIALIZE POPULATION ON GPU */
/********************************/
__global__ void initialize_population_GPU(float * __restrict pop, const float * __restrict minima, const float * __restrict maxima,
	curandState * __restrict state, const int D, const int Np) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i < D) && (j < Np)) pop[j*D + i] = (maxima[i] - minima[i]) * curand_uniform(&state[j*D + i]) + minima[i];
}

/****************************************/
/* EVALUATION OF THE OBJECTIVE FUNCTION */
/****************************************/
#ifdef TRUSS_52BARS_PROBLEM
#define MAXIMA 21612.860f
#define MINIMA 71.613f

__host__ __device__ void fix(float* __restrict X, const int D)
{
    const static float standard_A[64] = {71.613, 90.968, 126.451, 161.29, 198.064, 252.258, 285.161,363.225, 388.386, 494.193,
                               506.451, 641.289, 645.16, 792.256, 816.773, 939.998, 1008.385,
                               1045.159, 1161.288, 1283.868, 1374.191, 1535.481, 1690.319, 1696.771, 1858.061,
                               1890.319, 1993.544, 2019.351, 2180.641, 2238.705, 2290.318, 2341.931, 2477.414,
                               2496.769, 2503.221, 2696.769, 2722.575, 2896.768, 2961.284, 3096.768, 3206.445,
                               3303.219, 3703.218, 4658.055, 5141.925, 5503.215, 5999.988, 6999.986, 7419.340, 8709.660, 8967.724, 9161.272,
                               9999.980, 10322.560, 10903.204, 12129.008, 12838.684, 14193.520, 14774.164, 15806.420, 17096.740, 18064.480, 19354.800, 21612.860}; //Standard cross-sectional areas for design variable in^2

	float temp1, temp2;

    for (int i = 0; i < D; i++)
    {
        for (int j = 0; j < 64; j++)
        {
            if ((X[i] > standard_A[j]))
            {
                continue;
            }
            else
            {
                temp1 = X[i] - standard_A[j];
                temp2 = X[i] - standard_A[j - 1];
                X[i] = (fabs(temp1) <= fabs(temp2)) ? standard_A[j] : standard_A[j - 1];
                break;
            }
        }
    }
}
#endif // #ifdef TRUSS_52BARS_PROBLEM
#ifdef TRUSS_10BARS_PROBLEM
#define MAXIMA 33.5
#define MINIMA 1.62

__host__ __device__ void fix(float* __restrict X, const int D)
{
	const static float standard_A[42] = {1.62, 1.80, 1.99, 2.13, 2.38, 2.62, 2.63, 2.88, 2.93, 3.09, 3.13, 3.38,
                      3.47, 3.55, 3.63, 3.84, 3.87, 3.88, 4.18, 4.22, 4.49, 4.59, 4.80, 4.97,
                      5.12, 5.74, 7.22, 7.97, 11.50, 13.50, 13.90, 14.20, 15.50, 16.00, 16.90,
                      18.80, 19.90, 22.00, 22.90, 26.50, 30.00, 33.50}; //Standard cross-sectional areas for design variable in^2

	float temp1, temp2;

    for (int i = 0; i < D; i++)
    {
        for (int j = 0; j < 42; j++)
        {
            if ((X[i] > standard_A[j]))
            {
                continue;
            }
            else
            {
                temp1 = X[i] - standard_A[j];
                temp2 = X[i] - standard_A[j - 1];
                X[i] = (fabs(temp1) <= fabs(temp2)) ? standard_A[j] : standard_A[j - 1];
                break;
            }
        }
    }
}
#endif
#ifdef TRUSS_200BARS_PROBLEM
#define MAXIMA 33.700f
#define MINIMA 0.100f

__host__ __device__ void fix(float* __restrict X, const int D)
{
	const static float standard_A[30] = {0.100, 0.347, 0.440, 0.539, 0.954, 1.081, 1.174, 1.333, 1.488, 1.764, 2.142, 2.697, 2.800,
                               3.131, 3.565, 3.813, 4.805, 5.952, 6.572, 7.192, 8.525, 9.300, 10.850,
                               13.330, 14.290, 17.170, 19.180, 23.680, 28.080, 33.700}; //Standard cross-sectional areas for design variable in^2

	float temp1, temp2;

    for (int i = 0; i < D; i++)
    {
        for (int j = 0; j < 30; j++)
        {
            if ((X[i] > standard_A[j]))
            {
                continue;
            }
            else
            {
                temp1 = X[i] - standard_A[j];
                temp2 = X[i] - standard_A[j - 1];
                X[i] = (fabs(temp1) <= fabs(temp2)) ? standard_A[j] : standard_A[j - 1];
                break;
            }
        }
    }
}
#endif //#ifdef TRUSS_200BARS_PROBLEM
#ifdef TRUSS_72BARS_PROBLEM
#define MAXIMA 33.5f
#define MINIMA 0.111f

__host__ __device__ void fix(float* __restrict X, const int D)
{
const static float standard_A[64] = {0.111, 0.141, 0.196, 0.25, 0.307, 0.391, 0.442, 0.563, 0.602, 0.766,
                       				 0.785, 0.994, 1.00, 1.228, 1.266, 1.457, 1.563,
                       				 1.62, 1.80, 1.99, 2.13, 2.38, 2.62, 2.63, 2.88, 2.93, 3.09, 3.13, 3.38,
                       				 3.47, 3.55, 3.63, 3.84, 3.87, 3.88, 4.18, 4.22, 4.49, 4.59, 4.80, 4.97,
                       				 5.12, 5.74, 7.22, 7.97, 8.53, 9.3, 10.85, 11.50, 13.50, 13.90, 14.20, 15.50, 16.00, 16.90,
                       				 18.80, 19.90, 22.00, 22.90, 24.5, 26.50, 28.0, 30.00, 33.50}; //Standard cross-sectional areas for design variable in^2

	float temp1, temp2;

    for (int i = 0; i < D; i++)
    {
        for (int j = 0; j < 64; j++)
        {
            if ((X[i] > standard_A[j]))
            {
                continue;
            }
            else
            {
                temp1 = X[i] - standard_A[j];
                temp2 = X[i] - standard_A[j - 1];
                X[i] = (fabs(temp1) <= fabs(temp2)) ? standard_A[j] : standard_A[j - 1];
                break;
            }
        }
    }
}
#endif //#ifdef TRUSS_72BARS_PROBLEM
#ifdef TRUSS_160BARS_PROBLEM
#define MAXIMA 94.13f
#define MINIMA 1.84f

__host__ __device__ void fix(float* __restrict X, const int D)
{
// in cm2
const static float standard_A[42] = {1.84, 2.26, 2.66, 3.07, 3.47, 3.88, 4.79, 5.27, 5.75, 6.25, 6.84, 7.44,
									 8.06, 8.66, 9.40, 10.47, 11.38, 12.21, 13.79, 15.39, 17.03, 19.03, 21.12,
									 23.20, 25.12, 27.50, 29.88, 32.76, 33.90, 34.77, 39.16, 43.00, 45.65, 46.94,
									 51.00, 52.10, 61.82, 61.90, 68.30, 76.38, 90.60, 94.13}; //Standard cross-sectional areas for design variable cm^2

	float temp1, temp2;

    for (int i = 0; i < D; i++)
    {
        for (int j = 0; j < 42; j++)
        {
            if ((X[i] > standard_A[j]))
            {
                continue;
            }
            else
            {
                temp1 = X[i] - standard_A[j];
                temp2 = X[i] - standard_A[j - 1];
                X[i] = (fabs(temp1) <= fabs(temp2)) ? standard_A[j] : standard_A[j - 1];
                break;
            }
        }
    }
}
#endif //#ifdef TRUSS_160BARS_PROBLEM

/********************************/
/* POPULATION EVALUATION ON GPU */
/********************************/
__global__ void evaluation_GPU(const int Np,
							   const int D,
							   float * __restrict pop,
							   float * __restrict fobj,
							   float * d_invK, // this one and below parameters are used for inverse matrix calc
	                           float * d_localLU,
	                           float * d_s) 
{
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	fix(&pop[j*D], D);

	if (j < Np)  fobj[j] = functional(&pop[j*D], D,
	                                  &d_invK[j*TOTAL_DOF*TOTAL_DOF], 
									  &d_localLU[j*TOTAL_DOF*TOTAL_DOF],
									  &d_s[j*TOTAL_DOF*TOTAL_DOF]);
}

__global__ void generation_new_population_mutation_crossover_selection_evaluation_GPU_current_to_best(
	float *__restrict__ pop, const int Np, const int D,
	float *__restrict__ npop, const float F, const float CR,
	const float *__restrict__ minimum, float *__restrict__ maximum,
	float *__restrict__ fobj,
	curandState *__restrict__ state,
	float *d_invK, // this one and below parameters are used for inverse matrix calc
	float *d_localLU,
	float *d_s,
	const int best_old_gen_ind)
{

	int j = threadIdx.x + blockIdx.x * blockDim.x;

	// --- Generate mutation indices and crossover values

	int a, b, c;
	float Rand;

	if (j < Np) {

		// --- Mutation indices
		do a = Np*(curand_uniform(&state[j]));	while (a == j);
		do b = Np*(curand_uniform(&state[j]));	while (b == j || b == a);
		do c = Np*(curand_uniform(&state[j]));	while (c == j || c == a || b == a);

		//// --- Crossover values
		//Rand = curand_uniform(&state[j]);

		// --- Generate new population

		// --- Mutation and crossover
		for (int i = 0; i<D; i++) {
			// --- Crossover values
			Rand = curand_uniform(&state[j]);
			if (Rand < CR) npop[j*D+i] = pop[j*D+i] + F*(pop[best_old_gen_ind*D+i] - pop[j*D+i]) + F*(pop[a*D+i]-pop[b*D+i]);
			else           npop[j*D + i] = pop[j*D + i];
		}

		// --- Saturation due to constraints on the unknown parameters
		for (int i = 0; i<D; i++) if (npop[j*D + i]>maximum[i]) npop[j*D + i] = maximum[i];
		else if (npop[j*D + i]<minimum[i])npop[j*D + i] = minimum[i];

		// --- Evaluation and selection
		fix(&npop[j*D], D);
		float nfobj = functional(&npop[j*D], D,
								 &d_invK[j*TOTAL_DOF*TOTAL_DOF],
								 &d_localLU[j*TOTAL_DOF*TOTAL_DOF],
								 &d_s[j*TOTAL_DOF*TOTAL_DOF]);

		float temp = fobj[j];

		if (nfobj < temp) {
			for (int i = 0; i<D; i++) pop[j*D + i] = npop[j*D + i];
			fobj[j] = nfobj;
		}
	}
}

__global__ void generation_new_population_mutation_crossover_selection_evaluation_GPU_rand_2(
    float * __restrict__ pop, const int Np, const int D,
	float * __restrict__ npop, const float F, const float CR,
	const float * __restrict__ minimum, float * __restrict__ maximum,
	float * __restrict__ fobj,
	curandState * __restrict__ state,
	float * d_invK, // this one and below parameters are used for inverse matrix calc
	float * d_localLU,
	float * d_s) 
{

	int j = threadIdx.x + blockIdx.x * blockDim.x;

	// --- Generate mutation indices and crossover values

	int a, b, c, d, e;
	float Rand;

	if (j < Np) {

		// --- Mutation indices
		do a = Np*(curand_uniform(&state[j]));	while (a == j);
		do b = Np*(curand_uniform(&state[j]));	while (b == j || b == a);
		do c = Np*(curand_uniform(&state[j]));	while (c == j || c == a || b == a);
		do d = Np*(curand_uniform(&state[j]));	while (d == j || d == c || d == b || d == a);
		do e = Np*(curand_uniform(&state[j]));	while (e == j || e == d || e == c || e == b || e == a);

		// --- Generate new population

		// --- Mutation and crossover
		for (int i = 0; i < D; i++) {
			// --- Crossover values
			Rand = curand_uniform(&state[j]);
    		if (Rand < CR)	npop[j*D + i] = pop[e*D+i] + F*(pop[a*D+i]+pop[b*D+i] - pop[c*D+i]-pop[d*D+i]);
		    else			npop[j*D + i] = pop[j*D + i];
		}

		// --- Saturation due to constraints on the unknown parameters
		for (int i = 0; i<D; i++) if (npop[j*D + i]>maximum[i]) npop[j*D + i] = maximum[i];
		else if (npop[j*D + i]<minimum[i])npop[j*D + i] = minimum[i];

		// --- Evaluation and selection
		fix(&npop[j*D], D);
		float nfobj = functional(&npop[j*D], D,
								 &d_invK[j*TOTAL_DOF*TOTAL_DOF],
								 &d_localLU[j*TOTAL_DOF*TOTAL_DOF],
								 &d_s[j*TOTAL_DOF*TOTAL_DOF]);
		float temp = fobj[j];

		if (nfobj < temp) {
			for (int i = 0; i<D; i++) pop[j*D + i] = npop[j*D + i];
			fobj[j] = nfobj;
		}
	}
}

__global__ void generation_new_population_mutation_crossover_selection_evaluation_GPU_best_2(
    float * __restrict__ pop, const int Np, const int D,
	float * __restrict__ npop, const float F, const float CR,
	const float * __restrict__ minimum, float * __restrict__ maximum,
	float * __restrict__ fobj,
	curandState * __restrict__ state,
	float * d_invK, // this one and below parameters are used for inverse matrix calc
	float * d_localLU,
	float * d_s,
    const int best_old_gen_ind) 
{

	int j = threadIdx.x + blockIdx.x * blockDim.x;

	// --- Generate mutation indices and crossover values

	int a, b, c, d;
	float Rand;

	if (j < Np) {

		// --- Mutation indices
		do a = Np*(curand_uniform(&state[j]));	while (a == j);
		do b = Np*(curand_uniform(&state[j]));	while (b == j || b == a);
		do c = Np*(curand_uniform(&state[j]));	while (c == j || c == a || b == a);
		do d = Np*(curand_uniform(&state[j]));	while (d == j || d == c || d == b || d == a);


		// --- Generate new population

		// --- Mutation and crossover
		for (int i = 0; i < D; i++) {
			// --- Crossover values
			Rand = curand_uniform(&state[j]);
    		if (Rand < CR)	npop[j*D+i] = pop[best_old_gen_ind*D+i] + F*(pop[a*D+i]-pop[b*D+i]) + F*(pop[c*D+i]-pop[d*D+i]);
		    else			npop[j*D + i] = pop[j*D + i];
		}

		// --- Saturation due to constraints on the unknown parameters
		for (int i = 0; i<D; i++) if (npop[j*D + i]>maximum[i]) npop[j*D + i] = maximum[i];
		else if (npop[j*D + i]<minimum[i])npop[j*D + i] = minimum[i];

		// --- Evaluation and selection
		fix(&npop[j*D], D);
		float nfobj = functional(&npop[j*D], D,
								 &d_invK[j*TOTAL_DOF*TOTAL_DOF],
								 &d_localLU[j*TOTAL_DOF*TOTAL_DOF],
								 &d_s[j*TOTAL_DOF*TOTAL_DOF]);
		float temp = fobj[j];

		if (nfobj < temp) {
			for (int i = 0; i<D; i++) pop[j*D + i] = npop[j*D + i];
			fobj[j] = nfobj;
		}
	}
}

/***********************/
/* FIND MINIMUM ON GPU */
/***********************/
void find_minimum_GPU(const int N, float *t, float * __restrict minval, int * __restrict index) {

	// --- Wrap raw pointer with a device_ptr 
	device_ptr<float> dev_ptr = device_pointer_cast(t);

	// --- Use device_ptr in thrust min_element
	device_ptr<float> min_ptr = thrust::min_element(dev_ptr, dev_ptr + N);

	index[0] = &min_ptr[0] - &dev_ptr[0];

	minval[0] = min_ptr[0];

}

/***********************/
/* FIND best index     */
/***********************/
int find_best_index(const int N, float *pObjValue)
{
	// --- Wrap raw pointer with a device_ptr
	device_ptr<float> dev_ptr_pObjValue = device_pointer_cast(pObjValue);

	// --- Create the array of indices
	thrust::device_vector<int> d_Idx(N, 0);
	thrust::sequence(d_Idx.begin(), d_Idx.end());

	// --- Use device_ptr in thrust min_element
	thrust::sort_by_key(dev_ptr_pObjValue, dev_ptr_pObjValue + N, d_Idx.data()); //Sort main obj pop

	return d_Idx[0];
}

/*********************************************/
/* Apply elitist members to input population */
/*********************************************/
void applyElitistToPop(float *__restrict elitistObjVal,
					   float *__restrict elitistPop,
					   const int elitistPopSize,
					   float *__restrict copyCurrentObjVal,
					   float *__restrict actualCurrentObjVal,
					   float *__restrict actualCurrentPop,
					   const int popSize,
					   const int D)
{
	// --- Wrap raw pointer with a device_ptr
	device_ptr<float> dev_ptr_pCurrentObjValue = device_pointer_cast(copyCurrentObjVal);

	// --- Create the array of indices
	thrust::device_vector<int> d_Idx(popSize, 0);
	thrust::sequence(d_Idx.begin(), d_Idx.end());

	// --- Sorting with index of objective value
	thrust::sort_by_key(dev_ptr_pCurrentObjValue, dev_ptr_pCurrentObjValue + popSize, d_Idx.data()); //Sort main obj pop

	for(int i = 0; i < elitistPopSize; i++)
	{
		// --- Starting the replacement from the worst member by the best member
		gpuErrchk(cudaMemcpy(&actualCurrentObjVal[d_Idx[popSize - 1 - i]], &elitistObjVal[i], sizeof(float), cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaMemcpy(&actualCurrentPop[d_Idx[popSize - 1 - i]*D], &elitistPop[i*D], D*sizeof(float), cudaMemcpyDeviceToDevice));
	}
}

/***********************/
/* Extract elite pop   */
/***********************/
void extractElitistPop(const int popSize, 
					   const int D,
					   const int numOfMigratedPop,
					   float * __restrict inObjVal,
					   float * __restrict inPop,
					   float * __restrict outObjVal,
					   float * __restrict outPop) {

	// --- Wrap raw pointer with a device_ptr 
	device_ptr<float> dev_ptr_inObjValue = device_pointer_cast(inObjVal);

	// --- Create the array of indices
	thrust::device_vector<int> d_Idx(popSize, 0);
	thrust::sequence(d_Idx.begin(), d_Idx.end());

	// --- Create an union of in

	// --- Use device_ptr in sorting objective values
	thrust::sort_by_key(dev_ptr_inObjValue, dev_ptr_inObjValue + popSize, d_Idx.data()); //Sort main obj pop

	float* raw_ptr_pObjValue = thrust::raw_pointer_cast(&dev_ptr_inObjValue[0]);

	gpuErrchk(cudaMemcpy(&outObjVal[0], &raw_ptr_pObjValue[0], numOfMigratedPop*sizeof(float), cudaMemcpyDeviceToDevice));
	for(int i = 0; i < numOfMigratedPop; i++)
	{
		gpuErrchk(cudaMemcpy(&outPop[i*D], &inPop[d_Idx[i]*D], D*sizeof(float), cudaMemcpyDeviceToDevice));
	}
}

// --- This function will save the value of trial vectors for later elitist selection
__global__ void generation_new_population_mutation_crossover_GPU_current_to_best(
	float *__restrict__ pop, const int Np, const int D,
	float *__restrict__ npop, const float F, const float CR,
	const float *__restrict__ minimum, float *__restrict__ maximum,
	float *__restrict__ fobj,
	curandState *__restrict__ state,
	float *d_invK, // this one and below parameters are used for inverse matrix calc
	float *d_localLU,
	float *d_s,
	const int best_old_gen_ind,
	float *__restrict__ nfobj)
{

	int j = threadIdx.x + blockIdx.x * blockDim.x;

	// --- Generate mutation indices and crossover values

	int a, b, c;
	float Rand;

	if (j < Np) {

		// --- Mutation indices
		do a = Np*(curand_uniform(&state[j]));	while (a == j);
		do b = Np*(curand_uniform(&state[j]));	while (b == j || b == a);
		do c = Np*(curand_uniform(&state[j]));	while (c == j || c == a || b == a);

		//// --- Crossover values
		//Rand = curand_uniform(&state[j]);

		// --- Generate new population

		// --- Mutation and crossover
		for (int i = 0; i<D; i++) {
			// --- Crossover values
			Rand = curand_uniform(&state[j]);
			if (Rand < CR) npop[j*D+i] = pop[j*D+i] + F*(pop[best_old_gen_ind*D+i] - pop[j*D+i]) + F*(pop[a*D+i]-pop[b*D+i]);
			else           npop[j*D + i] = pop[j*D + i];
		}

		// --- Saturation due to constraints on the unknown parameters
		for (int i = 0; i<D; i++) if (npop[j*D + i]>maximum[i]) npop[j*D + i] = maximum[i];
		else if (npop[j*D + i]<minimum[i])npop[j*D + i] = minimum[i];

		// --- Evaluation and selection
		fix(&npop[j*D], D);
		nfobj[j] = functional(&npop[j*D], D,
								 &d_invK[j*TOTAL_DOF*TOTAL_DOF],
								 &d_localLU[j*TOTAL_DOF*TOTAL_DOF],
								 &d_s[j*TOTAL_DOF*TOTAL_DOF]);
	}
}

__global__ void generation_new_population_mutation_crossover_GPU_rand_2(
    float * __restrict__ pop, const int Np, const int D,
	float * __restrict__ npop, const float F, const float CR,
	const float * __restrict__ minimum, float * __restrict__ maximum,
	float * __restrict__ fobj,
	curandState * __restrict__ state,
	float * d_invK, // this one and below parameters are used for inverse matrix calc
	float * d_localLU,
	float * d_s,
	float *__restrict__ nfobj) 
{

	int j = threadIdx.x + blockIdx.x * blockDim.x;

	// --- Generate mutation indices and crossover values

	int a, b, c, d, e;
	float Rand;

	if (j < Np) {

		// --- Mutation indices
		do a = Np*(curand_uniform(&state[j]));	while (a == j);
		do b = Np*(curand_uniform(&state[j]));	while (b == j || b == a);
		do c = Np*(curand_uniform(&state[j]));	while (c == j || c == a || b == a);
		do d = Np*(curand_uniform(&state[j]));	while (d == j || d == c || d == b || d == a);
		do e = Np*(curand_uniform(&state[j]));	while (e == j || e == d || e == c || e == b || e == a);

		// --- Generate new population

		// --- Mutation and crossover
		for (int i = 0; i < D; i++) {
			// --- Crossover values
			Rand = curand_uniform(&state[j]);
    		if (Rand < CR)	npop[j*D + i] = pop[e*D+i] + F*(pop[a*D+i]+pop[b*D+i] - pop[c*D+i]-pop[d*D+i]);
		    else			npop[j*D + i] = pop[j*D + i];
		}

		// --- Saturation due to constraints on the unknown parameters
		for (int i = 0; i<D; i++) if (npop[j*D + i]>maximum[i]) npop[j*D + i] = maximum[i];
		else if (npop[j*D + i]<minimum[i])npop[j*D + i] = minimum[i];

		// --- Evaluation and selection
		fix(&npop[j*D], D);
		nfobj[j] = functional(&npop[j*D], D,
								 &d_invK[j*TOTAL_DOF*TOTAL_DOF],
								 &d_localLU[j*TOTAL_DOF*TOTAL_DOF],
								 &d_s[j*TOTAL_DOF*TOTAL_DOF]);

	}
}

__global__ void generation_new_population_mutation_crossover_GPU_best_2(
    float * __restrict__ pop, const int Np, const int D,
	float * __restrict__ npop, const float F, const float CR,
	const float * __restrict__ minimum, float * __restrict__ maximum,
	float * __restrict__ fobj,
	curandState * __restrict__ state,
	float * d_invK, // this one and below parameters are used for inverse matrix calc
	float * d_localLU,
	float * d_s,
    const int best_old_gen_ind,
	float *__restrict__ nfobj) 
{

	int j = threadIdx.x + blockIdx.x * blockDim.x;

	// --- Generate mutation indices and crossover values

	int a, b, c, d;
	float Rand;

	if (j < Np) {

		// --- Mutation indices
		do a = Np*(curand_uniform(&state[j]));	while (a == j);
		do b = Np*(curand_uniform(&state[j]));	while (b == j || b == a);
		do c = Np*(curand_uniform(&state[j]));	while (c == j || c == a || b == a);
		do d = Np*(curand_uniform(&state[j]));	while (d == j || d == c || d == b || d == a);


		// --- Generate new population

		// --- Mutation and crossover
		for (int i = 0; i < D; i++) {
			// --- Crossover values
			Rand = curand_uniform(&state[j]);
    		if (Rand < CR)	npop[j*D+i] = pop[best_old_gen_ind*D+i] + F*(pop[a*D+i]-pop[b*D+i]) + F*(pop[c*D+i]-pop[d*D+i]);
		    else			npop[j*D + i] = pop[j*D + i];
		}

		// --- Saturation due to constraints on the unknown parameters
		for (int i = 0; i<D; i++) if (npop[j*D + i]>maximum[i]) npop[j*D + i] = maximum[i];
		else if (npop[j*D + i]<minimum[i])npop[j*D + i] = minimum[i];

		// --- Evaluation and selection
		fix(&npop[j*D], D);
		nfobj[j] = functional(&npop[j*D], D,
								 &d_invK[j*TOTAL_DOF*TOTAL_DOF],
								 &d_localLU[j*TOTAL_DOF*TOTAL_DOF],
								 &d_s[j*TOTAL_DOF*TOTAL_DOF]);

	}
}

/*******************************/
/* POPULATION SELECTION ON GPU */
/*******************************/
// --- This is for one input population
void performElitistSelection(const int popSize, const int D,
							 float * __restrict pPop, float * __restrict pNpop, 
							 float * __restrict pFobj, float * __restrict pNfobj,
							 float * __restrict pObjMergeBuffer,
							 float * __restrict pPopMergeBuffer) 
{
	// --- Copy to common buffer to perform sorting
	gpuErrchk(cudaMemcpy(&pObjMergeBuffer[0], &pFobj[0], popSize*sizeof(float), cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpy(&pObjMergeBuffer[popSize], &pNfobj[0], popSize*sizeof(float), cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpy(&pPopMergeBuffer[0], &pPop[0], D*popSize*sizeof(float), cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpy(&pPopMergeBuffer[D*popSize], &pNpop[0], D*popSize*sizeof(float), cudaMemcpyDeviceToDevice));
	// --- Wrap raw pointer with a device_ptr 
	device_ptr<float> dev_ptr_pObjMergeBuffer = device_pointer_cast(pObjMergeBuffer);

	// --- Create the array of indices
	thrust::device_vector<int> d_Idx(2*popSize, 0);
	thrust::sequence(d_Idx.begin(), d_Idx.end());

	// --- Use device_ptr in thrust
	thrust::sort_by_key(dev_ptr_pObjMergeBuffer, dev_ptr_pObjMergeBuffer + (2*popSize), d_Idx.data()); //Sort main obj pop

	float* raw_ptr_pObjValue = thrust::raw_pointer_cast(&dev_ptr_pObjMergeBuffer[0]);

	gpuErrchk(cudaMemcpy(&pFobj[0], &raw_ptr_pObjValue[0], popSize*sizeof(float), cudaMemcpyDeviceToDevice));
	for(int i = 0; i < popSize; i++)
	{
		gpuErrchk(cudaMemcpy(&pPop[i*D], &pPopMergeBuffer[d_Idx[i]*D], D*sizeof(float), cudaMemcpyDeviceToDevice));
	}
}

int usage(char *str)
{
   fprintf(stderr,"Usage: %s [-h] [-u] [-s] [-N NP (20*D)] ", str);
   fprintf(stderr,"[-G Gmax (1000)]\n");
   fprintf(stderr,"\t[-o <outputfile>]\n\n");
   exit(-1);
}


/********/
/* MAIN */
/********/
int main(int argc, char **argv)
{
	// --- Number of individuals in the population (Np >=4 for mutation purposes)
	int			Np = NOP;
	// --- Number of individuals in the sub population (Np >=4 for mutation purposes)
	int			subPopSize = Np/3;
	// --- Number of individuals in the sub population to migrate
	int			numOfMigratePop = subPopSize/DIVIDER_FOR_MIGRATION_POP;
	// --- Dimensionality of each individual (number of unknowns)
	int			D = OP_DIMENSION;

    // --- the rate to perform elitist strategy
	int		    updateRate = 1; // --- Need to be tune depend on Gmax. Tradeoff with performance
#if defined(TRUSS_10BARS_PROBLEM)
	float		F = 0.6f, F2 = 0.3f, F3 = 0.4f;
	// --- Maximum number of generations
	int			Gmax = ITERATION;
	// --- Crossover constant (0 < CR <= 1)
	float		CR = 0.7f, CR2 = 0.3f, CR3 = 0.2f;
	//int *d_best_index;			// --- Device side current optimal member index
#endif
#if defined(TRUSS_72BARS_PROBLEM)
	float		F = 0.6f, F2 = 0.3f, F3 = 0.4f;
	// --- Maximum number of generations
	int			Gmax = ITERATION;
	// --- Crossover constant (0 < CR <= 1)
	float		CR = 0.7f, CR2 = 0.3f, CR3 = 0.2f;
	//int *d_best_index;			// --- Device side current optimal member index
#endif
#if defined(TRUSS_200BARS_PROBLEM)
	float		F = 0.4f, F2 = 0.1f, F3 = 0.4f;
	// --- Maximum number of generations
	int			Gmax = ITERATION;
	// --- Crossover constant (0 < CR <= 1)
	float		CR = 1.0f, CR2 = 0.3f, CR3 = 0.2f;
	//int *d_best_index;			// --- Device side current optimal member index
#endif
#if defined(TRUSS_52BARS_PROBLEM)
	float		F = 0.7f, F2 = 0.2f, F3 = 0.4f;
	// --- Maximum number of generations
	int			Gmax = ITERATION;
	// --- Crossover constant (0 < CR <= 1)
	float		CR = 0.9f, CR2 = 0.8f, CR3 = 0.8f;
#endif
#if defined(TRUSS_160BARS_PROBLEM)
	float		F = 0.45f, F2 = 0.1f, F3 = 0.2f;
	// --- Maximum number of generations
	int			Gmax = ITERATION;
	// --- Crossover constant (0 < CR <= 1)
	float		CR = 0.9f, CR2 = 0.3f, CR3 = 0.2f;
#endif

	// --- Mutually different random integer indices selected from {1, 2, … ,Np}
	int *h_best_index_dev;		// --- Host side current optimal member index of device side

	float *d_subPop_1,				// --- Device side sub-population 1
	  	  *d_subPop_2,				// --- Device side sub-population 2
	  	  *d_subPop_3,				// --- Device side sub-population 3
		  *d_subPop_1_Copy,			// --- Device side sub-population 1
	  	  *d_subPop_2_Copy,			// --- Device side sub-population 2
	  	  *d_subPop_3_Copy,			// --- Device side sub-population 3
	  	  *d_npop_1,				// --- Device side new population 1 (trial vectors)
	  	  *d_npop_2,				// --- Device side new population 2 (trial vectors)
	  	  *d_npop_3,				// --- Device side new population 3 (trial vectors)
	  	  *d_Rand_1,				// --- Device side crossover rand vector (uniformly distributed in (0,1))
	  	  *d_Rand_2,				// --- Device side crossover rand vector (uniformly distributed in (0,1))
	  	  *d_Rand_3,				// --- Device side crossover rand vector (uniformly distributed in (0,1))
	  	  *d_fobj_1,				// --- Device side objective function value
	  	  *d_fobj_2,				// --- Device side objective function value
	  	  *d_fobj_3,				// --- Device side objective function value
#if defined (ELITIST_SELECTION)
		  *d_nfobj_1,				// --- Device side new objective function value
	  	  *d_nfobj_2,				// --- Device side new objective function value
	  	  *d_nfobj_3,				// --- Device side new objective function value
#endif
	  	  *d_fobj_1_Copy,			// --- Device side objective function value
	  	  *d_fobj_2_Copy,			// --- Device side objective function value
	  	  *d_fobj_3_Copy,			// --- Device side objective function value
	  	  *d_maxima_1,				// --- Device side maximum constraints vector
	  	  *d_maxima_2,				// --- Device side maximum constraints vector
	  	  *d_maxima_3,				// --- Device side maximum constraints vector
	  	  *d_minima_1,				// --- Device side minimum constraints vector
	  	  *d_minima_2,				// --- Device side minimum constraints vector
	  	  *d_minima_3;				// --- Device side minimum constraints vector
	int *d_evaluation;
#if defined (ELITIST_SELECTION)
	float *d_objMergeBuffer,
		  *d_popMergeBuffer;
#endif

    float *elitistObj_1,
		  *elitistSubPop_1,
		  *elitistObj_2,
		  *elitistSubPop_2,
		  *elitistObj_3,
		  *elitistSubPop_3,
		  *elitistObj,
		  *elitistSubPop,
		  *mergeElitistObj,
		  *mergeElitistPop;

    float	*h_pop_dev_res_1,		// --- Host side population result of GPU computations
		    *h_pop_dev_res_2,	    // --- Host side population result of GPU computations
		    *h_pop_dev_res_3,	    // --- Host side population result of GPU computations
		    *h_best_dev_1,			// --- Host side population best value history of device side
		    *h_best_dev_2,			// --- Host side population best value history of device side
		    *h_best_dev_3,			// --- Host side population best value history of device side
		    *h_maxima,				// --- Host side maximum constraints vector
		    *h_minima,
		    *h_testBufferObj,
		    *h_testBufferPop;
	int	  *h_best_index_dev_1, *h_best_index_dev_2, *h_best_index_dev_3;
    curandState *devState_1;		// --- Device side random generator state vector
    curandState *devState_2;		// --- Device side random generator state vector
    curandState *devState_3;		// --- Device side random generator state vector
   
   	char filename[] = "160bars_1.txt";
	   char *ofile = NULL;
   	FILE *fid;

   /* Parse command line arguments given by user	*/
   for (int index_1 = 1; index_1 < argc; index_1++)
   {
      if (argv[index_1][0] != '-')
         usage(argv[0]);

      char c = argv[index_1][1];

      switch (c)
      {
         case 'N':
                if (++index_1 >= argc)
                   usage(argv[0]);

		        Np = atoi(argv[index_1]);
                break;
         case 'G':
                if (++index_1 >= argc)
                   usage(argv[0]);

                Gmax = atoi(argv[index_1]);
                break;
         case 'o':
                if (++index_1 >= argc)
                   usage(argv[0]);

		        ofile = argv[index_1];
                break;
         default:
		usage(argv[0]);
      }
   }

	subPopSize = Np/3;
	// --- Number of individuals in the sub population to migrate
	numOfMigratePop = subPopSize/DIVIDER_FOR_MIGRATION_POP;

	// --- Device side memory allocations
	gpuErrchk(cudaMalloc((void**)&d_evaluation, sizeof(int)));

	gpuErrchk(cudaMalloc((void**)&d_subPop_1, D*subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_subPop_2, D*subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_subPop_3, D*subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_subPop_1_Copy, D*subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_subPop_2_Copy, D*subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_subPop_3_Copy, D*subPopSize*sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&d_npop_1, D*subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_npop_2, D*subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_npop_3, D*subPopSize*sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&d_Rand_1, D*subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_Rand_2, D*subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_Rand_3, D*subPopSize*sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&d_fobj_1, subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_fobj_2, subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_fobj_3, subPopSize*sizeof(float)));
#if defined (ELITIST_SELECTION)
	gpuErrchk(cudaMalloc((void**)&d_nfobj_1, subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_nfobj_2, subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_nfobj_3, subPopSize*sizeof(float)));
#endif
	gpuErrchk(cudaMalloc((void**)&d_fobj_1_Copy, subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_fobj_2_Copy, subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_fobj_3_Copy, subPopSize*sizeof(float)));
#if defined (ELITIST_SELECTION)
	gpuErrchk(cudaMalloc((void**)&d_objMergeBuffer, 2*subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_popMergeBuffer, 2*D*subPopSize*sizeof(float)));
#endif
	gpuErrchk(cudaMalloc((void**)&elitistObj_1, numOfMigratePop*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&elitistSubPop_1, D*numOfMigratePop*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&elitistObj_2, numOfMigratePop*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&elitistSubPop_2, D*numOfMigratePop*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&elitistObj_3, numOfMigratePop*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&elitistSubPop_3, D*numOfMigratePop*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&elitistObj, numOfMigratePop*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&elitistSubPop, D*numOfMigratePop*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&mergeElitistObj, 3*numOfMigratePop*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&mergeElitistPop, 3*D*numOfMigratePop*sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&d_maxima_1, D*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_minima_1, D*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_maxima_2, D*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_minima_2, D*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_maxima_3, D*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_minima_3, D*sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&devState_1, D*subPopSize*sizeof(curandState)));
	gpuErrchk(cudaMalloc((void**)&devState_2, D*subPopSize*sizeof(curandState)));
	gpuErrchk(cudaMalloc((void**)&devState_3, D*subPopSize*sizeof(curandState)));
	
    // --- Device memory for matrix calculation
	float *d_invK_1, *d_localLU_1, *d_s_1; // --- Device side matrix calculation
	cudaMalloc((void**)&d_invK_1, subPopSize*TOTAL_DOF*TOTAL_DOF*sizeof(float));
    cudaMalloc((void**)&d_localLU_1, subPopSize*TOTAL_DOF*TOTAL_DOF*sizeof(float));
    cudaMalloc((void**)&d_s_1, subPopSize*TOTAL_DOF*TOTAL_DOF*sizeof(float));

    float *d_invK_2, *d_localLU_2, *d_s_2; // --- Device side matrix calculation
	cudaMalloc((void**)&d_invK_2, subPopSize*TOTAL_DOF*TOTAL_DOF*sizeof(float));
    cudaMalloc((void**)&d_localLU_2, subPopSize*TOTAL_DOF*TOTAL_DOF*sizeof(float));
    cudaMalloc((void**)&d_s_2, subPopSize*TOTAL_DOF*TOTAL_DOF*sizeof(float));

    float *d_invK_3, *d_localLU_3, *d_s_3; // --- Device side matrix calculation
	cudaMalloc((void**)&d_invK_3, subPopSize*TOTAL_DOF*TOTAL_DOF*sizeof(float));
    cudaMalloc((void**)&d_localLU_3, subPopSize*TOTAL_DOF*TOTAL_DOF*sizeof(float));
    cudaMalloc((void**)&d_s_3, subPopSize*TOTAL_DOF*TOTAL_DOF*sizeof(float));

	// --- Host side memory allocations
	h_pop_dev_res_1 = (float*)malloc(D*subPopSize*sizeof(float));
	h_pop_dev_res_2 = (float*)malloc(D*subPopSize*sizeof(float));
	h_pop_dev_res_3 = (float*)malloc(D*subPopSize*sizeof(float));
	h_best_dev_1 = (float*)malloc(Gmax*sizeof(float));
	h_best_dev_2 = (float*)malloc(Gmax*sizeof(float));
	h_best_dev_3 = (float*)malloc(Gmax*sizeof(float));
	h_best_index_dev_1 = (int*)malloc(Gmax*sizeof(int));
	h_best_index_dev_2 = (int*)malloc(Gmax*sizeof(int));
	h_best_index_dev_3 = (int*)malloc(Gmax*sizeof(int));
	h_maxima = (float*)malloc(D*sizeof(float));
	h_minima = (float*)malloc(D*sizeof(float));
	h_testBufferObj = (float*)malloc(Np*sizeof(float));
	h_testBufferPop = (float*)malloc(D*Np*sizeof(float));

	// --- Define grid sizes
	dim3 Grid_1(iDivUp(D, BLOCK_SIZE_UNKN), iDivUp(subPopSize, BLOCK_SIZE_POP));
	dim3 Block_1(BLOCK_SIZE_UNKN, BLOCK_SIZE_POP);
	dim3 Grid_2(iDivUp(D, BLOCK_SIZE_UNKN), iDivUp(subPopSize, BLOCK_SIZE_POP));
	dim3 Block_2(BLOCK_SIZE_UNKN, BLOCK_SIZE_POP);
	dim3 Grid_3(iDivUp(D, BLOCK_SIZE_UNKN), iDivUp(subPopSize, BLOCK_SIZE_POP));
	dim3 Block_3(BLOCK_SIZE_UNKN, BLOCK_SIZE_POP);

	// --- Set maxima and minima
	for (int i = 0; i<D; i++) {
		h_maxima[i] = MAXIMA;
		h_minima[i] = MINIMA;
	}
	gpuErrchk(cudaMemcpy(d_maxima_1, h_maxima, D*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_minima_1, h_minima, D*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_maxima_2, h_maxima, D*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_minima_2, h_minima, D*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_maxima_3, h_maxima, D*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_minima_3, h_minima, D*sizeof(float), cudaMemcpyHostToDevice));

	// --- Initialize cuRAND states
	curand_setup_kernel << <iDivUp(D*subPopSize, BLOCK_SIZE), BLOCK_SIZE >> >(devState_1, time(NULL)^12UL);
	curand_setup_kernel << <iDivUp(D*subPopSize, BLOCK_SIZE), BLOCK_SIZE >> >(devState_2, time(NULL)^21UL);
	curand_setup_kernel << <iDivUp(D*subPopSize, BLOCK_SIZE), BLOCK_SIZE >> >(devState_3, time(NULL)^321UL);

	// --- Initialize popultion
	initialize_population_GPU << <Grid_1, Block_1 >> >(d_subPop_1, d_minima_1, d_maxima_1, devState_1, D, subPopSize);
	initialize_population_GPU << <Grid_2, Block_2 >> >(d_subPop_2, d_minima_2, d_maxima_2, devState_2, D, subPopSize);
	initialize_population_GPU << <Grid_3, Block_3 >> >(d_subPop_3, d_minima_3, d_maxima_3, devState_3, D, subPopSize);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	// --- Evaluate population
	//evaluation_GPU <<<iDivUp(Np, BLOCK_SIZE), BLOCK_SIZE >>>(Np, D, d_pop, d_fobj, d_invK, d_localLU, d_s);
	evaluation_GPU <<<iDivUp(subPopSize, BLOCK_SIZE), BLOCK_SIZE >>>(subPopSize, D, d_subPop_1, d_fobj_1, d_invK_1, d_localLU_1, d_s_1);
	evaluation_GPU <<<iDivUp(subPopSize, BLOCK_SIZE), BLOCK_SIZE >>>(subPopSize, D, d_subPop_2, d_fobj_2, d_invK_2, d_localLU_2, d_s_2);
	evaluation_GPU <<<iDivUp(subPopSize, BLOCK_SIZE), BLOCK_SIZE >>>(subPopSize, D, d_subPop_3, d_fobj_3, d_invK_3, d_localLU_3, d_s_3);

#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
    int bestIndex_1 = 0;
    int bestIndex_3 = 0;
	TimingGPU timerGPU;
    if ((fid=(FILE *)fopen(ofile,"a")) == NULL)
         fprintf(stderr,"Error in opening file %s\n\n",ofile);
	timerGPU.StartCounter();
	for (int i = 1; i <= Gmax; i++) { 
		if (i == 10) updateRate = 2;
		if (i == 20) updateRate = 5;
        gpuErrchk(cudaMemcpy(d_fobj_3_Copy, d_fobj_3, subPopSize*sizeof(float), cudaMemcpyDeviceToDevice));
		bestIndex_3 = find_best_index(subPopSize, d_fobj_3_Copy);
        gpuErrchk(cudaMemcpy(d_fobj_1_Copy, d_fobj_1, subPopSize*sizeof(float), cudaMemcpyDeviceToDevice));
		bestIndex_1 = find_best_index(subPopSize, d_fobj_1_Copy);

#if defined (ELITIST_SELECTION)
		generation_new_population_mutation_crossover_GPU_current_to_best<<<iDivUp(subPopSize,BLOCK_SIZE_POP), BLOCK_SIZE_POP>>>(d_subPop_1,
																				subPopSize, D, d_npop_1, F, CR,
																				d_minima_1, d_maxima_1, d_fobj_1,
																				devState_1, d_invK_1, d_localLU_1, d_s_1, bestIndex_1, d_nfobj_1);
		performElitistSelection(subPopSize, D, d_subPop_1, d_npop_1, d_fobj_1, d_nfobj_1, d_objMergeBuffer, d_popMergeBuffer);

		generation_new_population_mutation_crossover_GPU_rand_2<<<iDivUp(subPopSize,BLOCK_SIZE_POP), BLOCK_SIZE_POP>>>(d_subPop_2,
																				subPopSize, D, d_npop_2, F2, CR2,
																				d_minima_2, d_maxima_2, d_fobj_2,
																				devState_2, d_invK_2, d_localLU_2, d_s_2, d_nfobj_1);
		performElitistSelection(subPopSize, D, d_subPop_2, d_npop_2, d_fobj_2, d_nfobj_1, d_objMergeBuffer, d_popMergeBuffer);

		generation_new_population_mutation_crossover_GPU_best_2<<<iDivUp(subPopSize,BLOCK_SIZE_POP), BLOCK_SIZE_POP>>>(d_subPop_3,
																				subPopSize, D, d_npop_3, F3, CR3,
																				d_minima_3, d_maxima_3, d_fobj_3,
																				devState_3, d_invK_3, d_localLU_3, d_s_3, bestIndex_3,d_nfobj_1);
		performElitistSelection(subPopSize, D, d_subPop_3, d_npop_3, d_fobj_3, d_nfobj_1, d_objMergeBuffer, d_popMergeBuffer);
#else
		generation_new_population_mutation_crossover_selection_evaluation_GPU_current_to_best<<<iDivUp(subPopSize,BLOCK_SIZE_POP), BLOCK_SIZE_POP>>>(d_subPop_1,
																				subPopSize, D, d_npop_1, F, CR,
																				d_minima_1, d_maxima_1, d_fobj_1,
																				devState_1, d_invK_1, d_localLU_1, d_s_1, bestIndex_1);

		generation_new_population_mutation_crossover_selection_evaluation_GPU_rand_2<<<iDivUp(subPopSize,BLOCK_SIZE_POP), BLOCK_SIZE_POP>>>(d_subPop_2,
																				subPopSize, D, d_npop_2, F2, CR2,
																				d_minima_2, d_maxima_2, d_fobj_2,
																				devState_2, d_invK_2, d_localLU_2, d_s_2);

		generation_new_population_mutation_crossover_selection_evaluation_GPU_best_2<<<iDivUp(subPopSize,BLOCK_SIZE_POP), BLOCK_SIZE_POP>>>(d_subPop_3,
																				subPopSize, D, d_npop_3, F3, CR3,
																				d_minima_3, d_maxima_3, d_fobj_3,
																				devState_3, d_invK_3, d_localLU_3, d_s_3, bestIndex_3);
#endif

        if(i%updateRate == 0)
        {
            // --- Copy to temp buffer before extracting elitist pop
			gpuErrchk(cudaMemcpy(d_fobj_1_Copy, d_fobj_1, subPopSize*sizeof(float), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(d_subPop_1_Copy, d_subPop_1, D*subPopSize*sizeof(float), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(d_fobj_2_Copy, d_fobj_2, subPopSize*sizeof(float), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(d_subPop_2_Copy, d_subPop_2, D*subPopSize*sizeof(float), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(d_fobj_3_Copy, d_fobj_3, subPopSize*sizeof(float), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(d_subPop_3_Copy, d_subPop_3, D*subPopSize*sizeof(float), cudaMemcpyDeviceToDevice));

			// --- Extract OK match
			extractElitistPop(subPopSize, D, numOfMigratePop, d_fobj_1_Copy, d_subPop_1_Copy, elitistObj_1, elitistSubPop_1);
			extractElitistPop(subPopSize, D, numOfMigratePop, d_fobj_2_Copy, d_subPop_2_Copy, elitistObj_2, elitistSubPop_2);
			extractElitistPop(subPopSize, D, numOfMigratePop, d_fobj_3_Copy, d_subPop_3_Copy, elitistObj_3, elitistSubPop_3);

			// --- Merge and finalize elitist pop - Success
			gpuErrchk(cudaMemcpy(mergeElitistObj, elitistObj_1, numOfMigratePop*sizeof(float), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(&mergeElitistObj[numOfMigratePop], elitistObj_2, numOfMigratePop*sizeof(float), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(&mergeElitistObj[numOfMigratePop + numOfMigratePop], elitistObj_3, numOfMigratePop*sizeof(float), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(mergeElitistPop, elitistSubPop_1, D*numOfMigratePop*sizeof(float), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(&mergeElitistPop[D*numOfMigratePop], elitistSubPop_2, D*numOfMigratePop*sizeof(float), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(&mergeElitistPop[D*(numOfMigratePop+numOfMigratePop)], elitistSubPop_3, D*numOfMigratePop*sizeof(float), cudaMemcpyDeviceToDevice));
			
			// --- Extract the best out of the merged elitist pop (Reuse elitistObj_1 for storage)
			extractElitistPop(3*numOfMigratePop, D, numOfMigratePop, mergeElitistObj, mergeElitistPop, elitistObj_1, elitistSubPop_1);

			// --- Apply finalized elitist pop to sub pop
			// 1. Create copy of current popolation obj value.
			gpuErrchk(cudaMemcpy(d_fobj_1_Copy, d_fobj_1, subPopSize*sizeof(float), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(d_fobj_2_Copy, d_fobj_2, subPopSize*sizeof(float), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(d_fobj_3_Copy, d_fobj_3, subPopSize*sizeof(float), cudaMemcpyDeviceToDevice));
			// 2. Sorting current population.
			// 3. Apply extracted members to replace the most worst members in current pop.
			applyElitistToPop(elitistObj_1, elitistSubPop_1, numOfMigratePop, d_fobj_1_Copy, d_fobj_1, d_subPop_1, subPopSize, D);
			applyElitistToPop(elitistObj_1, elitistSubPop_1, numOfMigratePop, d_fobj_2_Copy, d_fobj_2, d_subPop_2, subPopSize, D);
			applyElitistToPop(elitistObj_1, elitistSubPop_1, numOfMigratePop, d_fobj_3_Copy, d_fobj_3, d_subPop_3, subPopSize, D);
        }
		F = createFloatRand(0.3f, 0.7f); CR = createFloatRand(0.6f, 1.0f);
		F2 = createFloatRand(0.05f, 0.5f); CR2 = createFloatRand(0.2f, 0.7f);
		F3 = createFloatRand(0.2f, 0.7f); CR3 = createFloatRand(0.1f, 0.6f);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		find_minimum_GPU(subPopSize, d_fobj_1, &h_best_dev_1[i], &h_best_index_dev_1[i]);
		find_minimum_GPU(subPopSize, d_fobj_2, &h_best_dev_2[i], &h_best_index_dev_2[i]);
		find_minimum_GPU(subPopSize, d_fobj_3, &h_best_dev_3[i], &h_best_index_dev_3[i]);

    	gpuErrchk(cudaMemcpy(h_testBufferObj, d_fobj_1, subPopSize*sizeof(float), cudaMemcpyDeviceToHost));
    	gpuErrchk(cudaMemcpy(&h_testBufferObj[subPopSize], d_fobj_2, subPopSize*sizeof(float), cudaMemcpyDeviceToHost));
    	gpuErrchk(cudaMemcpy(&h_testBufferObj[subPopSize + subPopSize], d_fobj_3, subPopSize*sizeof(float), cudaMemcpyDeviceToHost));

#ifdef PRINT_FOR_DRAWING
		for (int index = 0; index < Np; index++)
		{
			fprintf(fid, "%.3f\n",h_testBufferObj[index]);
		}
#else
		for (int index = 0; index < Np; index+=3)
		{
			fprintf(fid,"No. %d: %.3f - No.%d: %.3f - No. %d: %.3f\n", index+i*Np-Np,     h_testBufferObj[index], 
																	   index+i*Np-Np + 1, h_testBufferObj[index+1], 
																	   index+i*Np-Np + 2, h_testBufferObj[index+2]);
		}
#endif

#ifndef PRINT_FOR_DRAWING
#ifdef TIMING
		//printf("Iteration: %i; best member value: %f - %f - %f: best member index: %i - %i - %i\n", i, h_best_dev_1[i], h_best_dev_2[i], h_best_dev_3[i], h_best_index_dev_1[i], h_best_index_dev_2[i], h_best_index_dev_3[i]);
		fprintf(fid, "%f - %f - %f at %d\n", h_best_dev_1[i], h_best_dev_2[i], h_best_dev_3[i], i);
#endif
#endif
	}
#ifdef TIMING
	fprintf(fid, "Total timing = %f [s]\n", timerGPU.GetCounter() * 0.001);
#endif// TIMING
    find_minimum_GPU(subPopSize, d_fobj_1, &h_best_dev_1[Gmax - 1], &h_best_index_dev_1[Gmax - 1]);
	find_minimum_GPU(subPopSize, d_fobj_2, &h_best_dev_2[Gmax - 1], &h_best_index_dev_2[Gmax - 1]);
	find_minimum_GPU(subPopSize, d_fobj_3, &h_best_dev_3[Gmax - 1], &h_best_index_dev_3[Gmax - 1]);
    gpuErrchk(cudaMemcpy(h_pop_dev_res_1, d_subPop_1, D*subPopSize*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_pop_dev_res_2, d_subPop_2, D*subPopSize*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_pop_dev_res_3, d_subPop_3, D*subPopSize*sizeof(float), cudaMemcpyDeviceToHost));

	printf("Obj = %.3f - Obj = %.3f - Obj = %.3f\n", h_best_dev_1[Gmax - 1], h_best_dev_2[Gmax - 1], h_best_dev_3[Gmax - 1]);
	for(int x = 0; x < D; x++)
	{
		fprintf(fid, "var[%d] = %.3f\n", x, h_pop_dev_res_1[h_best_index_dev_1[Gmax - 1]*D + x]);
	}
	for(int x = 0; x < D; x++)
	{
		printf("var[%d] = %.3f\n", x, h_pop_dev_res_2[h_best_index_dev_2[Gmax - 1]*D + x]);
	}
	for(int x = 0; x < D; x++)
	{
		printf("var[%d] = %.3f\n", x, h_pop_dev_res_3[h_best_index_dev_3[Gmax - 1]*D + x]);
	}
#ifdef PRINT_FOR_DRAWING
	fprintf(fid, "\n");
	for (int index = 0; index < Np*ITERATION; index++)
	{
		fprintf(fid, "%d\n", index);
	}

#endif
	fclose(fid);
	return 0;
}
