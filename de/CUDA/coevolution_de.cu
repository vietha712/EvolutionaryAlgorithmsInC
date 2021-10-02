#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include <curand.h>
#include <curand_kernel.h>

using namespace thrust;

#include <stdio.h>
#include <time.h>
#include <fstream>

#define TRUSS_52BARS_PROBLEM

#include "Utilities.cuh"
#include "TimingGPU.cuh"
#ifdef TRUSS_52BARS_PROBLEM
#include "planar_truss_52bars.cuh"
#define OP_DIMENSION 12
#endif
#ifdef TRUSS_10BARS_PROBLEM
#include "planar_truss_10bars.cuh"
#define OP_DIMENSION 10
#endif
#ifdef TRUSS_72BARS_PROBLEM
#include "planar_truss_72bars.cuh"
#define OP_DIMENSION 16
#endif
#define pi 3.14159265358979f

#if defined(TRUSS_72BARS_PROBLEM)
#define BLOCK_SIZE_POP		(16 )
#define BLOCK_SIZE_RAND1	(64 )
#define BLOCK_SIZE_RAND2	(64 )
#define BLOCK_SIZE_UNKN		(OP_DIMENSION)
#define BLOCK_SIZE			(256)
#endif
#if defined(TRUSS_52BARS_PROBLEM)
#define BLOCK_SIZE_POP		(32 )
#define BLOCK_SIZE_RAND1	(64 )
#define BLOCK_SIZE_RAND2	(64 )
#define BLOCK_SIZE_UNKN		(OP_DIMENSION)
#define BLOCK_SIZE			(256)
#endif
#if defined(TRUSS_10BARS_PROBLEM)
#define BLOCK_SIZE_POP		(32 )
#define BLOCK_SIZE_RAND1	(128 )
#define BLOCK_SIZE_RAND2	(128 )
#define BLOCK_SIZE_UNKN		(OP_DIMENSION)
#define BLOCK_SIZE			(512)
#endif

#define PI_f				3.14159265358979f

#define TIMING
//#define SHARED_VERSION
//#define REGISTER_VERSION

//#define DEBUG

//#define ANTENNAS

// --- REFERENCES
//     [1] R. Storn and K. Price, “Differential evolution – a simple and efficient heuristic for global optimization over continuous spaces,” 
//     Journal of Global Optimization, vol. 11, no. 4, pp. 341–359, 1997

//     [2] Lucas de P. Veronese and Renato A. Krohling, “Differential Evolution Algorithm on the GPU with C-CUDA,” 
//     Proc. of the IEEE Congress on Evolutionary Computation, Barcelona, Spain, Jul. 18-23, 2010, pp. 1-7.

// Conventions: the index j addresses the population member while the index i addresses the member component
//              the homologous host and device variables have the same name with a "h_" or "d_" prefix, respectively
//				the __host__ and __device__ functions pointer parameters have the same name for comparison purposes. it is up to the caller to use 
//				host or device pointers, as appropriate

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
#endif
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
#endif
#if 0
#ifndef ANTENNAS
__host__ __device__ float functional(const float * __restrict x, const int D) {

	// --- More functionals at https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume24/ortizboyer05a-html/node6.html
	
	float sum;
	// --- De Jong function (hypersphere)
	//#define MINIMA -5.12
	//#define MAXIMA  5.12
	//sum = 0.f;
	//for (int i=0; i<D; i++) sum = sum + x[i] * x[i];
	// --- Rosenbrock's saddle - xopt = (1., 1., ..., 1.)
	//#define MINIMA -2.048
	//#define MAXIMA  2.048
	//sum = 0.f;
	//for (int i = 1; i<D; i++) sum = sum + 100.f * (x[i] - x[i - 1] * x[i - 1]) * (x[i] - x[i - 1] * x[i - 1]) + (x[i - 1] - 1.f) * (x[i - 1] - 1.f);
	// --- Rastrigin - xopt = (0., 0., ..., 0.)
	//#define MINIMA -5.12
	//#define MAXIMA  5.12
	//sum = 10.f * D;
	//for (int i = 1; i <= D; i++) sum = sum + (x[i - 1] * x[i - 1] - 10.f * cos(2.f * PI_f * x[i - 1]));
	// --- Schwfel - xopt(-420.9698, -420.9698, ..., -420.9698)
	#define MINIMA -500.0
	#define MAXIMA  500.0
	sum = 418.9829 * D;
	for (int i = 1; i <= D; i++) sum = sum + x[i - 1] * sin(sqrt(fabs(x[i - 1])));

	return sum;
}
#else
#define MINIMA -PI_f
#define MAXIMA  PI_f
__host__ __device__ float cheb(const float x, const int N) {

	if (fabs(x) <= 1.f) return cos(N * acos(x));
	else				return cosh(N * acosh(x));

}

__host__ __device__ float pattern(const float u, const float beta, const int N) {

	const float temp = cheb(u / (beta * 0.3), N);

	return 1.f / sqrt(1.f + 0.1 * fabs(temp) * fabs(temp));

}

__host__ __device__ float functional(float* x, int D, int N, float d, float beta, float Deltau) {

	// --- Functional value
	float sum = 0.f;

	// --- Spectral variable
	float u;

	// --- Real and imaginary parts of the array factor and squared absolute value
	float Fr, Fi, F2, Frref, Firef, F2ref;
	// --- Reference pattern (ASSUMED real for simplicity!)
	float R;
	// --- Maximum absolute value of the array factor
	float maxF = -FLT_MAX;

	// --- Calculating the array factor and the maximum of its absolute value
	for (int i = 0; i<N; i++) {
		u = -beta + i * Deltau;
		Fr = Fi = 0.;
		Frref = Firef = 0.;

		for (int j = 0; j<D; j++) {
			Fr = Fr + cos(j * u * d + x[j]);
			Fi = Fi + sin(j * u * d + x[j]);
		}
		F2 = Fr * Fr + Fi * Fi;
		//F2ref = (3.f * cos(u / (0.5*beta)) * cos(u / (0.5*beta)) * cos(u / (0.5*beta))) * (3.f * cos(u / (0.5*beta)));
		F2ref = (3.f * cos((u - 0.1*beta) / (0.5*beta)) * cos((u - 0.1*beta) / (0.5*beta)) * cos((u - 0.1*beta) / (0.5*beta))) * (3.f * cos((u - 0.1*beta) / (0.5*beta)));
		//F2ref = 2.f * pattern(u, beta, N);
		//F2ref = F2ref * F2ref;
		sum = sum + (F2 - F2ref) * (F2 - F2ref);
	}

	return sum;
}
#endif
#endif
/********************************/
/* POPULATION EVALUATION ON GPU */
/********************************/
#ifndef ANTENNAS
__global__ void evaluation_GPU(const int Np, const int D, float * __restrict pop, float * __restrict fobj) {

	int j = threadIdx.x + blockIdx.x * blockDim.x;
	fix(&pop[j*D], D);
	if (j < Np)  fobj[j] = function(&pop[j*D], D);
}
#else
__global__ void evaluation_GPU(int Np, int D, float *pop, float *fobj, int N, float d, float beta, float Deltau) {

	int j = threadIdx.x + blockIdx.x * blockDim.x;

	if (j < Np) fobj[j] = functional(&pop[j*D], D, N, d, beta, Deltau);

}
#endif
/**********************************************************/
/* GENERATE MUTATION INDICES AND CROSS-OVER VALUES ON GPU */
/**********************************************************/
__global__ void generate_crossover_values_GPU(float * __restrict Rand, const int Np, const int D, curandState * __restrict state) {

	int j = threadIdx.x + blockIdx.x * blockDim.x;

	//if (j < D * Np) Rand[j] = curand_uniform(&state[j*Np]);
	if (j < D * Np) Rand[j] = curand_uniform(&state[j]);
}

/**********************************************************/
/* GENERATE MUTATION INDICES AND CROSS-OVER VALUES ON GPU */
/**********************************************************/
__global__ void generate_mutation_indices_GPU(int * __restrict mutation, const int Np, const int D, curandState * __restrict state) {

	int j = threadIdx.x + blockIdx.x * blockDim.x;

	int a, b, c;

	if (j < Np) {

		//do a = Np*(curand_uniform(&state[j*D]));	while (a == j);
		//do b = Np*(curand_uniform(&state[j*D]));	while (b == j || b == a);
		//do c = Np*(curand_uniform(&state[j*D]));	while (c == j || c == a || c == b);
		do a = Np*(curand_uniform(&state[j]));	while (a == j);
		do b = Np*(curand_uniform(&state[j]));	while (b == j || b == a);
		do c = Np*(curand_uniform(&state[j]));	while (c == j || c == a || c == b);
		mutation[j * 3] = a;
		mutation[j * 3 + 1] = b;
		mutation[j * 3 + 2] = c;

	}
}

__global__ void generate_mutation_indices_GPU_withRand2(int * __restrict mutation, const int Np, const int D, curandState * __restrict state) {

	int j = threadIdx.x + blockIdx.x * blockDim.x;

	int a, b, c, d, e;

	if (j < Np) 
	{
		do a = Np*(curand_uniform(&state[j]));	while (a == j);
		do b = Np*(curand_uniform(&state[j]));	while (b == j || b == a);
		do c = Np*(curand_uniform(&state[j]));	while (c == j || c == a || c == b);
		do d = Np*(curand_uniform(&state[j]));	while (d == j || d == c || d == b || d == a);
		do e = Np*(curand_uniform(&state[j]));	while (e == j || e == d || e == c || e == b || e == a);

		mutation[j * 5] = a;
		mutation[j * 5 + 1] = b;
		mutation[j * 5 + 2] = c;
		mutation[j * 5 + 3] = d;
		mutation[j * 5 + 4] = e;
	}
}

__global__ void generate_mutation_indices_GPU_withBest2(int * __restrict mutation, const int Np, const int D, curandState * __restrict state) {

	int j = threadIdx.x + blockIdx.x * blockDim.x;

	int a, b, c, d;

	if (j < Np) 
	{
		do a = Np*(curand_uniform(&state[j]));	while (a == j);
		do b = Np*(curand_uniform(&state[j]));	while (b == j || b == a);
		do c = Np*(curand_uniform(&state[j]));	while (c == j || c == a || c == b);
		do d = Np*(curand_uniform(&state[j]));	while (d == j || d == a || d == b || d == c);

		mutation[j * 4] = a;
		mutation[j * 4 + 1] = b;
		mutation[j * 4 + 2] = c;
		mutation[j * 4 + 3] = d;
	}
}

/**********************************/
/* GENERATION OF A NEW POPULATION */
/**********************************/
__global__ void generation_new_population_GPU(const float * __restrict pop, const int NP, const int D, float * __restrict npop, const float F,
	const float CR, const float * __restrict rand, const int * __restrict mutation,
	const float * __restrict minimum, const float * __restrict maximum) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i < D) && (j < NP)) {

		// --- Mutation indices
		int a = mutation[j * 3];
		int b = mutation[j * 3 + 1];
		int c = mutation[j * 3 + 2];

		// --- Mutation and binomial crossover
		// --- DE/rand/1. One of the best strategies. Try F = 0.7 and CR = 0.5 as a first guess.
		//if (rand[j*D + i]<CR)	npop[j*D + i] = pop[a*D + i] + F*(pop[b*D + i] - pop[c*D + i]);
		if (rand[j]<CR)	npop[j*D + i] = pop[a*D + i] + F*(pop[b*D + i] - pop[c*D + i]);
		else			npop[j*D + i] = pop[j*D + i];
		//printf("%f\n", npop[j*D + i]);

		// --- Other possible approaches to mutation and crossover
		// --- DE/best/1 --- Not bad, but found several optimization problems where misconvergence occurs.
		//npop[j*D+i] = pop[best_old_gen_ind*D+i] + F*(pop[b*D+i]-pop[c*D+i]);
		// --- DE/rand to best/1 --- F1 can be different or equal to F2
		//npop[j*D+i] = pop[a*D+i] + F1*(pop[best_old_gen_ind*D+i] - pop[a*D+i]) + F2*(pop[b*D+i]-pop[c*D+i]);
		// --- DE/current to best/1 --- One of the best strategies. Try F = 0.85 and CR = 1. In case of misconvergence, try to increase NP. If this doesn't help,
		//     play around with all the control variables --- F1 can be different or equal to F2
		//npop[j*D+i] = pop[j*D+i] + F1*(pop[best_old_gen_ind*D+i] - pop[j*D+i]) + F2*(pop[a*D+i]-pop[b*D+i]);
		// --- DE/rand/2 --- Robust optimizer for many functions.
		//npop[j*D+i] = pop[e*D+i] + F*(pop[a*D+i]+pop[b*D+i]-pop[c*D+i]-pop[d*D+i]);
		// --- DE/best/2 --- Powerful strategy worth trying.
		//npop[j*D+i] = pop[best_old_gen_ind*D+i] + F*(pop[a*D+i]+pop[b*D+i]-pop[c*D+i]-pop[d*D+i]);

		// --- Saturation due to constraints on the unknown parameters
		if (npop[j*D + i]>maximum[i])		npop[j*D + i] = maximum[i];
		else if (npop[j*D + i]<minimum[i])	npop[j*D + i] = minimum[i];

	}

}

__global__ void generation_new_population_GPU_withRand2(const float * __restrict pop, 
											  			const int NP, 
											  			const int D, 
											  			float * __restrict npop, 
											  			const float F,
											  			const float CR, 
											  			const float * __restrict rand, 
											  			const int * __restrict mutation,
											  			const float * __restrict minimum, 
											  			const float * __restrict maximum) 
{

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i < D) && (j < NP)) {

		// --- Mutation indices
		int a = mutation[j * 5];
		int b = mutation[j * 5 + 1];
		int c = mutation[j * 5 + 2];
		int d = mutation[j * 5 + 3];
		int e = mutation[j * 5 + 4];

		// --- Mutation and binomial crossover
		if (rand[j]<CR)	npop[j*D + i] = pop[e*D+i] + F*(pop[a*D+i]+pop[b*D+i]-pop[c*D+i]-pop[d*D+i]);
		else			npop[j*D + i] = pop[j*D + i];
		//printf("%f\n", npop[j*D + i]);

		// --- Saturation due to constraints on the unknown parameters
		if (npop[j*D + i]>maximum[i])		npop[j*D + i] = maximum[i];
		else if (npop[j*D + i]<minimum[i])	npop[j*D + i] = minimum[i];

		// For truss problem the fix() should be run here.

	}
}

__global__ void generation_new_population_GPU_withBest2(const float * __restrict pop, 
											  			const int NP, 
											  			const int D,
											  			float * __restrict npop, 
											  			const float F,
											  			const float CR, 
											  			const float * __restrict rand, 
											  			const int * __restrict mutation,
											  			const float * __restrict minimum, 
											  			const float * __restrict maximum,
														const int best_old_gen_ind) 
{

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i < D) && (j < NP)) {

		// --- Mutation indices
		int a = mutation[j * 4];
		int b = mutation[j * 4 + 1];
		int c = mutation[j * 4 + 2];
		int d = mutation[j * 4 + 3];

		// --- Mutation and binomial crossover
		// --- DE/best/2 --- Powerful strategy worth trying.
		if (rand[j]<CR)	npop[j*D+i] = pop[best_old_gen_ind*D+i] + F*(pop[a*D+i]-pop[b*D+i]) + F*(pop[c*D+i]-pop[d*D+i]);
		else			npop[j*D + i] = pop[j*D + i];
		//printf("%f\n", npop[j*D + i]);
		

		// --- Saturation due to constraints on the unknown parameters
		if (npop[j*D + i]>maximum[i])		npop[j*D + i] = maximum[i];
		else if (npop[j*D + i]<minimum[i])	npop[j*D + i] = minimum[i];

		// For truss problem the fix() should be run here.

	}
}

/*******************************/
/* POPULATION SELECTION ON GPU */
/*******************************/
// Assumption: all the optimization variables are associated to the same thread block

__global__ void selection_and_evaluation_GPU(const int Np, const int D, float * __restrict pop, float * __restrict npop, float * __restrict fobj) {

	int i = threadIdx.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i < D) && (j < Np)) {
		fix(&npop[j*D], D);
		float nfobj = function(&npop[j*D], D);

		float temp = fobj[j];

		if (nfobj < temp) {
			pop[j*D + i] = npop[j*D + i];
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

/***********************/
/* Perform migration   */
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

	// --- Use device_ptr in thrust min_element
	thrust::sort_by_key(dev_ptr_inObjValue, dev_ptr_inObjValue + popSize, d_Idx.data()); //Sort main obj pop

	float* raw_ptr_pObjValue = thrust::raw_pointer_cast(&dev_ptr_inObjValue[0]);

	gpuErrchk(cudaMemcpy(&outObjVal[0], &raw_ptr_pObjValue[0], numOfMigratedPop*sizeof(float), cudaMemcpyDeviceToDevice));
	for(int i = 0; i < numOfMigratedPop; i++)
	{
		for(int j = 0; j < D; j++)
		{
			gpuErrchk(cudaMemcpy(&outPop[i*D], &inPop[d_Idx[i]*D], D*sizeof(float), cudaMemcpyDeviceToDevice));
		}
	}
}

/**************************/
/* Construct elitist Pop  */
/**************************/
/*
void constructElitistPop(const int inputSize, 
					 	 const int D,
					 	 const int outputSize,
					 	 float * __restrict pElitistObj_1,
					 	 float * __restrict pElitistPop_1,
					 	 float * __restrict pElitistObj_2,
					 	 float * __restrict pElitistPop_2,
					 	 float * __restrict pElitistObj_3,
					 	 float * __restrict pElitistPop_3,
					 	 float * __restrict pFinalObj,
					 	 float * __restrict pFinalPop) 
{

}*/

/********/
/* MAIN */
/********/
int main()
{
	// --- Number of individuals in the population (Np >=4 for mutation purposes)
	int			Np = 120;
	// --- Number of individuals in the sub population (Np >=4 for mutation purposes)
	int			subPopSize = Np/3;
	// --- Number of individuals in the sub population to migrate
	int			numOfMigratePop = subPopSize/10;
	// --- Dimensionality of each individual (number of unknowns)
	int			D = OP_DIMENSION;
	// --- Mutation factor (0 < F <= 2). Typically chosen in [0.5, 1], see Ref. [1]
#if defined(TRUSS_52BARS_PROBLEM)
	float		F = 0.4f, F2 = 0.2f, F3 = 0.4f;
	// --- Maximum number of generations
	int			Gmax = 250;
	// --- Crossover constant (0 < CR <= 1)
	float		CR = 0.6f, CR2 = 0.8f, CR3 = 0.8f;
#endif
#if defined(TRUSS_72BARS_PROBLEM)
	float		F = 0.6f, F2 = 0.3f, F3 = 0.4f;
	// --- Maximum number of generations
	int			Gmax = 200;
	// --- Crossover constant (0 < CR <= 1)
	float		CR = 0.7f, CR2 = 0.3f, CR3 = 0.2f;
	//int *d_best_index;			// --- Device side current optimal member index
#endif
#if defined(TRUSS_10BARS_PROBLEM)
	float		F = 0.6f, F2 = 0.3f, F3 = 0.4f;
	// --- Maximum number of generations
	int			Gmax = 250;
	// --- Crossover constant (0 < CR <= 1)
	float		CR = 0.7f, CR2 = 0.3f, CR3 = 0.2f;
	//int *d_best_index;			// --- Device side current optimal member index
#endif
	float *h_pop_dev_res_1,			// --- Host side population result of GPU computations
		  *h_pop_dev_res_2,			// --- Host side population result of GPU computations
		  *h_pop_dev_res_3,			// --- Host side population result of GPU computations
		  *h_best_dev_1,			// --- Host side population best value history of device side
		  *h_best_dev_2,			// --- Host side population best value history of device side
		  *h_best_dev_3,			// --- Host side population best value history of device side
		  *h_maxima,				// --- Host side maximum constraints vector
		  *h_minima,				// --- Host side minimum constraints vector
		  *h_testBufferObj,
		  *h_testBufferPop;
	int	  *h_best_index_dev_1, *h_best_index_dev_2, *h_best_index_dev_3;
	// --- the rate to perform elitist strategy
	int		    updateRate = 10; // --- Need to be tune depend on Gmax. Tradeoff with performance
	//curandState *devState;		// --- Device side random generator state vector

	// --- Coevolution
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
	  	  *d_fobj_1_Copy,			// --- Device side objective function value
	  	  *d_fobj_2_Copy,			// --- Device side objective function value
	  	  *d_fobj_3_Copy,			// --- Device side objective function value
	  	  *d_maxima_1,				// --- Device side maximum constraints vector
	  	  *d_maxima_2,				// --- Device side maximum constraints vector
	  	  *d_maxima_3,				// --- Device side maximum constraints vector
	  	  *d_minima_1,				// --- Device side minimum constraints vector
	  	  *d_minima_2,				// --- Device side minimum constraints vector
	  	  *d_minima_3;				// --- Device side minimum constraints vector

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

	int *d_mutation_1,				// --- Device side mutation vector for subpop 1
		*d_mutation_2,				// --- Device side mutation vector for subpop 2
		*d_mutation_3;				// --- Device side mutation vector for subpop 3

	curandState *devState_1;		// --- Device side random generator state vector
	curandState *devState_2;		// --- Device side random generator state vector
	curandState *devState_3;		// --- Device side random generator state vector

	// --- Coevo Device side mem alloc
	gpuErrchk(cudaMalloc((void**)&d_subPop_1, D*subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_subPop_2, D*subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_subPop_3, D*subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_subPop_1_Copy, D*subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_subPop_2_Copy, D*subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_subPop_3_Copy, D*subPopSize*sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&d_fobj_1, subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_fobj_2, subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_fobj_3, subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_fobj_1_Copy, subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_fobj_2_Copy, subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_fobj_3_Copy, subPopSize*sizeof(float)));

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

	gpuErrchk(cudaMalloc((void**)&d_npop_1, D*subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_npop_2, D*subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_npop_3, D*subPopSize*sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&d_Rand_1, D*subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_Rand_2, D*subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_Rand_3, D*subPopSize*sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&d_mutation_1, 3 * subPopSize * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_mutation_2, 5 * subPopSize * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_mutation_3, 4 * subPopSize * sizeof(int)));

	gpuErrchk(cudaMalloc((void**)&d_maxima_1, D*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_minima_1, D*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_maxima_2, D*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_minima_2, D*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_maxima_3, D*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_minima_3, D*sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&devState_1, D*subPopSize*sizeof(curandState)));
	gpuErrchk(cudaMalloc((void**)&devState_2, D*subPopSize*sizeof(curandState)));
	gpuErrchk(cudaMalloc((void**)&devState_3, D*subPopSize*sizeof(curandState)));

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
	h_testBufferObj = (float*)malloc(subPopSize*sizeof(float));
	h_testBufferPop = (float*)malloc(D*subPopSize*sizeof(float));

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
	curand_setup_kernel << <iDivUp(D*subPopSize, BLOCK_SIZE), BLOCK_SIZE >> >(devState_1, time(NULL)^15467855035453992347UL);
	curand_setup_kernel << <iDivUp(D*subPopSize, BLOCK_SIZE), BLOCK_SIZE >> >(devState_2, time(NULL)^17523746072705851699UL);
	curand_setup_kernel << <iDivUp(D*subPopSize, BLOCK_SIZE), BLOCK_SIZE >> >(devState_3, time(NULL)^12956554043759569915UL);

	// --- Initialize popultion
	initialize_population_GPU << <Grid_1, Block_1 >> >(d_subPop_1, d_minima_1, d_maxima_1, devState_1, D, subPopSize);
	initialize_population_GPU << <Grid_2, Block_2 >> >(d_subPop_2, d_minima_2, d_maxima_2, devState_2, D, subPopSize);
	initialize_population_GPU << <Grid_3, Block_3 >> >(d_subPop_3, d_minima_3, d_maxima_3, devState_3, D, subPopSize);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	// --- Evaluate population
	//evaluation_GPU <<<iDivUp(Np, BLOCK_SIZE), BLOCK_SIZE >>>(Np, D, d_pop, d_fobj);
	evaluation_GPU <<<iDivUp(subPopSize, BLOCK_SIZE), BLOCK_SIZE >>>(subPopSize, D, d_subPop_1, d_fobj_1);
	evaluation_GPU <<<iDivUp(subPopSize, BLOCK_SIZE), BLOCK_SIZE >>>(subPopSize, D, d_subPop_2, d_fobj_2);
	evaluation_GPU <<<iDivUp(subPopSize, BLOCK_SIZE), BLOCK_SIZE >>>(subPopSize, D, d_subPop_3, d_fobj_3);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
	int bestIndex_3 = 0;
	TimingGPU timerGPU;
	timerGPU.StartCounter();
	for (int i = 1; i < Gmax; i++) {

		gpuErrchk(cudaMemcpy(d_fobj_3_Copy, d_fobj_3, subPopSize*sizeof(float), cudaMemcpyDeviceToDevice));
		bestIndex_3 = find_best_index(subPopSize, d_fobj_3_Copy);

		// --- Generate mutation indices 
		generate_mutation_indices_GPU << <iDivUp(subPopSize, BLOCK_SIZE_RAND1), BLOCK_SIZE_RAND1 >> >(d_mutation_1, subPopSize, D, devState_1);
		//generate_mutation_indices_GPU << <iDivUp(subPopSize, BLOCK_SIZE_RAND1), BLOCK_SIZE_RAND1 >> >(d_mutation_2, subPopSize, D, devState_2);
		generate_mutation_indices_GPU_withRand2 << <iDivUp(subPopSize, BLOCK_SIZE_RAND1), BLOCK_SIZE_RAND1 >> >(d_mutation_2, subPopSize, D, devState_2);
		//generate_mutation_indices_GPU << <iDivUp(subPopSize, BLOCK_SIZE_RAND1), BLOCK_SIZE_RAND1 >> >(d_mutation_3, subPopSize, D, devState_3);
		generate_mutation_indices_GPU_withBest2 << <iDivUp(subPopSize, BLOCK_SIZE_RAND1), BLOCK_SIZE_RAND1 >> >(d_mutation_3, subPopSize, D, devState_3);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// --- Generate crossover values 
		generate_crossover_values_GPU << <iDivUp(D * subPopSize, BLOCK_SIZE_RAND2), BLOCK_SIZE_RAND2 >> >(d_Rand_1, subPopSize, D, devState_1);
		generate_crossover_values_GPU << <iDivUp(D * subPopSize, BLOCK_SIZE_RAND2), BLOCK_SIZE_RAND2 >> >(d_Rand_2, subPopSize, D, devState_2);
		generate_crossover_values_GPU << <iDivUp(D * subPopSize, BLOCK_SIZE_RAND2), BLOCK_SIZE_RAND2 >> >(d_Rand_3, subPopSize, D, devState_3);
		
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// --- Generate new population
		generation_new_population_GPU << <Grid_1, Block_1 >> >(d_subPop_1, subPopSize, D, d_npop_1, F, CR, d_Rand_1, d_mutation_1, d_minima_1, d_maxima_1);
		//generation_new_population_GPU << <Grid_2, Block_2 >> >(d_subPop_2, subPopSize, D, d_npop_2, F, CR, d_Rand_2, d_mutation_2, d_minima_2, d_maxima_2);
		generation_new_population_GPU_withRand2 << <Grid_2, Block_2 >> >(d_subPop_2, subPopSize, D, d_npop_2, F2, CR2, d_Rand_2, d_mutation_2, d_minima_2, d_maxima_2);
		//generation_new_population_GPU << <Grid_3, Block_3 >> >(d_subPop_3, subPopSize, D, d_npop_3, F, CR, d_Rand_3, d_mutation_3, d_minima_3, d_maxima_3);
		generation_new_population_GPU_withBest2<< <Grid_3, Block_3 >> >(d_subPop_3, subPopSize, D, d_npop_3, F3, CR3, d_Rand_3, d_mutation_3, d_minima_3, d_maxima_3, bestIndex_3);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// --- Select new population and evaluate it
		// --- 		selection_and_evaluation_GPU << <Grid, Block >> >(Np, D, d_pop, d_npop, d_fobj);
		selection_and_evaluation_GPU << <Grid_1, Block_1 >> >(subPopSize, D, d_subPop_1, d_npop_1, d_fobj_1);
		selection_and_evaluation_GPU << <Grid_2, Block_2 >> >(subPopSize, D, d_subPop_2, d_npop_2, d_fobj_2);
		selection_and_evaluation_GPU << <Grid_3, Block_3 >> >(subPopSize, D, d_subPop_3, d_npop_3, d_fobj_3);

		// --- OK NP -> 300 to 1800 -> design vars and obj are matched
/*
		if (i == 600)
		{
			gpuErrchk(cudaMemcpy(h_testBufferObj, d_fobj_3, subPopSize*sizeof(float), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(h_testBufferPop, d_subPop_3, D*subPopSize*sizeof(float), cudaMemcpyDeviceToHost));
			for(int z = 0; z < subPopSize; z++)
			{
				printf("Obj[%d] = %f\n", z, h_testBufferObj[z]);
				for(int x = 0; x < D; x++)
				{
					printf("var[%d] of Obj[%d] = %f\n", x, z, h_testBufferPop[z*D + x]);
				}
			}
			return 0;
		}
*/
		
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
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
			
			// --- Extract the best out of the merged elitist pop - OK
			extractElitistPop(3*numOfMigratePop, D, numOfMigratePop, mergeElitistObj, mergeElitistPop, elitistObj_1, elitistSubPop_1);

			// --- Apply finalized elitist pop to sub pop
			int indexForReplacement = subPopSize - numOfMigratePop;
			gpuErrchk(cudaMemcpy(&d_fobj_1[indexForReplacement], elitistObj_1, numOfMigratePop*sizeof(float), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(&d_subPop_1[D*indexForReplacement], elitistSubPop_1, D*numOfMigratePop*sizeof(float), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(&d_fobj_2[indexForReplacement], elitistObj_1, numOfMigratePop*sizeof(float), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(&d_subPop_2[D*indexForReplacement], elitistSubPop_1, D*numOfMigratePop*sizeof(float), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(&d_fobj_3[indexForReplacement], elitistObj_1, numOfMigratePop*sizeof(float), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(&d_subPop_3[D*indexForReplacement], elitistSubPop_1, D*numOfMigratePop*sizeof(float), cudaMemcpyDeviceToDevice));


			//gpuErrchk(cudaMemcpy(h_testBufferObj, d_fobj_1, subPopSize*sizeof(float), cudaMemcpyDeviceToHost));
			//gpuErrchk(cudaMemcpy(h_testBufferPop, d_subPop_1, D*subPopSize*sizeof(float), cudaMemcpyDeviceToHost));
			//for(int z = 0; z < subPopSize; z++)
			//{
			//	printf("Obj[%d] = %f\n", z, h_testBufferObj[z]);
			//	for(int x = 0; x < D; x++)
			//	{
			//		printf("var[%d] of Obj[%d] = %f\n", x, z, h_testBufferPop[z*D + x]);
			//	}
			//}
			//return 0;
		}

		find_minimum_GPU(subPopSize, d_fobj_1, &h_best_dev_1[i], &h_best_index_dev_1[i]);
		find_minimum_GPU(subPopSize, d_fobj_2, &h_best_dev_2[i], &h_best_index_dev_2[i]);
		find_minimum_GPU(subPopSize, d_fobj_3, &h_best_dev_3[i], &h_best_index_dev_3[i]);

#ifdef TIMING
		printf("Iteration: %i; best member value: %f - %f - %f: best member index: %i - %i - %i\n", i, h_best_dev_1[i], h_best_dev_2[i], h_best_dev_3[i], h_best_index_dev_1[i], h_best_index_dev_2[i], h_best_index_dev_3[i]);
#endif

	}
#ifdef TIMING
	printf("Total timing = %f [s]\n", timerGPU.GetCounter() * 0.001);
#endif TIMING

	//gpuErrchk(cudaMemcpy(h_pop_dev_res, d_pop, D*Np*sizeof(float), cudaMemcpyDeviceToHost));
	//for (int i = 0; i<D; i++) printf("Variable nr. %i = %.4f\n", i, h_pop_dev_res[(Gmax-1) * D + i]);
	//printf("Objective value: = %.4f\n", h_best_dev[Gmax - 1]);

	//gpuErrchk(cudaMemcpy(h_testBufferObj, d_fobj_1, subPopSize*sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_testBufferPop, d_subPop_1, D*subPopSize*sizeof(float), cudaMemcpyDeviceToHost));

	printf("Obj = %.3f\n", h_best_dev_1[Gmax - 1]);
	for(int x = 0; x < D; x++)
	{
		printf("var[%d] = %.3f\n", x, h_testBufferPop[h_best_index_dev_1[Gmax - 1]*D + x]);
	}

	return 0;
}
