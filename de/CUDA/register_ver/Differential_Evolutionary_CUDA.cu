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

#define TRUSS_200BARS_PROBLEM

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
#else
#include "matrix.cuh"
#endif //#ifdef TRUSS_200BARS_PROBLEM
#ifdef TRUSS_72BARS_PROBLEM
#include "planar_truss_72bars.cuh"
#define OP_DIMENSION 16
#endif //#ifdef TRUSS_72BARS_PROBLEM
#define pi 3.14159265358979f

#define BLOCK_SIZE_POP		(16 )
#define BLOCK_SIZE_RAND1	(32 )
#define BLOCK_SIZE_RAND2	(32 )
#define BLOCK_SIZE_UNKN		(OP_DIMENSION	)
#define BLOCK_SIZE			(128)

#define PI_f				3.14159265358979f

#define TIMING
//#define SHARED_VERSION
#define REGISTER_VERSION

#define DEBUG

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
#ifndef TRUSS_200BARS_PROBLEM
__global__ void evaluation_GPU(const int Np,
							   const int D,
							   float * __restrict pop,
							   float * __restrict fobj) 
{
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	fix(&pop[j*D], D);

	if (j < Np)  fobj[j] = functional(&pop[j*D], D);
}
#else
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
#endif //#ifndef TRUSS_200BARS_PROBLEM

#ifndef REGISTER_VERSION
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
#endif //REGISTER_VERSION
/*******************************/
/* POPULATION SELECTION ON GPU */
/*******************************/
// Assumption: all the optimization variables are associated to the same thread block
#ifndef REGISTER_VERSION
#ifndef ANTENNAS
__global__ void selection_and_evaluation_GPU(const int Np,
											 const int D,
											 float * __restrict pop,
											 float * __restrict npop,
											 float * __restrict fobj) 
{
	int i = threadIdx.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i < D) && (j < Np)) {
		fix(&npop[j*D], D);

		float nfobj = functional(&npop[j*D], D);

		float temp = fobj[j];

		if (nfobj < temp) {
			pop[j*D + i] = npop[j*D + i];
			fobj[j] = nfobj;
		}
	}
}
#else
// Assumption: all the optimization variables are associated to the same thread block
__global__ void selection_and_evaluation_GPU(int Np, int D, float *pop, float *npop, float *fobj, int N, float d, float beta, float Deltau) {

	int i = threadIdx.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i < D) && (j < Np)) {

		float nfobj = functional(&npop[j*D], D, N, d, beta, Deltau);

		float temp = fobj[j];

		if (nfobj < temp) {
			pop[j*D + i] = npop[j*D + i];
			fobj[j] = nfobj;
		}
	}
}
#endif
#endif // #ifndef REGISTER_VERSION

#ifdef REGISTER_VERSION
#ifndef TRUSS_200BARS_PROBLEM
__global__ void generation_new_population_mutation_crossover_selection_evaluation_GPU(float * __restrict__ pop, const int Np, const int D,
	float * __restrict__ npop, const float F, const float CR,
	const float * __restrict__ minimum, float * __restrict__ maximum,
	float * __restrict__ fobj,
	curandState * __restrict__ state) 
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
		//if (Rand < CR)	for (int i = 0; i<D; i++) npop[j*D + i] = pop[a*D + i] + F*(pop[b*D + i] - pop[c*D + i]);
		//else			for (int i = 0; i<D; i++) npop[j*D + i] = pop[j*D + i];
		for (int i = 0; i<D; i++) {
			// --- Crossover values
			Rand = curand_uniform(&state[j]);
			if (Rand < CR) npop[j*D + i] = pop[a*D + i] + F*(pop[b*D + i] - pop[c*D + i]);
			else           npop[j*D + i] = pop[j*D + i];
		}

		// --- Saturation due to constraints on the unknown parameters
		for (int i = 0; i<D; i++) if (npop[j*D + i]>maximum[i]) npop[j*D + i] = maximum[i];
		else if (npop[j*D + i]<minimum[i])npop[j*D + i] = minimum[i];

		// --- Evaluation and selection
		fix(&npop[j*D], D);
		float nfobj = functional(&npop[j*D], D);

		float temp = fobj[j];

		if (nfobj < temp) {
			for (int i = 0; i<D; i++) pop[j*D + i] = npop[j*D + i];
			fobj[j] = nfobj;
		}

	}
}
#else //#ifndef TRUSS_200BARS_PROBLEM
__global__ void generation_new_population_mutation_crossover_selection_evaluation_GPU(float * __restrict__ pop, const int Np, const int D,
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
		//if (Rand < CR)	for (int i = 0; i<D; i++) npop[j*D + i] = pop[a*D + i] + F*(pop[b*D + i] - pop[c*D + i]);
		//else			for (int i = 0; i<D; i++) npop[j*D + i] = pop[j*D + i];
		for (int i = 0; i<D; i++) {
			// --- Crossover values
			Rand = curand_uniform(&state[j]);
			if (Rand < CR) npop[j*D + i] = pop[a*D + i] + F*(pop[b*D + i] - pop[c*D + i]);
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
#endif
#endif

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
	int			Np = 100;
	// --- Dimensionality of each individual (number of unknowns)
	int			D = OP_DIMENSION;
	// --- Mutation factor (0 < F <= 2). Typically chosen in [0.5, 1], see Ref. [1]
	float		F = 0.5f;
	// --- Maximum number of generations
	int			Gmax = 400;
	// --- Crossover constant (0 < CR <= 1)
	float		CR = 0.3f;

	// --- Mutually different random integer indices selected from {1, 2, … ,Np}
	int *d_mutation,			// --- Device side mutation vector
		*h_best_index_dev;		// --- Host side current optimal member index of device side
	//int *d_best_index;			// --- Device side current optimal member index

#ifdef ANTENNAS
	// --- Wavelength
	float		lambda = 1.f;
	// --- Interelement distance
	float		d = lambda / 2.f;
	// --- Wavenumber
	float		beta = 2.f*pi / lambda;
	// --- Spectral oversampling factor
	float		overs = 4.f;
	// --- Sampling step in the spectral domain
	float		Deltau = pi / (overs*(D - 1)*d);
	// --- Number of spectral sampling points
	int			N = floor(4 * (D - 1)*d*overs / lambda);
#endif

	float *d_pop,				// --- Device side population
		*d_npop,					// --- Device side new population (trial vectors)
		*d_Rand,					// --- Device side crossover rand vector (uniformly distributed in (0,1))
		*d_fobj,					// --- Device side objective function value
		*d_maxima,					// --- Device side maximum constraints vector
		*d_minima,					// --- Device side minimum constraints vector
		*h_pop_dev_res,				// --- Host side population result of GPU computations
		*h_best_dev,				// --- Host side population best value history of device side
		*h_maxima,					// --- Host side maximum constraints vector
		*h_minima,
		*h_testBufferObj,
		*h_testBufferPop;			// --- Host side minimum constraints vector
	curandState *devState;		// --- Device side random generator state vector

	// --- Device side memory allocations
	gpuErrchk(cudaMalloc((void**)&d_pop, D*Np*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_npop, D*Np*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_Rand, D*Np*sizeof(float)));
	//gpuErrchk(cudaMalloc((void**)&d_Rand, Np*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_fobj, Np*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_mutation, 3 * Np*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_maxima, D*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_minima, D*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&devState, D*Np*sizeof(curandState)));
	// --- Device memory for matrix calculation
#if defined(TRUSS_200BARS_PROBLEM)
	float *d_invK, *d_localLU, *d_s; // --- Device side matrix calculation
	cudaMalloc((void**)&d_invK, Np*TOTAL_DOF*TOTAL_DOF*sizeof(float));
    cudaMalloc((void**)&d_localLU, Np*TOTAL_DOF*TOTAL_DOF*sizeof(float));
    cudaMalloc((void**)&d_s, Np*TOTAL_DOF*TOTAL_DOF*sizeof(float));
#endif
	//cudaMalloc((void**)&d_temp_nobj, Np*sizeof(float));

	// --- Host side memory allocations
	h_pop_dev_res = (float*)malloc(D*Np*sizeof(float));
	h_best_dev = (float*)malloc(Gmax*sizeof(float));
	h_best_index_dev = (int*)malloc(Gmax*sizeof(int));
	h_maxima = (float*)malloc(D*sizeof(float));
	h_minima = (float*)malloc(D*sizeof(float));
	h_testBufferObj = (float*)malloc(Np*sizeof(float));
	h_testBufferPop = (float*)malloc(D*Np*sizeof(float));

	// --- Define grid sizes
	int Num_Blocks_Pop = iDivUp(Np, BLOCK_SIZE_POP);
	dim3 Grid(iDivUp(D, BLOCK_SIZE_UNKN), iDivUp(Np, BLOCK_SIZE_POP));
	dim3 Block(BLOCK_SIZE_UNKN, BLOCK_SIZE_POP);

	// --- Set maxima and minima
	for (int i = 0; i<D; i++) {
		h_maxima[i] = MAXIMA;
		h_minima[i] = MINIMA;
	}
	gpuErrchk(cudaMemcpy(d_maxima, h_maxima, D*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_minima, h_minima, D*sizeof(float), cudaMemcpyHostToDevice));

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

	// --- Initialize cuRAND states
	curand_setup_kernel << <iDivUp(D*Np, BLOCK_SIZE), BLOCK_SIZE >> >(devState, time(NULL));

	// --- Initialize popultion
	initialize_population_GPU << <Grid, Block >> >(d_pop, d_minima, d_maxima, devState, D, Np);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	// --- Evaluate population
#ifndef ANTENNAS
#if defined(TRUSS_200BARS_PROBLEM)
	evaluation_GPU <<<iDivUp(Np, BLOCK_SIZE), BLOCK_SIZE >>>(Np, D, d_pop, d_fobj, d_invK, d_localLU, d_s);
#else
	evaluation_GPU <<<iDivUp(Np, BLOCK_SIZE), BLOCK_SIZE >>>(Np, D, d_pop, d_fobj);
#endif
	printf("Grid size line 802 <<<%d, %d>>>\n", iDivUp(Np, BLOCK_SIZE), BLOCK_SIZE);
#else
	evaluation_GPU<<<Num_Blocks_Pop,BLOCK_SIZE_POP>>>(Np, D, d_pop, d_fobj, N, d, beta, Deltau);
#endif
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
	if ((fid=(FILE *)fopen(ofile,"a")) == NULL) fprintf(stderr,"Error in opening file %s\n\n",ofile);
	TimingGPU timerGPU;
	timerGPU.StartCounter();
	for (int i = 1; i < Gmax; i++) {
#ifdef SHARED_VERSION
		generation_new_population_mutation_crossover_selection_evaluation_GPU << <Grid, Block >> >(d_pop, Np, D, d_npop, F, CR, d_minima, d_maxima, d_fobj, devState);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
#elif defined REGISTER_VERSION
#if defined(TRUSS_200BARS_PROBLEM)
		generation_new_population_mutation_crossover_selection_evaluation_GPU<<<iDivUp(Np,BLOCK_SIZE_POP), BLOCK_SIZE_POP>>>(d_pop,
																				Np, D, d_npop, F, CR,
																				d_minima, d_maxima, d_fobj,
																				devState, d_invK, d_localLU, d_s);
#else
		generation_new_population_mutation_crossover_selection_evaluation_GPU<<<iDivUp(Np,BLOCK_SIZE_POP), BLOCK_SIZE_POP>>>(d_pop,
																				Np, D, d_npop, F, CR,
																				d_minima, d_maxima, d_fobj,
																				devState);
#endif
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
#else
		// --- Generate mutation indices 
		generate_mutation_indices_GPU << <iDivUp(Np, BLOCK_SIZE_RAND1), BLOCK_SIZE_RAND1 >> >(d_mutation, Np, D, devState);
		//generate_mutation_indices_GPU_withRand2<< <iDivUp(Np, BLOCK_SIZE_RAND1), BLOCK_SIZE_RAND1 >> >(d_mutation, Np, D, devState);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// --- Generate crossover values 
		generate_crossover_values_GPU << <iDivUp(D * Np, BLOCK_SIZE_RAND2), BLOCK_SIZE_RAND2 >> >(d_Rand, Np, D, devState);
		
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// --- Generate new population
		generation_new_population_GPU << <Grid, Block >> >(d_pop, Np, D, d_npop, F, CR, d_Rand, d_mutation, d_minima, d_maxima);
		//generation_new_population_GPU_withRand2<< <Grid, Block >> >(d_pop, Np, D, d_npop, F, CR, d_Rand, d_mutation, d_minima, d_maxima);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// --- Select new population and evaluate it
#ifndef ANTENNAS
		selection_and_evaluation_GPU << <Grid, Block >> >(Np, D, d_pop, d_npop, d_fobj);
#else
		selection_and_evaluation_GPU << <Grid, Block >> >(Np, D, d_pop, d_npop, d_fobj, N, d, beta, Deltau);
#endif
/*
	if( i == 200)
	{
		gpuErrchk(cudaMemcpy(h_testBufferObj, d_fobj, Np*sizeof(float), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(h_testBufferPop, d_pop, D*Np*sizeof(float), cudaMemcpyDeviceToHost));
		for(int z = 0; z < Np; z++)
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
#endif
		find_minimum_GPU(Np, d_fobj, &h_best_dev[i], &h_best_index_dev[i]);
		//printf("aaaa = %f\n", h_best_dev[i]);

#ifdef TIMING
		printf("Iteration: %i; best member value: %f: best member index: %i\n", i, h_best_dev[i], h_best_index_dev[i]);
#endif

	}
#ifdef TIMING
	printf("Total timing = %f [s]\n", timerGPU.GetCounter() * 0.001);
#endif TIMING

	gpuErrchk(cudaMemcpy(h_pop_dev_res, d_pop, D*Np*sizeof(float), cudaMemcpyDeviceToHost));
	for (int i = 0; i<D; i++) printf("Variable nr. %i = %.4f\n", i, h_pop_dev_res[(Gmax-1) * D + i]);
	printf("Objective value: = %.4f\n", h_best_dev[Gmax - 1]);

	return 0;
}
