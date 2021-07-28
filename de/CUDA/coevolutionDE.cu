#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

#include <curand.h>
#include <curand_kernel.h>

using namespace thrust;

#include <stdio.h>
#include <time.h>
#include <fstream>

#include "Utilities.cuh"
#include "TimingGPU.cuh"

#define pi 3.14159265358979f

#define BLOCK_SIZE_POP		(32  * 2)
#define BLOCK_SIZE_RAND1	(64  * 2)
#define BLOCK_SIZE_RAND2	(64  * 2)
#define BLOCK_SIZE_UNKN		(8	 * 2)
#define BLOCK_SIZE			(256 * 2)

#define PI_f				3.14159265358979f

#define TIMING
//#define SHARED_VERSION
//#define REGISTER_VERSION

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
	#define MINIMA -512.03
	#define MAXIMA  511.97
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

/********************************/
/* POPULATION EVALUATION ON GPU */
/********************************/
#ifndef ANTENNAS
__global__ void evaluation_GPU(const int Np, const int D, const float * __restrict pop, float * __restrict fobj) {

	int j = threadIdx.x + blockIdx.x * blockDim.x;

	if (j < Np)  fobj[j] = functional(&pop[j*D], D);
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

		// For truss problem the fix() should be run here.

	}

}

/*******************************/
/* POPULATION SELECTION ON GPU */
/*******************************/
// Assumption: all the optimization variables are associated to the same thread block
#ifndef ANTENNAS
__global__ void selection_and_evaluation_GPU(const int Np, const int D, float * __restrict pop, const float * __restrict npop, float * __restrict fobj) {

	int i = threadIdx.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i < D) && (j < Np)) {

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

/***********************************************************************************/
/* GENERATION OF A NEW POPULATION, MUTATION, CROSSOVER AND SELECTION - GPU VERSION */
/***********************************************************************************/
#ifdef SHARED_VERSION
// --- It assumes that BLOCK_SIZE_POP >= D
__global__ void generation_new_population_mutation_crossover_selection_evaluation_GPU(float * __restrict__ pop, const int Np, const int D, float * __restrict__ npop, const float F, const float CR,
																					  const float * __restrict__ minimum, float * __restrict__ maximum, float * __restrict__ fobj, curandState * __restrict__ state) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	// --- Shared memory is used as a controlled cache
	__shared__ int		a[BLOCK_SIZE_POP], b[BLOCK_SIZE_POP], c[BLOCK_SIZE_POP];
	//__shared__ float	Rand[BLOCK_SIZE_POP], nfobj[BLOCK_SIZE_POP], temp[BLOCK_SIZE_POP];
	__shared__ float	nfobj[BLOCK_SIZE_POP], temp[BLOCK_SIZE_POP];

	// --- Generate mutation indices and crossover values
	if ((i == 0) && (j < Np)) {

		// --- Mutation indices
		do a[threadIdx.y] = Np*(curand_uniform(&state[j]));	while (a[threadIdx.y] == j);
		do b[threadIdx.y] = Np*(curand_uniform(&state[j]));	while (b[threadIdx.y] == j || b[threadIdx.y] == a[threadIdx.y]);
		do c[threadIdx.y] = Np*(curand_uniform(&state[j]));	while (c[threadIdx.y] == j || c[threadIdx.y] == a[threadIdx.y] || b[threadIdx.y] == a[threadIdx.y]);

		//// --- Crossover values
		//Rand[threadIdx.y] = curand_uniform(&state[j]);
	}

	__syncthreads();

	// --- Generate new population
	if ((i < D) && (j < Np)) {

		// --- Crossover values
		float Rand = curand_uniform(&state[j]);
		
		// --- Mutation and crossover
		//if (Rand[threadIdx.y] < CR)	npop[j*D + i] = pop[a[threadIdx.y] * D + i] + F*(pop[b[threadIdx.y] * D + i] - pop[c[threadIdx.y] * D + i]);
		if (Rand < CR)	npop[j*D + i] = pop[a[threadIdx.y] * D + i] + F*(pop[b[threadIdx.y] * D + i] - pop[c[threadIdx.y] * D + i]);
		else			npop[j*D + i] = pop[j*D + i];

		// --- Saturation due to constraints on the unknown parameters
		if (npop[j*D + i]>maximum[i]) npop[j*D + i] = maximum[i];
		else if (npop[j*D + i]<minimum[i])npop[j*D + i] = minimum[i];

	}

	__threadfence();

	if ((i == 0) && (j < Np)) {

		// --- Evaluation and selection
		nfobj[threadIdx.y] = functional(&npop[j*D], D);

		temp[threadIdx.y] = fobj[j];

	}

	__syncthreads();

	if ((i < D) && (j < Np)) {
		if (nfobj[threadIdx.y] < temp[threadIdx.y]) {
			pop[j*D + i] = npop[j*D + i];
			fobj[j] = nfobj[threadIdx.y];
		}

	}
}
#endif

#ifdef REGISTER_VERSION
__global__ void generation_new_population_mutation_crossover_selection_evaluation_GPU(float * __restrict__ pop, const int Np, const int D,
	float * __restrict__ npop, const float F, const float CR,
	const float * __restrict__ minimum, float * __restrict__ maximum,
	float * __restrict__ fobj, curandState * __restrict__ state) {

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
		float nfobj = functional(&npop[j*D], D);

		float temp = fobj[j];

		if (nfobj < temp) {
			for (int i = 0; i<D; i++) pop[j*D + i] = npop[j*D + i];
			fobj[j] = nfobj;
		}

	}
}
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

#if 0

void sortPopWithIndex(const int N, 						  // --- Number of populaiton
					  float *objValue, 					  // --- Objective value of population to be sorted
					  float * __restrict sortedObjValue,  // --- Output array after sorting
					  int* __restrict sortedIndex)        // --- To store the index of the input array after sort
{
	// --- Create the array of indices
	thrust::device_vector<int> d_Idx(N, 0);
	thrust::sequence(d_Idx.begin(), d_Idx.end());

	// --- Wrap raw pointer with a device_ptr 
	device_ptr<float> dev_ptr = device_pointer_cast(objValue);

	thrust::sort_by_key(dev_ptr, dev_ptr + N, d_Idx.data());

	for (int i = 0; i < N; i++)
	{
		sortedObjValue[i] = dev_ptr[i];
		sortedIndex[i] = d_Idx[i];
	}
}
#endif

/****************************************************/
/* Perform elitist redistribution to sub population */
/****************************************************/

void performElitistOperation(const int N,
							 const int subPopSize,
							 const int D,
							 float *pObjValue,
							 int* __restrict pIndex,
							 float* __restrict pObjValue_1, 
							 float* __restrict pObjValue_2, 
							 float* __restrict pObjValue_3) 
{
	// --- Wrap raw pointer with a device_ptr 
	device_ptr<float> dev_ptr_pObjValue = device_pointer_cast(pObjValue);

	// --- Create the array of indices
	thrust::device_vector<int> d_Idx(N, 0);
	thrust::sequence(d_Idx.begin(), d_Idx.end());

	thrust::sort_by_key(dev_ptr_pObjValue, dev_ptr_pObjValue + N, d_Idx.data()); //Sort main obj pop

	float* raw_ptr_pObjValue = thrust::raw_pointer_cast(&dev_ptr_pObjValue[0]);
	// --- Test sorted value with index
	for(int i = 0; i < N; i++)
	{
		std::cout << "dev_ptr_pObjValue[" << i << "] = " << dev_ptr_pObjValue[i];
		std::cout << " --- index[" << i << "] = " << d_Idx[i] << std::endl;
	}
	//Redistribute obj value after sort
	gpuErrchk(cudaMemcpy(pObjValue_1, raw_ptr_pObjValue, subPopSize*sizeof(float), cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpy(pObjValue_2, raw_ptr_pObjValue+subPopSize, subPopSize*sizeof(float), cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpy(pObjValue_3, raw_ptr_pObjValue+subPopSize+subPopSize, subPopSize*sizeof(float), cudaMemcpyDeviceToDevice));

	printf("elitist\n");

}

/********/
/* MAIN */
/********/
int main()
{
	// --- Number of individuals in the population (Np >=4 for mutation purposes)
	int			Np = 300;
	// --- Dimensionality of each individual (number of unknowns)
	int			D = 10;
	// --- Mutation factor (0 < F <= 2). Typically chosen in [0.5, 1], see Ref. [1]
	float		F = 0.5f;
	// --- Maximum number of generations
	int			Gmax = 1000;
	// --- Crossover constant (0 < CR <= 1)
	float		CR = 0.2f;

	// --- Mutually different random integer indices selected from {1, 2, … ,Np}
	int *h_best_index_dev;		// --- Host side current optimal member index of device side
	//int *d_best_index;		// --- Device side current optimal member index

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

	float *d_pop,					// --- Device side population
		  *d_fobj,					// --- Device side objective function value
		  *h_pop_dev_res,			// --- Host side population result of GPU computations
		  *h_best_dev,				// --- Host side population best value history of device side
		  *h_maxima,				// --- Host side maximum constraints vector
		  *h_minima,				// --- Host side minimum constraints vector
		  *h_testBufferObj,
		  *h_testBufferPop;

	float *d_subPop_1,				// --- Device side sub-population 1
		  *d_subPop_2,				// --- Device side sub-population 2
		  *d_subPop_3,				// --- Device side sub-population 3
		  *d_npop_1,				// --- Device side new population 1 (trial vectors)
		  *d_npop_2,				// --- Device side new population 2 (trial vectors)
		  *d_npop_3,				// --- Device side new population 3 (trial vectors)
		  *d_Rand_1,				// --- Device side crossover rand vector (uniformly distributed in (0,1))
		  *d_Rand_2,				// --- Device side crossover rand vector (uniformly distributed in (0,1))
		  *d_Rand_3,				// --- Device side crossover rand vector (uniformly distributed in (0,1))
		  *d_fobj_1,				// --- Device side objective function value
		  *d_fobj_2,				// --- Device side objective function value
		  *d_fobj_3,				// --- Device side objective function value
		  *d_maxima_1,				// --- Device side maximum constraints vector
		  *d_maxima_2,				// --- Device side maximum constraints vector
		  *d_maxima_3,				// --- Device side maximum constraints vector
		  *d_minima_1,				// --- Device side minimum constraints vector
		  *d_minima_2,				// --- Device side minimum constraints vector
		  *d_minima_3;				// --- Device side minimum constraints vector

	int *d_mutation_1,				// --- Device side mutation vector for subpop 1
		*d_mutation_2,				// --- Device side mutation vector for subpop 2
		*d_mutation_3,				// --- Device side mutation vector for subpop 3
		*d_afterSortIndex;		    // --- Device side index for population redistribution

	int subPopSize = Np/3;			// --- Np value has to be diviable for 3

	curandState *devState_1;			// --- Device side random generator state vector
	curandState *devState_2;			// --- Device side random generator state vector
	curandState *devState_3;			// --- Device side random generator state vector

	// --- Device side memory allocations
	gpuErrchk(cudaMalloc((void**)&d_pop, D*Np*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_fobj, Np*sizeof(float)));
	//gpuErrchk(cudaMalloc((void**)&d_npop, D*Np*sizeof(float)));
	//gpuErrchk(cudaMalloc((void**)&d_Rand, D*Np*sizeof(float)));
	//gpuErrchk(cudaMalloc((void**)&d_Rand, Np*sizeof(float)));
	//gpuErrchk(cudaMalloc((void**)&d_mutation, 3 * Np * sizeof(int)));
	//gpuErrchk(cudaMalloc((void**)&d_maxima, D*sizeof(float)));
	//gpuErrchk(cudaMalloc((void**)&d_minima, D*sizeof(float)));
	//gpuErrchk(cudaMalloc((void**)&devState, D*Np*sizeof(curandState)));

	// --- Host side memory allocations
	h_pop_dev_res = (float*)malloc(D*Np*sizeof(float));
	h_best_dev = (float*)malloc(1*sizeof(float));
	h_best_index_dev = (int*)malloc(1*sizeof(int));
	h_maxima = (float*)malloc(D*sizeof(float));
	h_minima = (float*)malloc(D*sizeof(float));
	h_testBufferObj = (float*)malloc(Np*sizeof(float));
	h_testBufferPop = (float*)malloc(D*Np*sizeof(float));

	// --- Define grid sizes
	//int Num_Blocks_Pop_1 = iDivUp(subPopSize, BLOCK_SIZE_POP);
	dim3 Grid_1(iDivUp(D, BLOCK_SIZE_UNKN), iDivUp(subPopSize, BLOCK_SIZE_POP));
	dim3 Block_1(BLOCK_SIZE_UNKN, BLOCK_SIZE_POP);

	//int Num_Blocks_Pop_2 = iDivUp(subPopSize, BLOCK_SIZE_POP);
	dim3 Grid_2(iDivUp(D, BLOCK_SIZE_UNKN), iDivUp(subPopSize, BLOCK_SIZE_POP));
	dim3 Block_2(BLOCK_SIZE_UNKN, BLOCK_SIZE_POP);

	//int Num_Blocks_Pop_3 = iDivUp(subPopSize, BLOCK_SIZE_POP);
	dim3 Grid_3(iDivUp(D, BLOCK_SIZE_UNKN), iDivUp(subPopSize, BLOCK_SIZE_POP));
	dim3 Block_3(BLOCK_SIZE_UNKN, BLOCK_SIZE_POP);

	// --- Device side memory allocations
	gpuErrchk(cudaMalloc((void**)&d_subPop_1, D*subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_subPop_2, D*subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_subPop_3, D*subPopSize*sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&d_fobj_1, subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_fobj_2, subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_fobj_3, subPopSize*sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&d_npop_1, D*subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_npop_2, D*subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_npop_3, D*subPopSize*sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&d_Rand_1, D*subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_Rand_2, D*subPopSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_Rand_3, D*subPopSize*sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&d_mutation_1, 3 * subPopSize * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_mutation_2, 3 * subPopSize * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_mutation_3, 3 * subPopSize * sizeof(int)));

	gpuErrchk(cudaMalloc((void**)&d_afterSortIndex, Np * sizeof(int)));

	gpuErrchk(cudaMalloc((void**)&d_maxima_1, D*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_minima_1, D*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_maxima_2, D*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_minima_2, D*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_maxima_3, D*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_minima_3, D*sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&devState_1, D*subPopSize*sizeof(curandState)));
	gpuErrchk(cudaMalloc((void**)&devState_2, D*subPopSize*sizeof(curandState)));
	gpuErrchk(cudaMalloc((void**)&devState_3, D*subPopSize*sizeof(curandState)));


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
	curand_setup_kernel << <iDivUp(D*subPopSize, BLOCK_SIZE), BLOCK_SIZE >> >(devState_1, time(NULL));
	curand_setup_kernel << <iDivUp(D*subPopSize, BLOCK_SIZE), BLOCK_SIZE >> >(devState_2, time(NULL));
	curand_setup_kernel << <iDivUp(D*subPopSize, BLOCK_SIZE), BLOCK_SIZE >> >(devState_3, time(NULL));

	// --- Initialize population
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

	TimingGPU timerGPU;
	timerGPU.StartCounter();
	for (int i = 0; i < Gmax; i++) {
		// --- Start Co-evolution
		// --- Generate mutation indices
		//generate_mutation_indices_GPU << <iDivUp(Np, BLOCK_SIZE_RAND1), BLOCK_SIZE_RAND1 >> >(d_mutation, Np, D, devState);
		generate_mutation_indices_GPU << <iDivUp(subPopSize, BLOCK_SIZE_RAND1), BLOCK_SIZE_RAND1 >> >(d_mutation_1, subPopSize, D, devState_1);
		generate_mutation_indices_GPU << <iDivUp(subPopSize, BLOCK_SIZE_RAND1), BLOCK_SIZE_RAND1 >> >(d_mutation_2, subPopSize, D, devState_2);
		generate_mutation_indices_GPU << <iDivUp(subPopSize, BLOCK_SIZE_RAND1), BLOCK_SIZE_RAND1 >> >(d_mutation_3, subPopSize, D, devState_3);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
		// --- Generate crossover values 
		//generate_crossover_values_GPU << <iDivUp(D * Np, BLOCK_SIZE_RAND2), BLOCK_SIZE_RAND2 >> >(d_Rand, Np, D, devState);
		generate_crossover_values_GPU << <iDivUp(D * subPopSize, BLOCK_SIZE_RAND2), BLOCK_SIZE_RAND2 >> >(d_Rand_1, subPopSize, D, devState_1);
		generate_crossover_values_GPU << <iDivUp(D * subPopSize, BLOCK_SIZE_RAND2), BLOCK_SIZE_RAND2 >> >(d_Rand_2, subPopSize, D, devState_2);
		generate_crossover_values_GPU << <iDivUp(D * subPopSize, BLOCK_SIZE_RAND2), BLOCK_SIZE_RAND2 >> >(d_Rand_3, subPopSize, D, devState_3);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// --- Generate new population
		//generation_new_population_GPU << <Grid, Block >> >(d_pop, Np, D, d_npop, F, CR, d_Rand, d_mutation, d_minima, d_maxima);
		generation_new_population_GPU << <Grid_1, Block_1 >> >(d_subPop_1, subPopSize, D, d_npop_1, F, CR, d_Rand_1, d_mutation_1, d_minima_1, d_maxima_1);
		generation_new_population_GPU << <Grid_2, Block_2 >> >(d_subPop_2, subPopSize, D, d_npop_2, F, CR, d_Rand_2, d_mutation_2, d_minima_2, d_maxima_2);
		generation_new_population_GPU << <Grid_3, Block_3 >> >(d_subPop_3, subPopSize, D, d_npop_3, F, CR, d_Rand_3, d_mutation_3, d_minima_3, d_maxima_3);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// --- Select new population and evaluate it
		//selection_and_evaluation_GPU << <Grid, Block >> >(Np, D, d_pop, d_npop, d_fobj);
		selection_and_evaluation_GPU << <Grid_1, Block_1 >> >(subPopSize, D, d_subPop_1, d_npop_1, d_fobj_1);
		selection_and_evaluation_GPU << <Grid_2, Block_2 >> >(subPopSize, D, d_subPop_2, d_npop_2, d_fobj_2);
		selection_and_evaluation_GPU << <Grid_3, Block_3 >> >(subPopSize, D, d_subPop_3, d_npop_3, d_fobj_3);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
		//find_minimum_GPU(Np, d_fobj, &h_best_dev[i], &h_best_index_dev[i]);
		printf("asasd\n");
		if(i%50 == 0) // Condition to merge three pops and apply elitist strategy
		{
			// --- Merge population
			gpuErrchk(cudaMemcpy(d_pop, d_subPop_1, subPopSize*sizeof(float), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(&d_pop[subPopSize], d_subPop_2, subPopSize*sizeof(float), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(&d_pop[subPopSize+subPopSize], d_subPop_3, subPopSize*sizeof(float), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(d_fobj, d_fobj_1, subPopSize*sizeof(float), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(&d_fobj[subPopSize], d_fobj_2, subPopSize*sizeof(float), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(&d_fobj[subPopSize+subPopSize], d_fobj_3, subPopSize*sizeof(float), cudaMemcpyDeviceToDevice));

			performElitistOperation(Np, subPopSize, D, d_fobj, d_afterSortIndex, d_fobj_1, d_fobj_2, d_fobj_3);
			gpuErrchk(cudaMemcpy(h_testBufferObj, d_fobj_2, subPopSize*sizeof(float), cudaMemcpyDeviceToHost));
			for(int z = 0; z < subPopSize; z++)
			{
				printf("aaaa = %f\n", h_testBufferObj[z]);
			}
			return 0;
		}
	}

#ifdef TIMING
	printf("Total timing = %f [s]\n", timerGPU.GetCounter() * 0.001);
#endif //TIMING

	// --- Merge population and obtain the final results
	//for (int mergingIndex = 0; mergingIndex < subPopSize; mergingIndex++)
	//{
	//	d_pop[mergingIndex] = d_subPop_1[mergingIndex];
	//	d_pop[mergingIndex + subPopSize] = d_subPop_2[mergingIndex];
	//	d_pop[mergingIndex + subPopSize + subPopSize] = d_subPop_3[mergingIndex];
	//	d_fobj[mergingIndex] = d_fobj_1[mergingIndex];
	//	d_fobj[mergingIndex + subPopSize] = d_fobj_2[mergingIndex];
	//	d_fobj[mergingIndex + subPopSize + subPopSize] = d_fobj_3[mergingIndex];
	//}
	//find_minimum_GPU(Np, d_fobj_1, &h_best_dev[0], &h_best_index_dev[0]);
	find_minimum_GPU(subPopSize, d_fobj_1, &h_best_dev[0], &h_best_index_dev[0]);

	//gpuErrchk(cudaMemcpy(h_pop_dev_res, d_pop, Np*sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_pop_dev_res, d_subPop_1, subPopSize*sizeof(float), cudaMemcpyDeviceToHost));
	for (int i = 0; i<D; i++) printf("Variable nr. %i = %.4f\n", i, h_pop_dev_res[h_best_index_dev[Gmax - 1] * D + i]);
	printf("Objective value: %.5f\n", h_best_dev[0]);

	return 0;
}
