#include <stdio.h>
#include <assert.h>
//#include <math.h>

#include "cuda_runtime.h"
#include <cuda.h>

#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cufft.h>

#include "Utilities.cuh"

#define DEBUG

#define PI_R         3.14159265358979323846f

/*******************/
/* iDivUp FUNCTION */
/*******************/
//extern "C" int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }
__host__ __device__ int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }

/********************/
/* CUDA ERROR CHECK */
/********************/
// --- Credit to http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) { exit(code); }
	}
}

extern "C" void gpuErrchk(cudaError_t ans) { gpuAssert((ans), __FILE__, __LINE__); }