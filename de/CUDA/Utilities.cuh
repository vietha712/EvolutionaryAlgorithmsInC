#ifndef UTILITIES_CUH
#define UTILITIES_CUH

#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cufft.h>
#include "curand.h"

//#include <thrust/pair.h>

//extern "C" int iDivUp(int, int);
__host__ __device__ int iDivUp(int, int);
extern "C" void gpuErrchk(cudaError_t);

#endif