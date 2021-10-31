#ifndef __PLANAR_TRUSS_10BARS_CUH__
#define __PLANAR_TRUSS_10BARS_CUH__

#define TOTAL_DOF 12 // DOF * NUM_OF_NODES 

__host__ __device__ float functional(const float * __restrict A, const int D, float * d_invK, float * d_localLU, float * d_s);

#endif