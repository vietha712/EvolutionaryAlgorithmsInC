#ifndef __PLANAR_TRUSS_52BARS_CUH__
#define __PLANAR_TRUSS_52BARS_CUH__

#define TOTAL_DOF 40

__host__ __device__ float functional(const float * __restrict A, const int D, float * d_invK, float * d_localLU, float * d_s);

#endif