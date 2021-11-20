#ifndef __SPATIAL_TRUSS_72BARS_CUH__
#define __SPATIAL_TRUSS_72BARS_CUH__
#define TOTAL_DOF 60

__host__ __device__ float functional(const float * __restrict A, const int D, float * d_invK, float * d_localLU, float * d_s);

#endif