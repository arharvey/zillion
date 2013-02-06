#ifndef _zillion_solver_h
#define _zillion_solver_h

#include <cuda_runtime.h>

namespace Zillion {
    
// ---------------------------------------------------------------------------
    
void
accumulateForces(float3* Fd, unsigned N, float m, float g,
                 unsigned nMaxThreadsPerBlock);

void
forwardEulerSolve(float3* Pd, float3* Vd,
                  const float3* prevPd, const float3* Fd, unsigned N,
                  float m, float dt,
                  unsigned nMaxThreadsPerBlock);

void
handlePlaneCollisions(float3* Pd, float3* Vd, const float3* P0d,
                            unsigned N, float r, float dt, float Cr,
                            unsigned nMaxThreadsPerBlock);
    
unsigned
reduceWorkSize(int N, const cudaDeviceProp& prop);


extern void
minFloat3(float3& result, float3* d_work, int N, const cudaDeviceProp& prop);


extern void
maxFloat3(float3& result, float3* d_work, int N, const cudaDeviceProp& prop);


extern void
sumFloat3(float3& result, float3* d_work, int N, const cudaDeviceProp& prop);
    
    
// --------------------------------------------------------------------------- 
    
} // END NAMESPACE ZILLION

#endif
