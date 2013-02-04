#ifndef _zillion_solver_h
#define _zillion_solver_h

#include <cuda_runtime.h>

//namespace Zillion {
    
// ---------------------------------------------------------------------------
    
void
accumulateForces(float* Fd, unsigned N, float m, float g,
                 unsigned nMaxThreadsPerBlock);

void
forwardEulerSolve(float* Pd, float* Vd,
                  const float* prevPd, const float* Fd, unsigned N,
                  float m, float dt,
                  unsigned nMaxThreadsPerBlock);

void
handlePlaneCollisions(float* Pd, float* Vd, const float* P0d,
                            unsigned N, float r, float dt, float Cr,
                            unsigned nMaxThreadsPerBlock);
    
unsigned
reduceWorkSize(const int N, const cudaDeviceProp& prop);


extern void
minFloat3(float* result, float** work_d, int N, const cudaDeviceProp& prop);


extern void
maxFloat3(float* result, float** work_d, int N, const cudaDeviceProp& prop);


extern void
sumFloat3(float* result, float** work_d, int N, const cudaDeviceProp& prop);
    
    
// --------------------------------------------------------------------------- 
    
//} // END NAMESPACE ZILLION

#endif
