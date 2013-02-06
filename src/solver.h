#ifndef _zillion_solver_h
#define _zillion_solver_h

#include <cuda_runtime.h>

namespace Zillion {

// ---------------------------------------------------------------------------

namespace {
    const unsigned MAX_OCCUPANCY = 2; // Particles per cell
} // END ANONYMOUS NAMESPACE
    

// ---------------------------------------------------------------------------
    
void
accumulateForces(float3* Fd, unsigned N, float m, float g,
                 const cudaDeviceProp& prop);

void
forwardEulerSolve(float3* Pd, float3* Vd,
                  const float3* prevPd, const float3* Fd, unsigned N,
                  float m, float dt,
                  const cudaDeviceProp& prop);

void
handlePlaneCollisions(float3* Pd, float3* Vd, const float3* P0d,
                            unsigned N, float r, float dt, float Cr,
                            const cudaDeviceProp& prop);


void
populateCollisionGrid(int* d_G, int* d_GN, const float3* const d_P,
                      const int N, const float3 origin, 
                      const int3 dims, const float cellSize,
                      const cudaDeviceProp& prop);

    
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
