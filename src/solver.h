#ifndef _zillion_solver_h
#define _zillion_solver_h

#include <cuda_runtime.h>

namespace Zillion {

// ---------------------------------------------------------------------------

namespace {
    const float REPULSION = 3000;
    const float DAMPING = 4;
    const float SHEAR = 4;
    
    const unsigned MAX_OCCUPANCY = 8; // Particles per cell
    const unsigned MAX_WORK_IDS = 3*MAX_OCCUPANCY;
    
    const float CUDA_FLOAT_MAX = 1e16;
} // END ANONYMOUS NAMESPACE
    

// ---------------------------------------------------------------------------
    
void
accumulateForces(float3* Fd, unsigned N, float m, float g,
                 const cudaDeviceProp& prop);

void
populateCollisionGrid(int* d_G, int* d_GN, const float3* const d_P,
                      const int N, const float3 origin, 
                      const int3 dims, const float cellSize,
                      const cudaDeviceProp& prop);

void
resolveCollisions(float3* d_F, const int* const d_G, const int* const d_GN,
                 const float3* const d_P, const float3* const d_V, const int N, 
                 const float3 origin, const int3 dims, const float cellSize,
                 const float r,
                 const cudaDeviceProp& prop);


void
handlePlaneCollisions(float3* Pd, float3* Vd, float3* Fd, unsigned N, float r,
                      const float4& plane, float restitution, float dynamicFriction,
                      const cudaDeviceProp& prop);


void
forwardEulerSolve(float3* Pd, float3* Vd,
                  const float3* prevPd, const float3* Fd, unsigned N,
                  float m, float dt,
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
