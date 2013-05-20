#include <algorithm>
#include <iostream>

#include <assert.h>

#include "cudaUtils.h"
#include "solver.h"

#define SOLVER_DIAGNOSTICS (0)

namespace Zillion {

// ---------------------------------------------------------------------------
    
__device__
float3&
unpack(float* array, unsigned n)
{
    return *(float3*)&array[n*3];
}


__device__
const float3&
unpack(const float* array, unsigned n)
{
    return *(const float3*)&array[n*3];
}

// ---------------------------------------------------------------------------

inline
int
roundUpToPower2(int v)
{
    // Handle case where v is already a power of 2
    v--;
    
    // Copy highest set bit to all bits below
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    
    // New power of 2
    v++;
    
    return v;
}



__host__
void
calcDims(int& nBlocks, int& nThreads, const int N, const cudaDeviceProp& prop)
{
    nThreads = std::min(N, prop.maxThreadsPerBlock);
    nBlocks = (N + prop.maxThreadsPerBlock-1) / prop.maxThreadsPerBlock;
}

// ---------------------------------------------------------------------------

__global__
void
accumulateForcesKernel(float3* Fd, unsigned N, float m, float g)
{
    unsigned n = blockIdx.x*blockDim.x + threadIdx.x;
    while(n < N)
    {
        float3& F = Fd[n];

        F.x = 0.0f;
        F.y = m*g;
        F.z = 0.0f;
        
        n += blockDim.x * gridDim.x;
    }
}

__host__
void
accumulateForces(float3* Fd, unsigned N, float m, float g, const cudaDeviceProp& prop)
{
    int nBlocks, nThreads;
    calcDims(nBlocks, nThreads, N, prop);
    
    accumulateForcesKernel<<<nBlocks, nThreads>>>(Fd, N, m, g);
    cudaCheckLastError();
}


// ---------------------------------------------------------------------------

inline
__device__
int
cellIndex(const float3& P, const float3& origin, const int3& dims,
          const float cellSize)
{
    float3 I = P;
    I -= origin;
    I *= 1.0/cellSize;

    return int(I.x) + int(I.y)*dims.x + int(I.z)*dims.x*dims.y;
}


__global__
void
populateCollisionGridKernel(int* d_G, int* d_GN, const float3* const d_P,
                        const int N, const float3 origin, 
                        const int3 dims, const float cellSize)
{
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    while(n < N)
    {
        const int cell = cellIndex(d_P[n], origin, dims, cellSize);
        
        const int i = atomicAdd(d_GN+cell, 1);
        if(i < MAX_OCCUPANCY)
        {
            int* out = d_G + cell*MAX_OCCUPANCY;
            out[i] = n;
        }
        
        n += blockDim.x * gridDim.x;
    }
}


__host__
void
populateCollisionGrid(int* d_G, int* d_GN, const float3* const d_P,
                      const int N, const float3 origin, 
                      const int3 dims, const float cellSize,
                      const cudaDeviceProp& prop)
{
    int nBlocks, nThreads;
    calcDims(nBlocks, nThreads, N, prop);
    
    populateCollisionGridKernel<<<nBlocks, nThreads>>>(d_G, d_GN, d_P, N, 
                                                   origin, dims, cellSize);
    cudaCheckLastError();
}

// ---------------------------------------------------------------------------

inline
__device__
void
addCollisionReaction(float3& F, const float3& Rij_dir,
                      const float d, const float3& Vij)
{
    // Repulsive force
    F += REPULSION*d * Rij_dir;
    
    // Damping force
    const float3 Vijn = (Vij^Rij_dir)*Rij_dir;
    F += -DAMPING * Vijn;
    
    // Shear force
    F += -SHEAR*(Vij - Vijn);
}



// ---------------------------------------------------------------------------


__global__
void
resolveCollisionsKernel(float3* d_F, const int* const d_G, const int* const d_GN,
                 const float3* const d_P, const float3* const d_V, const int N, 
                 const float3 origin, const int3 dims, const float cellSize,
                 const float r)
{
    extern __shared__ int ids[];
    
    const int strideY = dims.x;
    const int strideZ = dims.x*dims.y;
    
    const float minDistSq = 4.0f*r*r;
    
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    while(n < N)
    {
        float3 F = make_float3(0, 0, 0);
        
        const float3 P = d_P[n];
        const float3 V = d_V[n];
        const int central = cellIndex(P, origin, dims, cellSize);
        
        // We gather all the collisions in horizontal groups (X & Z dimensions),
        // as the particles are more likely to be distributed horizontally than
        // vertically. This of course is just a heuristic :)
        for(int j = -strideY; j <= strideY; j += strideY)
        {
            
            
            for(int k = -strideZ; k <= strideZ; k += strideZ)
            {
                int collisions = 0;
                
                for(int i = -1; i <= 1; ++i)
                {
                    const int cell = central + i + j + k;
                    
                    int count = d_GN[cell];
                    if(count > MAX_OCCUPANCY)
                        count = MAX_OCCUPANCY;
                    
                    const int* cellIds = d_G + cell*MAX_OCCUPANCY;
                    for(int m = 0; m < count; ++m)
                    {
                        const int id = cellIds[m];
                        
                        // Cannot collide with self!
                        if(id != n)
                        {
                            // Avoid shared bank conflicts
                            ids[blockDim.x*collisions + threadIdx.x] = id;
                            ++collisions;
                        }
                    }
                }
                
                // Now process queued collisions
                for(int c = 0; c < collisions; ++c)
                {
                    const int otherParticle = ids[blockDim.x*c + threadIdx.x];

                    const float3 Rij = P - d_P[otherParticle];
                    const float Rij_distSq = Rij^Rij;
                    if(Rij_distSq < minDistSq)
                    {
                        // We have a collision! Push particles away from each other
                        float Rij_dist = rsqrtf(Rij_distSq);
                        const float3 Rij_dir = Rij * Rij_dist;
                        Rij_dist *= Rij_distSq;

                        const float3& Vij = V - d_V[otherParticle];
                        addCollisionReaction(F, Rij_dir, 2*r-Rij_dist, Vij);
                    }
                }
            }
            
            
        }
        
        d_F[n] += F;
        
        n += blockDim.x * gridDim.x;
    }
}


__host__
void
resolveCollisions(float3* d_F, const int* const d_G, const int* const d_GN,
                 const float3* const d_P, const float3* const d_V, const int N, 
                 const float3 origin, const int3 dims, const float cellSize,
                 const float r,
                 const cudaDeviceProp& prop)
{
    int nBlocks, nThreads, nShared;
    
    nThreads = prop.sharedMemPerBlock / (MAX_WORK_IDS * sizeof(int));
    nThreads = std::min(nThreads, 128);
    nThreads = std::max(prop.warpSize, std::min(N, nThreads));
    
    // Round down threads to nearest warp size and ensure it is within
    // sensible bounds
    nThreads &= ~(prop.warpSize-1);
    assert(nThreads > 0);
    
    nBlocks = (N + nThreads-1) / nThreads;
    nShared = nThreads * (MAX_WORK_IDS * sizeof(int));
    
    //std::cout << nBlocks << " " << nThreads << " " << nShared << std::endl;
    
    resolveCollisionsKernel<<<nBlocks, nThreads, nShared>>>
            (d_F, d_G, d_GN, d_P, d_V, N, origin, dims, cellSize, r);
    cudaCheckLastError();
}


// ---------------------------------------------------------------------------

__global__
void
handlePlaneCollisionsKernel(float3* Pd, float3* Vd, float3* Fd,
                            unsigned N, float r, const float4 plane,
                            float restitution, float kineticFriction)
{
    const float3 planeDir = make_float3(plane.x, plane.y, plane.z);
    
    unsigned n = blockIdx.x*blockDim.x + threadIdx.x;
    while(n < N)
    {
        float3 P = Pd[n];
        
        float distanceFromPlane = plane ^ P;
        
        // Have we collided with the plane?
        if(distanceFromPlane < r)
        {
            float3 F = Fd[n];
            addCollisionReaction(F, planeDir, -(distanceFromPlane-r), Vd[n]);
            Fd[n] = F;
        }
        
        n += blockDim.x * gridDim.x;
    }
    
}


__host__
void
handlePlaneCollisions(float3* Pd, float3* Vd, float3* Fd,
                      unsigned N, float r, const float4& plane,
                      float restitution, float kineticFriction,
                      const cudaDeviceProp& prop)
{
    int nBlocks, nThreads;
    calcDims(nBlocks, nThreads, N, prop);
    
    handlePlaneCollisionsKernel<<<nBlocks, nThreads>>>(Pd, Vd, Fd, N, r, plane,
                                                       restitution,
                                                       kineticFriction);
    cudaCheckLastError();
}


// ---------------------------------------------------------------------------

__global__
void
handleSphereCollisionsKernel(float3* Pd, float3* Vd, float3* Fd,
                            unsigned N, float r,
                            const float3 center, const float radius,
                            float restitution, float kineticFriction)
{
    const float minDistSq = (radius+r)*(radius+r);
    
    unsigned n = blockIdx.x*blockDim.x + threadIdx.x;
    while(n < N)
    {
        float3& P = Pd[n];
        float3& V = Vd[n];
        float3& F = Fd[n];
        
        const float3 Rij = P - center;
        const float Rij_distSq = Rij^Rij;
        if(Rij_distSq < minDistSq)
        {
            // We have a collision! Push particles away from each other
            float Rij_dist = rsqrtf(Rij_distSq);
            const float3 Rij_dir = Rij * Rij_dist;
            Rij_dist *= Rij_distSq;

            const float3& Vij = V;//- d_V[otherParticle];
            addCollisionReaction(F, Rij_dir, (radius+r)-Rij_dist, Vij);
        }
        
        n += blockDim.x * gridDim.x;
    }
    
}


__host__
void
handleSphereCollisions(float3* Pd, float3* Vd, float3* Fd, unsigned N, float r,
                       const float3& center, const float radius,
                       float restitution, float dynamicFriction,
                       const cudaDeviceProp& prop)
{
    int nBlocks, nThreads;
    calcDims(nBlocks, nThreads, N, prop);
    
    handleSphereCollisionsKernel<<<nBlocks, 256>>>(Pd, Vd, Fd, N, r,
                                                       center, radius,
                                                       restitution,
                                                       dynamicFriction);
    cudaCheckLastError();
}


// ---------------------------------------------------------------------------

__global__
void
forwardEulerSolveKernel(float3* Pd, float3* Vd,
                        const float3* P0d, const float3* Fd,
                        unsigned N,
                        float m, float dt)
{
    unsigned n = blockIdx.x*blockDim.x + threadIdx.x;
    while(n < N)
    {
        const float3& F = Fd[n];
        float3& V = Vd[n];
        
        // a = F/m
        
        const float _1_m = 1.0f/m;
        
        V += F * (_1_m * dt);
        
        const float3& P0 = P0d[n];
        float3& P = Pd[n];

        P = P0 + V*dt;
        
        n += blockDim.x * gridDim.x;
    }
}

__host__
void
forwardEulerSolve(float3* Pd, float3* Vd,
                  const float3* prevPd, const float3* Fd,
                  unsigned N, float m, float dt, const cudaDeviceProp& prop)
{
    int nBlocks, nThreads;
    calcDims(nBlocks, nThreads, N, prop);
    
    forwardEulerSolveKernel<<<nBlocks, nThreads>>>(Pd, Vd, prevPd, Fd, N,
                                                   m, dt);
    cudaCheckLastError();
}


// ---------------------------------------------------------------------------

__global__
void
fillKernel(float3* Pd, const float3 v)
{
    Pd[threadIdx.x] = v;
}


template<class Op>
__global__
void
float3ReduceKernel(float3* out_d, const float3* in_d)
{
    extern __shared__ float3 sm[];
    
    // Each thread loads one element from the position array into shared mem
    const int tid = threadIdx.x;
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    sm[tid] = in_d[i];

    __syncthreads();
    
    int s = (blockDim.x/2);
 
    for(; s >= 1; s >>= 1)
    {
        if(tid < s)
        {
            float3* a = sm + tid;
            const float3* b = a + s;
            
            Op::doIt(*a, *b);
        }
        
        __syncthreads();
    }
    
    
    // Write result to global memory
    if(tid == 0)
        out_d[blockIdx.x] = sm[0];
};


__host__
void
reduceDims(int& nBlocks, int& nThreads, const int N, const cudaDeviceProp& prop)
{
    nThreads = std::min(roundUpToPower2(N), prop.maxThreadsPerBlock);
    
    // Kernel assumes that input float3 array has base-2 number of elements
    nThreads = std::max(64, std::min(nThreads, 128));
    
    nBlocks = (N + nThreads-1) / nThreads;
    
    //assert(nBlocks <= prop.maxGridSize[0]);
}


__host__
unsigned
reduceWorkSize(int N, const cudaDeviceProp& prop)
{
    int total = 0;
    
    for(int n = 0; n < 2; n++)
    {
        int nBlocks, nThreads;
        reduceDims(nBlocks, nThreads, N, prop);
    
        total += nBlocks * nThreads;
        
        N = nBlocks;
    }
    
    return total;
}


template<class Op>
__host__
unsigned
float3ReducePass(float3* d_out, float3* d_in, int N, const cudaDeviceProp& prop)
{
    int nBlocks = 0, nThreads = 0;
    reduceDims(nBlocks, nThreads, N, prop);
    
    int nResidualThreads = (nBlocks * nThreads) - N;
    
#if SOLVER_DIAGNOSTICS
    std::cout << "N: " << N 
              << ", Blocks: " << nBlocks
              << ", Threads: " << nThreads
              << ", Residual: " << nResidualThreads << std::endl;
#endif
    
    if(nResidualThreads > 0)
    {
        fillKernel<<<1, nResidualThreads>>>(d_in+N, Op::padding);
        cudaCheckLastError();
    }
    
#if SOLVER_DIAGNOSTICS
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
#endif
    
    int nAllocSharedMemPerBlock = nThreads * sizeof(float3);
    float3ReduceKernel<Op><<<nBlocks, nThreads, nAllocSharedMemPerBlock>>>(d_out, d_in);
    cudaCheckLastError();
    
#if SOLVER_DIAGNOSTICS
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuElapsed;
    cudaEventElapsedTime(&gpuElapsed, start, stop);
    
    std::cout << "GPU time: " << gpuElapsed << " ms" << std::endl;
#endif
    
    return nBlocks;
};


template<class Op>
__host__
void
float3Reduce(float3& result, float3* d_work, int N, const cudaDeviceProp& prop)
{
    int nBlocks, nThreads;
    reduceDims(nBlocks, nThreads, N, prop);
    int firstPassSize = nBlocks * nThreads;
    
    float3* d_W[2] = {d_work, d_work + firstPassSize};
    
    int i = 0;
    while(N > 1)
    {
        float3* d_out = d_W[1-i];
        float3* d_in = d_W[i];
        
        N = float3ReducePass<Op>(d_out, d_in, N, prop);
        
        // Alternate buffers
        i = 1-i;
    }
    
    cudaMemcpy(&result, d_W[i], sizeof(float3), cudaMemcpyDeviceToHost);
}


// ---------------------------------------------------------------------------

struct MinReductionOp
{
    static float3 padding;
    
    static inline
    __device__
    void
    doIt(float3& a, const float3& b)
    {
        {
            const float Xa = a.x, Xb = b.x;
            if(Xb < Xa)
                a.x = Xb;
        }

        {
            const float Ya = a.y, Yb = b.y;
            if(Yb < Ya)
                a.y = Yb;
        }

        {
            const float Za = a.z, Zb = b.z;
            if(Zb < Za)
                a.z = Zb;
        }
    }
};


float3 MinReductionOp::padding = {CUDA_FLOAT_MAX,
                                  CUDA_FLOAT_MAX,
                                  CUDA_FLOAT_MAX};


__host__
void
minFloat3(float3& result, float3* d_work, int N, const cudaDeviceProp& prop)
{
    float3Reduce<MinReductionOp>(result, d_work, N, prop);
}


// ---------------------------------------------------------------------------

struct MaxReductionOp
{
    static float3 padding;
    
    static inline
    __device__
    void
    doIt(float3& a, const float3& b)
    {
        {
            const float Xa = a.x, Xb = b.x;
            if(Xb > Xa)
                a.x = Xb;
        }

        {
            const float Ya = a.y, Yb = b.y;
            if(Yb > Ya)
                a.y = Yb;
        }

        {
            const float Za = a.z, Zb = b.z;
            if(Zb > Za)
                a.z = Zb;
        }
    }
};


float3 MaxReductionOp::padding = {-CUDA_FLOAT_MAX,
                                  -CUDA_FLOAT_MAX,
                                  -CUDA_FLOAT_MAX};


__host__
void
maxFloat3(float3& result, float3* d_work, int N, const cudaDeviceProp& prop)
{
    float3Reduce<MaxReductionOp>(result, d_work, N, prop);
}



// ---------------------------------------------------------------------------

struct SumReductionOp
{
    static float3 padding;
    
    static inline
    __device__
    void
    doIt(float3& a, const float3& b)
    {
        a += b;
    }
};


float3 SumReductionOp::padding = {0.0, 0.0, 0.0};


__host__
void
sumFloat3(float3& result, float3* d_work, int N, const cudaDeviceProp& prop)
{
    float3Reduce<SumReductionOp>(result, d_work, N, prop);
}

// ---------------------------------------------------------------------------

} // END NAMESPACE ZILLION
