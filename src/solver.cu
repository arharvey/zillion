#include <algorithm>

#include <assert.h>

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


inline
__host__
__device__
float3&
operator*=(float3& a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    
    return a;
}


inline
__host__
__device__
float3
operator*(const float3& a, float b)
{
    float3 v;
    v.x = a.x*b;
    v.y = a.y*b;
    v.z = a.z*b;
    
    return v;
}


inline
__host__
__device__
float3
operator*(float a, const float3& b)
{
    float3 v;
    v.x = b.x*a;
    v.y = b.y*a;
    v.z = b.z*a;
    
    return v;
}


inline
__host__
__device__
float3
operator+(const float3& a, const float3& b)
{
    float3 v;
    v.x = a.x+b.x;
    v.y = a.y+b.y;
    v.z = a.z+b.z;
    
    return v;
}


inline
__host__
__device__
float3&
operator+=(float3& a, const float3& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    
    return a;
}


inline
__host__
__device__
float3
operator-(const float3& a, const float3& b)
{
    float3 v;
    v.x = a.x-b.x;
    v.y = a.y-b.y;
    v.z = a.z-b.z;
    
    return v;
}


inline
__host__
__device__
float3&
operator-=(float3& a, const float3& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    
    return a;
}


inline
__host__
__device__
float
operator^(const float3& a, const float3& b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}


inline
__host__
__device__
float
operator^(const float4& a, const float3& b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w;
}


inline
__host__
__device__
float
operator^(const float3& a, const float4& b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z + b.w;
}


// ---------------------------------------------------------------------------

inline
unsigned
roundUpToPower2(unsigned v)
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


// ---------------------------------------------------------------------------

__global__
void
accumulateForcesKernel(float* Fd, unsigned N, float m, float g)
{
    unsigned n = blockIdx.x*blockDim.x + threadIdx.x;
    while(n < N)
    {
        float3& F = unpack(Fd, n);

        F.x = 0.0f;
        F.y = m*g;
        F.z = 0.0f;
        
        n += blockDim.x * gridDim.x;
    }
}

__host__
void
accumulateForces(float* Fd, unsigned N, float m, float g, unsigned nMaxThreadsPerBlock)
{
    dim3 dimBlock( std::min(N, nMaxThreadsPerBlock) );
    dim3 dimGrid( (N + nMaxThreadsPerBlock-1) / nMaxThreadsPerBlock );
    
    accumulateForcesKernel<<<dimGrid, dimBlock>>>(Fd, N, m, g);
}


// ---------------------------------------------------------------------------

__global__
void
forwardEulerSolveKernel(float* Pd, float* Vd,
                        const float* P0d, const float* Fd,
                        unsigned N,
                        float m, float dt)
{
    unsigned n = blockIdx.x*blockDim.x + threadIdx.x;
    while(n < N)
    {
        const float3& F = unpack(Fd, n);
        float3& V = unpack(Vd, n);
        
        // a = F/m
        
        const float _1_m = 1.0f/m;
        
        V += F * (_1_m * dt);
        
        const float3& P0 = unpack(P0d, n);
        float3& P = unpack(Pd, n);

        P = P0 + V*dt;
        
        n += blockDim.x * gridDim.x;
    }
}

__host__
void
forwardEulerSolve(float* Pd, float* Vd,
                  const float* prevPd, const float* Fd,
                  unsigned N, float m, float dt, unsigned nMaxThreadsPerBlock)
{
    dim3 dimBlock( std::min(N, nMaxThreadsPerBlock) );
    dim3 dimGrid( (N + nMaxThreadsPerBlock-1) / nMaxThreadsPerBlock );
    
    forwardEulerSolveKernel<<<dimGrid, dimBlock>>>(Pd, Vd, prevPd, Fd, N,
                                                   m, dt);
}


// ---------------------------------------------------------------------------


__global__
void
handlePlaneCollisionsKernel(float* Pd, float* Vd, const float* P0d,
                            unsigned N, float r, float dt, float Cr)
{
    unsigned n = blockIdx.x*blockDim.x + threadIdx.x;
    while(n < N)
    {
        const float3& P0 = unpack(P0d, n);
        float3& V = unpack(Vd, n);
        float3& P = unpack(Pd, n);
        
        const float3 plane = make_float3(0.0f, 1.0f, 0.0f);
        const float d = 0.0f;
        
        const float distanceFromPlane = (plane ^ P) - d - r;
        
        // Have we collided with the plane?
        if(distanceFromPlane <= 1e-6f)
        {
            const float perpSpeed = (V ^ plane);
            
            // Components of velocity perpendicular and tangent to plane
            const float3 Vp = perpSpeed * plane;
            const float3 Vt = V-Vp;
            
            // Bounce or contact?
            V = Vt;
            if(perpSpeed < -0.1f)
                V -= Vp*Cr;
            else
                V *= (1.0f - 0.5f*dt);
            
            P.y = r;
        }
        
        n += blockDim.x * gridDim.x;
    }
    
}


__host__
void
handlePlaneCollisions(float* Pd, float* Vd, const float* P0d,
                      unsigned N, float r, float dt, float Cr,
                      unsigned nMaxThreadsPerBlock)
{
    dim3 dimBlock( std::min(N, nMaxThreadsPerBlock) );
    dim3 dimGrid( (N + nMaxThreadsPerBlock-1) / nMaxThreadsPerBlock );
    
    handlePlaneCollisionsKernel<<<dimGrid, dimBlock>>>(Pd, Vd, P0d, N, r, dt, Cr);
}

// ---------------------------------------------------------------------------

__global__
void
fillKernel(float* Pd, const float3 v)
{
    float3& dest = unpack(Pd, threadIdx.x);
    
    dest = v;
}


template<class Op>
__global__
void
float3ReduceKernel(float* Pd)
{
    extern __shared__ float sm[];
    
    // Each thread loads one element from the position array into shared mem
    unsigned tid = threadIdx.x * 3;
    unsigned i = (blockIdx.x*blockDim.x + threadIdx.x) * 3;
    
    sm[tid]   = Pd[i];
    sm[tid+1] = Pd[i+1];
    sm[tid+2] = Pd[i+2];

    __syncthreads();
    
    unsigned s = blockDim.x/2;
    s += s<<1;
    
    for(; s >= 3; s >>= 1)
    {
        if(tid < s)
        {
            float* a = sm + tid;
            const float* b = a + s;
            
            Op::doIt(a, b);
        }
        
        __syncthreads();
    }
    
    
    // Write result to global memory
    if(tid == 0)
    {
        unsigned bid = blockIdx.x*3;
        
        Pd[bid]   = sm[0];
        Pd[bid+1] = sm[1];
        Pd[bid+2] = sm[2];
    }
};


__host__
void
reduceDims(unsigned& nBlocks, unsigned& nThreads, 
                 const unsigned N, const cudaDeviceProp& prop)
{
    unsigned nThreadsPerBlockRaw = std::min(int(N), prop.maxThreadsPerBlock);
    
    nBlocks = (N + prop.maxThreadsPerBlock-1) / prop.maxThreadsPerBlock;
    
    // Kernel assumes that input float3 array has base-2 number of elements
    nThreads = roundUpToPower2(nThreadsPerBlockRaw);
}


__host__
unsigned
reduceWorkSize(const unsigned N, const cudaDeviceProp& prop)
{
    unsigned nBlocks = 0, nThreads = 0;
    reduceDims(nBlocks, nThreads, N, prop);
    
    return nBlocks * nThreads;
}


template<class Op>
__host__
unsigned
float3ReducePass(float* Pd, unsigned N, const cudaDeviceProp& prop)
{
    unsigned nBlocks = 0, nThreads = 0;
    reduceDims(nBlocks, nThreads, N, prop);
    
    unsigned nResidualThreads = (nBlocks * nThreads) - N;
    if(nResidualThreads)
        fillKernel<<<dim3(1), dim3(nResidualThreads)>>>(Pd+N*3, Op::padding);
    
    dim3 dimBlock(nThreads);
    dim3 dimGrid(nBlocks);
    
    unsigned nAllocSharedMemPerBlock = nThreads * 3 * sizeof(float);
    assert(nAllocSharedMemPerBlock < prop.sharedMemPerBlock);
    
    float3ReduceKernel<Op><<<dimGrid, dimBlock, nAllocSharedMemPerBlock>>>(Pd);
    
    return nBlocks;
};


template<class Op>
__host__
void
float3Reduce(float* result, float* Pd, unsigned N, const cudaDeviceProp& prop)
{
    while(N > 1)
        N = float3ReducePass<Op>(Pd, N, prop);
    
    cudaMemcpy(result, Pd, 3*sizeof(float), cudaMemcpyDeviceToHost);
}


// ---------------------------------------------------------------------------

struct MinReductionOp
{
    static float3 padding;
    
    static inline
    __device__
    void
    doIt(float* a, const float* b)
    {
        {
            const float Xa = a[0], Xb = b[0];
            if(Xb < Xa)
                a[0] = Xb;
        }

        {
            const float Ya = a[1], Yb = b[1];
            if(Yb < Ya)
                a[1] = Yb;
        }

        {
            const float Za = a[2], Zb = b[2];
            if(Zb < Za)
                a[2] = Zb;
        }
    }
};


float3 MinReductionOp::padding = {std::numeric_limits<float>::max(),
                                  std::numeric_limits<float>::max(),
                                  std::numeric_limits<float>::max()};


__host__
void
minFloat3(float* result, float* Pd, unsigned N, const cudaDeviceProp& prop)
{
    float3Reduce<MinReductionOp>(result, Pd, N, prop);
}


// ---------------------------------------------------------------------------

struct MaxReductionOp
{
    static float3 padding;
    
    static inline
    __device__
    void
    doIt(float* a, const float* b)
    {
        {
            const float Xa = a[0], Xb = b[0];
            if(Xb > Xa)
                a[0] = Xb;
        }

        {
            const float Ya = a[1], Yb = b[1];
            if(Yb > Ya)
                a[1] = Yb;
        }

        {
            const float Za = a[2], Zb = b[2];
            if(Zb > Za)
                a[2] = Zb;
        }
    }
};


float3 MaxReductionOp::padding = {-std::numeric_limits<float>::max(),
                                  -std::numeric_limits<float>::max(),
                                  -std::numeric_limits<float>::max()};


__host__
void
maxFloat3(float* result, float* Pd, unsigned N, const cudaDeviceProp& prop)
{
    float3Reduce<MaxReductionOp>(result, Pd, N, prop);
}



// ---------------------------------------------------------------------------

struct SumReductionOp
{
    static float3 padding;
    
    static inline
    __device__
    void
    doIt(float* a, const float* b)
    {
        a[0] += b[0];
        a[1] += b[1];
        a[2] += b[2];
    }
};


float3 SumReductionOp::padding = {0.0, 0.0, 0.0};


__host__
void
sumFloat3(float* result, float* Pd, unsigned N, const cudaDeviceProp& prop)
{
    float3Reduce<SumReductionOp>(result, Pd, N, prop);
}
