#include <algorithm>

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