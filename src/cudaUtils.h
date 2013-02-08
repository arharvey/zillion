#ifndef _zillion_cudaUtils_h
#define _zillion_cudaUtils_h

#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

#define cudaCheckError(status) \
Zillion::__cudaCheckError((status), __LINE__, __FILE__)

#define cudaCheckLastError() \
cudaCheckError(cudaGetLastError())


namespace Zillion {
    
inline
void
__cudaCheckError(cudaError_t status, int nLine, const char* szFile)
{
    if(status != cudaSuccess)
    {
        std::cerr << "CUDA ERROR: " << cudaGetErrorString(status) << std::endl
                  << szFile << " at line " << nLine << std::endl;
        exit(EXIT_FAILURE);
    }
}


// ---------------------------------------------------------------------------

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
float3
operator-(const float3& a)
{
    float3 v;
    v.x = -a.x;
    v.y = -a.y;
    v.z = -a.z;
    
    return v;
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



#if defined(__CUDACC__)

inline
__host__
__device__
float3
normalized(const float3& a)
{
    return a * rsqrtf(a^a);
}

#endif


inline __host__
std::ostream&
operator<<(std::ostream& out, float3& v) 
{
	out << v.x << ", " << v.y << ", " << v.z;
	return out;
}


inline __host__
std::ostream&
operator<<(std::ostream& out, int3& v) 
{
	out << v.x << ", " << v.y << ", " << v.z;
	return out;
}


// ---------------------------------------------------------------------------


} // END NAMESPACE ZILLION

#endif
