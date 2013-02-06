#include <limits>
#include <iostream>

#include <time.h>

#include "cudaUtils.h"
#include "solver.h"


#include "utils.h"

// -------------------------------------------------------------------------

namespace Zillion {


// -------------------------------------------------------------------------

float3*
randomFloat3Array(int seed, unsigned N)
{
    srand(seed);
    
    float3* data = new float3[N];
    for(unsigned n = 0; n < N; n++)
    {
        float3& d = data[n];
        d.x = frand();
        d.y = frand();
        d.z = frand();
    }
    
    return data;
}


void
minFloat3CPU(float3& result, const float3* P, unsigned N)
{
    if(N == 0)
        return;
    
    float3 m = {std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max()};
    
    for(unsigned n = 0; n < N; n++)
    {
        const float3& p = P[n];
        
        if(p.x < m.x) m.x = p.x;
        if(p.y < m.y) m.y = p.y;
        if(p.z < m.z) m.z = p.z;
    }
    
    result = m;
}


void
maxFloat3CPU(float3& result, const float3* P, unsigned N)
{
    if(N == 0)
        return;
    
    float3 m = {-std::numeric_limits<float>::max(),
                -std::numeric_limits<float>::max(),
                -std::numeric_limits<float>::max()};
    
    for(unsigned n = 0; n < N; n++)
    {
        const float3& p = P[n];
        
        if(p.x > m.x) m.x = p.x;
        if(p.y > m.y) m.y = p.y;
        if(p.z > m.z) m.z = p.z;
    }
    
    result = m;
}

void
sumFloat3CPU(float3& result, const float3* P, unsigned N)
{
    if(N == 0)
        return;
    
    float3 sum = {0, 0, 0};
    
    for(unsigned n = 0; n < N; n++)
    {
        const float3& p = P[n];
        
        sum.x += p.x;
        sum.y += p.y;
        sum.z += p.z;
    }
    
    result = sum;
}


void
minTest(int N, int seed, int cudaDevice)
{
    std::cout << "N = " << N << std::endl;
    const float3* P = randomFloat3Array(seed, N);
    
    // CPU
    
    std::cout << "Calculating minimum on CPU" << std::endl;
    
    float3 minCPU;
    
    clock_t cpuStart = clock();
    minFloat3CPU(minCPU, P, N);
    clock_t cpuEnd = clock();
    double cpuElapsed = ((double) (cpuEnd - cpuStart)) / CLOCKS_PER_SEC;
    
    // GPU
    
    std::cout << "Calculating minimum on CUDA GPU" << std::endl;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cudaDevice);
    
    float3* d_work;
    cudaCheckError( cudaMalloc( (void**)&d_work,
                                reduceWorkSize(N,prop)*sizeof(float3) ) );
    
    cudaMemcpy(d_work, P, N*3*sizeof(float), cudaMemcpyHostToDevice);
    
    float3 minCUDA;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    minFloat3(minCUDA, d_work, N, prop);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpuElapsed;
    cudaEventElapsedTime(&gpuElapsed, start, stop);
    
    cudaFree(d_work);
    
    // Check and report
    
    std::cout << "CPU min: " << minCPU.x << ", "
                             << minCPU.y << ", "
                             << minCPU.z << std::endl;
    
    std::cout << "GPU min: " << minCUDA.x << ", "
                             << minCUDA.y << ", "
                             << minCUDA.z << std::endl;
    
    
    float3 diff = minCPU - minCUDA;
    bool match = (fabs(diff.x) < 1e-6) &&
                 (fabs(diff.y) < 1e-6) &&
                 (fabs(diff.z) < 1e-6);
    
    if(!match)
        std::cerr << "ERROR! CPU and GPU results do not match!" << std::endl;
    
    std::cout << "CPU time: " << cpuElapsed << " ms" << std::endl;
    std::cout << "GPU time: " << gpuElapsed << " ms" << std::endl;
    
    delete [] P;
}

} // END NAMESPACE ZILLION
        