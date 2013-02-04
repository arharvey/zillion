#include <limits>
#include <iostream>

#include "solver.h"


#include "utils.h"

// -------------------------------------------------------------------------

namespace Zillion {


// -------------------------------------------------------------------------

float*
randomFloatArray(int seed, unsigned N)
{
    srand(seed);
    
    float* data = new float[N];
    for(unsigned n = 0; n < N; n++)
        data[n] = frand();
    
    return data;
}


void
minFloat3CPU(float* minimum, const float* P, unsigned N)
{
    if(N == 0)
        return;
    
    float x = std::numeric_limits<float>::max();
    float y = std::numeric_limits<float>::max();
    float z = std::numeric_limits<float>::max();
    
    const float* p = P;
    for(unsigned n = 0; n < N; n++, p += 3)
    {
        if(p[0] < x) x = p[0];
        if(p[1] < y) y = p[1];
        if(p[2] < z) z = p[2];
    }
    
    minimum[0] = x;
    minimum[1] = y;
    minimum[2] = z;
}


void
maxFloat3CPU(float* minimum, const float* P, unsigned N)
{
    if(N == 0)
        return;
    
    float x = -std::numeric_limits<float>::max();
    float y = -std::numeric_limits<float>::max();
    float z = -std::numeric_limits<float>::max();
    
    const float* p = P;
    for(unsigned n = 0; n < N; n++, p += 3)
    {
        if(p[0] > x) x = p[0];
        if(p[1] > y) y = p[1];
        if(p[2] > z) z = p[2];
    }
    
    minimum[0] = x;
    minimum[1] = y;
    minimum[2] = z;
}

void
sumFloat3CPU(float* minimum, const float* P, unsigned N)
{
    if(N == 0)
        return;
    
    float x = 0;
    float y = 0;
    float z = 0;
    
    const float* p = P;
    for(unsigned n = 0; n < N; n++, p += 3)
    {
        x += p[0];
        y += p[1];
        z += p[2];
    }
    
    minimum[0] = x;
    minimum[1] = y;
    minimum[2] = z;
}


void
minTest(unsigned N, int seed, int cudaDevice)
{
    float* P = randomFloatArray(seed, N*3);
    
    // CPU
    
    std::cout << "Calculating minimum on CPU" << std::endl;
    
    float minCPU[3] = {0, 0, 0};
    sumFloat3CPU(minCPU, P, N);
    
    
    // GPU
    
    std::cout << "Calculating minimum on CUDA GPU" << std::endl;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cudaDevice);
    
    float* Pd;
    cudaMalloc( (void**)&Pd, reduceWorkSize(N,prop)*3*sizeof(float) );
    cudaMemcpy(Pd, P, N*3*sizeof(float), cudaMemcpyHostToDevice);
    
    float minCUDA[3];
    sumFloat3(minCUDA, Pd, N, prop);
    
    cudaFree(Pd);
    
    // Check and report
    
    std::cout << "CPU min: " << minCPU[0] << ", "
                             << minCPU[1] << ", "
                             << minCPU[2] << std::endl;
    
    std::cout << "GPU min: " << minCUDA[0] << ", "
                             << minCUDA[1] << ", "
                             << minCUDA[2] << std::endl;
    
    
    bool match = true;
    for(unsigned i = 0; i < 3; i++)
        match = match && (minCPU[i] == minCUDA[i]);
    
    if(!match)
        std::cerr << "ERROR! CPU and GPU results do not match!" << std::endl;
    
    delete [] P;
}

} // END NAMESPACE ZILLION
        