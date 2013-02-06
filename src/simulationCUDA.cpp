#include <iostream>

#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "cudaUtils.h"

#include "constants.h"

#include "simulationCUDA.h"
#include "solver.h"

namespace Zillion {
    
// ---------------------------------------------------------------------------
    

SimulationCUDA::SimulationCUDA(int cudaDevice,
                               const float3* Pinit, const float3* Vinit,
                               unsigned nParticles, float particleRadius):
m_cudaDevice(cudaDevice),
m_Fd(NULL), m_Vd(NULL),
m_currentBuffer(0),
m_nParticles(nParticles), m_particleRadius(particleRadius)
{
    
    cudaGetDeviceProperties(&m_cudaProp, m_cudaDevice);
    
    for(unsigned n = 0; n < 2; n++)
    {
        m_P[n] = new SharedBuffer<float3>(GL_ARRAY_BUFFER, nParticles,
                                          GL_DYNAMIC_DRAW);
    }
    
    P().map();
    cudaMemcpy(P(), Pinit, nParticles*sizeof(float3), cudaMemcpyHostToDevice);
    P().unmap();

    cudaCheckError( cudaMalloc( (void**)&m_Fd, nParticles*sizeof(float3) ) );
    cudaMemset( m_Fd, 0, nParticles*sizeof(float3) );
    
    cudaCheckError( cudaMalloc( (void**)&m_Vd, nParticles*sizeof(float3) ) );
    cudaMemcpy(m_Vd, Vinit, nParticles*sizeof(float3), cudaMemcpyHostToDevice);
    
    int workSize = reduceWorkSize(nParticles, m_cudaProp) * sizeof(float3);
    cudaCheckError( cudaMalloc( (void**)&m_Wd, workSize) );
}


SimulationCUDA::~SimulationCUDA()
{
    cudaFree(m_Vd);
    cudaFree(m_Fd);
    
    for(unsigned n = 0; n < 2; n++)
        delete m_P[n];
}


// --------------------------------------------------------------------------- 

void
SimulationCUDA::stepForward(double dt)
{
    swapBuffers();
    
    prevP().map();
    P().map();
    
    accumulateForces(m_Fd, m_nParticles, MASS, GRAVITY, m_cudaProp.maxThreadsPerBlock);
    
    forwardEulerSolve(P(), m_Vd, prevP(), m_Fd, m_nParticles, MASS, dt,
                      m_cudaProp.maxThreadsPerBlock);
    
    handlePlaneCollisions(P(), m_Vd, prevP(), m_nParticles, m_particleRadius,
                          dt, RESTITUTION, m_cudaProp.maxThreadsPerBlock);
    
    P().unmap();
    prevP().unmap();
}


// --------------------------------------------------------------------------- 
    
} // END NAMESPACE ZILLION
