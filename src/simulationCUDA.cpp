#include <iostream>

#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "constants.h"

#include "simulationCUDA.h"


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


namespace Zillion {
    
// ---------------------------------------------------------------------------
    

SimulationCUDA::SimulationCUDA(int cudaDevice,
                               const float* Pinit, const float* Vinit,
                               unsigned nParticles, float particleRadius):
m_cudaDevice(cudaDevice),
m_Fd(NULL), m_Vd(NULL),
m_currentBuffer(0),
m_nParticles(nParticles), m_particleRadius(particleRadius)
{
    
    cudaGetDeviceProperties(&m_cudaProp, m_cudaDevice);
    
    for(unsigned n = 0; n < 2; n++)
    {
        m_P[n] = new SharedBuffer(GL_ARRAY_BUFFER,
                                  nParticles*3,
                                  GL_DYNAMIC_DRAW);
    }
    
    P().map();
    cudaMemcpy(P(), Pinit, nParticles*3*sizeof(float), cudaMemcpyHostToDevice);
    P().unmap();

    cudaMalloc( (void**)&m_Fd, nParticles*3*sizeof(float) );
    cudaMemset( m_Fd, 0, nParticles*3*sizeof(float) );
    
    cudaMalloc( (void**)&m_Vd, nParticles*3*sizeof(float) );
    cudaMemcpy(m_Vd, Vinit, nParticles*3*sizeof(float), cudaMemcpyHostToDevice);
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
