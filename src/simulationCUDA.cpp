#include <iostream>

#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "simulationCUDA.h"


void
initGrid(float* Pd, const float* P0d, unsigned nPts);


namespace Zillion {
    
// ---------------------------------------------------------------------------
    

SimulationCUDA::SimulationCUDA(const float* Pinit, unsigned nParticles,
                               float particleRadius):
m_nParticles(nParticles), m_particleRadius(particleRadius)
{
    for(unsigned n = 0; n < 2; n++)
    {
        m_P[n] = new SharedBuffer(GL_ARRAY_BUFFER,
                                  nParticles*3,
                                  GL_DYNAMIC_DRAW);
    }
        
    P(0).map();
    P(1).map();

    cudaMemcpy(P(0), Pinit, nParticles*3*sizeof(float), cudaMemcpyHostToDevice);
    initGrid(P(1), P(0), nParticles);

    P(1).unmap();
    P(0).unmap();
}


SimulationCUDA::~SimulationCUDA()
{
    for(unsigned n = 0; n < 2; n++)
    {
        delete m_P[n];
        m_P[n] = NULL;
    }
}



// --------------------------------------------------------------------------- 
    
} // END NAMESPACE ZILLION
