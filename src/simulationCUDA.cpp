#include <iostream>
#include <algorithm>
#include <cmath>
#include <assert.h>

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

    cudaCheckError( cudaMalloc( &m_Fd, nParticles*sizeof(float3) ) );
    cudaMemset( m_Fd, 0, nParticles*sizeof(float3) );
    
    cudaCheckError( cudaMalloc( &m_Vd, nParticles*sizeof(float3) ) );
    cudaMemcpy(m_Vd, Vinit, nParticles*sizeof(float3), cudaMemcpyHostToDevice);
    
    int workSize = reduceWorkSize(nParticles, m_cudaProp) * sizeof(float3);
    cudaCheckError( cudaMalloc( &m_Wd, workSize) );
    
    // Reserve space for our collision grid
    
    int nCollisionGridSize = COLLISION_GRID_MAX_SIZE * 1024 * 1024; // Bytes
    m_nMaxCells = nCollisionGridSize / (sizeof(int) * (1 + MAX_OCCUPANCY));
    
    cudaCheckError( cudaMalloc( &m_GNd, m_nMaxCells*sizeof(int) ) );
    cudaCheckError( cudaMalloc( &m_Gd, m_nMaxCells*MAX_OCCUPANCY*sizeof(int) ) );
    
#ifdef SANITY_CHECK_COLLISION_GRID
    m_GNh = new int[m_nMaxCells];
    m_Gh = new int[m_nMaxCells*MAX_OCCUPANCY];
#endif
            
    std::cout << "Reserved " << m_nMaxCells << " grid cells ("
              << COLLISION_GRID_MAX_SIZE << "MB) for collision detection" << std::endl;
}


SimulationCUDA::~SimulationCUDA()
{
    cudaFree(m_Gd);
    cudaFree(m_GNd);
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
    
    accumulateForces(m_Fd, m_nParticles, MASS, GRAVITY, m_cudaProp);
    
    forwardEulerSolve(P(), m_Vd, prevP(), m_Fd, m_nParticles, MASS, dt,
                      m_cudaProp);
    
    handlePlaneCollisions(P(), m_Vd, prevP(), m_nParticles, m_particleRadius,
                          dt, RESTITUTION, m_cudaProp);
    
    // Get extents
    float3 minExtent, maxExtent;
    
    cudaMemcpy(m_Wd, P(), m_nParticles*sizeof(float3), cudaMemcpyDeviceToDevice);
    minFloat3(minExtent, m_Wd, m_nParticles, m_cudaProp);
    
    cudaMemcpy(m_Wd, P(), m_nParticles*sizeof(float3), cudaMemcpyDeviceToDevice);
    maxFloat3(maxExtent, m_Wd, m_nParticles, m_cudaProp);
    
    // Work out extents of particle system
    const float3 halfParticle = make_float3(m_particleRadius,
                                            m_particleRadius,
                                            m_particleRadius);
    
    minExtent -= halfParticle;
    maxExtent += halfParticle;
    
    float3 range = maxExtent - minExtent;
    
    // Ensure we have enough space to store collision grid and reset it
    
    const float cellSize = m_particleRadius;
    
    int3 collDims;
    collDims.x = ceil(range.x / cellSize);
    collDims.y = ceil(range.y / cellSize);
    collDims.z = ceil(range.z / cellSize);
    
    int nCells = collDims.x * collDims.y * collDims.z;
    assert(nCells > 0);
    assert(nCells < m_nMaxCells);
    
    const float usage = float(nCells)/float(m_nMaxCells);
    if(usage >= 0.9)
    {
        std::cout << "Warning: " << 100.0f*usage
                  << "% of collision grid used" << std::endl;
    }
    
    cudaMemset(m_GNd, 0, nCells*sizeof(int));
    
    // Populate collision grid with particles
    populateCollisionGrid(m_Gd, m_GNd, P(), m_nParticles,
                          minExtent, collDims, cellSize, m_cudaProp);
    
    // DEBUG: Sanity check grid
#if SANITY_CHECK_COLLISION_GRID
    sanityCheckCollisionGrid(nCells);
#endif
    
    P().unmap();
    prevP().unmap();
    
}


void
SimulationCUDA::sanityCheckCollisionGrid(int nCells)
{
    cudaCheckError( cudaDeviceSynchronize() );
    
    cudaMemcpy(m_GNh, m_GNd, nCells*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(m_Gh, m_Gd, nCells*MAX_OCCUPANCY*sizeof(int), cudaMemcpyDeviceToHost);
    
    int nAccountedParticles = 0;
    int nOverflowCells = 0, nOverflowParticles = 0;
    for(int n = 0; n < nCells; n++)
    {
        const int nCount = m_GNh[n];
        if(nCount < 0)
        {
            std::cerr << "ERROR: Cell occupancy count must be >= 0. Found "
                      << nCount << std::endl;
            
            exit(EXIT_FAILURE);
        }
        
        
        if(nCount > MAX_OCCUPANCY)
        {
            nOverflowCells++;
            nOverflowParticles += (nCount - MAX_OCCUPANCY);
        }
      
        for(int i = 0; i < std::min<int>(nCount, MAX_OCCUPANCY); i++)
        {
            const int* pCellIds = m_Gh + n*MAX_OCCUPANCY;
            const int id = pCellIds[i];
            
            // Check particle IDs
            if(id < 0 || id >= m_nParticles)
            {
                std::cerr << "ERROR: Illegal particle ID. Found "
                      << id << " (max "<< m_nParticles << ") in cell "
                      << n << "/" << i << std::endl;
                
                exit(EXIT_FAILURE);
            }
            
        }
        
        nAccountedParticles += nCount;
    }
    
    if(nOverflowCells > 0)
    {
        std::cerr << "Warning: Ignoring " << nOverflowCells << " particles. "
                  << nOverflowCells << " cells exceed the maximum limit of "
                  << MAX_OCCUPANCY << " particles" << std::endl;
    }
    
    if(nAccountedParticles != m_nParticles)
    {
        std::cerr << "ERROR: Mismatched particles. Expected "
                  << m_nParticles << ", found " << nAccountedParticles
                  << " (" << (nAccountedParticles - m_nParticles) << ")"
                  << std::endl;
        
        exit(EXIT_FAILURE);
    }
}

// --------------------------------------------------------------------------- 
    
} // END NAMESPACE ZILLION
