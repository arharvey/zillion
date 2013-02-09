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
m_nParticles(0), m_particleRadius(particleRadius)

#ifdef SANITY_CHECK_COLLISION_GRID
, m_nCells(0), m_GNh(NULL), m_Gh(NULL)
#endif
{
    cudaGetDeviceProperties(&m_cudaProp, m_cudaDevice);
    
    // Reserve space for our collision grid
    
    int nCollisionGridSize = COLLISION_GRID_MAX_SIZE * 1024 * 1024; // Bytes
    m_nMaxCells = nCollisionGridSize / (sizeof(int) * (1 + MAX_OCCUPANCY));
    
    cudaCheckError( cudaMalloc( &m_GNd, m_nMaxCells*sizeof(int) ) );
    cudaCheckError( cudaMalloc( &m_Gd, m_nMaxCells*MAX_OCCUPANCY*sizeof(int) ) );
            
    std::cout << "Reserved " << m_nMaxCells << " grid cells ("
              << COLLISION_GRID_MAX_SIZE << "MB) for collision detection" << std::endl;
    
#ifdef SANITY_CHECK_COLLISION_GRID
    m_GNh = new int[m_nMaxCells];
    m_Gh = new int[m_nMaxCells*MAX_OCCUPANCY];
#endif
    
    for(unsigned n = 0; n < 2; n++)
        m_P[n] = NULL;
    
    resetParticles(Pinit, Vinit, nParticles);
}


SimulationCUDA::~SimulationCUDA()
{
    cleanupParticles();
    
#if SANITY_CHECK_COLLISION_GRID
    dumpCollisionGrid();
    
    delete [] m_GNh;
    delete [] m_Gh;
#endif
}


// --------------------------------------------------------------------------- 


void
SimulationCUDA::allocateParticles(unsigned nParticles)
{    
    assert(m_nParticles == 0);
    m_nParticles = nParticles;
    
    for(unsigned n = 0; n < 2; n++)
    {
        m_P[n] = new SharedBuffer<float3>(GL_ARRAY_BUFFER, nParticles,
                                          GL_DYNAMIC_DRAW);
    }

    cudaCheckError( cudaMalloc( &m_Fd, nParticles*sizeof(float3) ) );
    cudaCheckError( cudaMalloc( &m_Vd, nParticles*sizeof(float3) ) );
    
    int workSize = reduceWorkSize(nParticles, m_cudaProp) * sizeof(float3);
    cudaCheckError( cudaMalloc( &m_Wd, workSize) );
}


void
SimulationCUDA::cleanupParticles()
{
    if(m_nParticles == 0)
        return;
    
    cudaFree(m_Vd);
    cudaFree(m_Fd);
    cudaFree(m_Wd);
    
    m_Vd = NULL;
    m_Fd = NULL;
    m_Wd = NULL;
    
    for(unsigned n = 0; n < 2; n++)
    {
        delete m_P[n];
        m_P[n] = NULL;
    }
    
    m_nParticles = 0;
}


void
SimulationCUDA::resetParticles(const float3* Pinit, const float3* Vinit,
                               unsigned nParticles)
{
    if(m_nParticles != nParticles)
    {
        cleanupParticles();
        allocateParticles(nParticles);
    }
    
    m_currentBuffer = 0;
    
    P().map();
    cudaMemcpy(P(), Pinit, nParticles*sizeof(float3), cudaMemcpyHostToDevice);
    P().unmap();
    
    cudaCheckError( cudaMemset(m_Fd, 0, nParticles*sizeof(float3)) );
    cudaCheckError( cudaMemcpy(m_Vd, Vinit, nParticles*sizeof(float3),
                               cudaMemcpyHostToDevice) );
    
#if SANITY_CHECK_COLLISION_GRID
    m_nCells = 0;
#endif
}


void
SimulationCUDA::stepForward(double dt)
{
    if(m_nParticles == 0)
        return;
    
    swapBuffers();
    
    prevP().map();
    P().map();
    
    accumulateForces(m_Fd, m_nParticles, MASS, GRAVITY, m_cudaProp);
    
    // Get extents
    float3 minExtent, maxExtent;
    
    cudaMemcpy(m_Wd, prevP(), m_nParticles*sizeof(float3), cudaMemcpyDeviceToDevice);
    minFloat3(minExtent, m_Wd, m_nParticles, m_cudaProp);
    
    cudaMemcpy(m_Wd, prevP(), m_nParticles*sizeof(float3), cudaMemcpyDeviceToDevice);
    maxFloat3(maxExtent, m_Wd, m_nParticles, m_cudaProp);
    
    const float cellSize = 2.0f * m_particleRadius;
    
    // Work out extents of particle system
    const float3 gridPadding = make_float3(m_particleRadius + cellSize,
                                            m_particleRadius + cellSize,
                                            m_particleRadius + cellSize);
    
    minExtent -= gridPadding;
    maxExtent += gridPadding;
    
    float3 range = maxExtent - minExtent;
    
    // Ensure we have enough space to store collision grid and reset it
    
    int3 collDims;
    collDims.x = ceil(range.x / cellSize);
    collDims.y = ceil(range.y / cellSize);
    collDims.z = ceil(range.z / cellSize);
    
    const int nCells = collDims.x * collDims.y * collDims.z;
    assert(nCells > 0);
    assert(nCells <= m_nMaxCells);
    
    //std::cout << collDims << std::endl;
    
    const float usage = float(nCells)/float(m_nMaxCells);
    if(usage >= 0.9)
    {
        std::cout << "Warning: " << 100.0f*usage
                  << "% of collision grid used" << std::endl;
    }
    
    cudaMemset(m_GNd, 0, nCells*sizeof(int));
    
    // Populate collision grid with particles
    populateCollisionGrid(m_Gd, m_GNd, prevP(), m_nParticles,
                          minExtent, collDims, cellSize, m_cudaProp);
    
    // DEBUG: Sanity check grid
#if SANITY_CHECK_COLLISION_GRID
    sanityCheckCollisionGrid(nCells, collDims);
#endif
    
    resolveCollisions(m_Fd, m_Gd, m_GNd, prevP(), m_Vd, m_nParticles,
                      minExtent, collDims, cellSize, m_particleRadius, m_cudaProp);
    
    const float4 groundPlane = make_float4(0.0, 1.0, 0.0, 0.0);
    handlePlaneCollisions(prevP(), m_Vd, m_Fd, m_nParticles, m_particleRadius,
                          groundPlane,
                          RESTITUTION, KINETIC_FRICTION, m_cudaProp);
    
    
    
    const float _1_R2 = 1.0f/sqrtf(2.0);
    const float d = 0.1/_1_R2;
    
    float4 rampPlane = make_float4(-_1_R2, _1_R2, 0.0, -d+1);
    handlePlaneCollisions(prevP(), m_Vd, m_Fd, m_nParticles, m_particleRadius,
                          rampPlane, RESTITUTION, KINETIC_FRICTION, m_cudaProp);
    
    rampPlane = make_float4(_1_R2, _1_R2, 0.0, -d+1);
    handlePlaneCollisions(prevP(), m_Vd, m_Fd, m_nParticles, m_particleRadius,
                          rampPlane, RESTITUTION, KINETIC_FRICTION, m_cudaProp);
    
    rampPlane = make_float4(0, _1_R2, _1_R2, -d+0.5);
    handlePlaneCollisions(prevP(), m_Vd, m_Fd, m_nParticles, m_particleRadius,
                          rampPlane, RESTITUTION, KINETIC_FRICTION, m_cudaProp);
    
    rampPlane = make_float4(0, _1_R2, -_1_R2, -d+0.5);
    handlePlaneCollisions(prevP(), m_Vd, m_Fd, m_nParticles, m_particleRadius,
                          rampPlane, RESTITUTION, KINETIC_FRICTION, m_cudaProp);
    
    forwardEulerSolve(P(), m_Vd, prevP(), m_Fd, m_nParticles, MASS, dt,
                      m_cudaProp);
    
    
    P().unmap();
    prevP().unmap();
    
}


void
SimulationCUDA::sanityCheckCollisionGrid(int nCells, const int3& collDims)
{
    m_nCells = nCells;
    m_collDims = collDims;
    
    cudaCheckError( cudaDeviceSynchronize() );
    
    cudaMemcpy(m_GNh, m_GNd, m_nCells*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(m_Gh, m_Gd, m_nCells*MAX_OCCUPANCY*sizeof(int), cudaMemcpyDeviceToHost);
    
    int nAccountedParticles = 0;
    int nOverflowCells = 0, nOverflowParticles = 0;
    for(int n = 0; n < m_nCells; n++)
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


void
SimulationCUDA::dumpCollisionGrid()
{
    const int strideY = m_collDims.x;
    const int strideZ = m_collDims.x*m_collDims.y;
   
    //std::cout << "Particle radius:" << m_particleRadius << std::endl;
    
    int n = 0;
    for(int k = 0; k < m_collDims.z; k++)
    {
        for(int j = 0; j < m_collDims.y; j++)
        {
            for(int i = 0; i < m_collDims.x; i++, n++)
            {
                int count = m_GNh[n];
                if(count > 0)
                {
                    std::cout << "[" << i << ", " << j << ", " << k << "]: ("
                            << count << ") ";
                    
                    count = std::min(count, int(MAX_OCCUPANCY));
                    const int* ids = m_Gh + n*MAX_OCCUPANCY;
                    for(int c = 0; c < count; c++)
                        std::cout << ids[c] << " ";
                    
                    std::cout << std::endl;
                    
                }
            }
        }
    }
}


// --------------------------------------------------------------------------- 
    
} // END NAMESPACE ZILLION
