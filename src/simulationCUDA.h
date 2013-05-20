#ifndef _zillion_simulation_h
#define _zillion_simulation_h

#include "sharedBuffer.h"
#include "entity.h"

#define SANITY_CHECK_COLLISION_GRID 0


namespace Zillion {
    
// ---------------------------------------------------------------------------
    
class SimulationCUDA
{
public:
    SimulationCUDA(int cudaDevice, const float3* Pinit, const float3* Vinit,
                   unsigned nParticles, float particleRadius);
    virtual ~SimulationCUDA();
    
    void resetParticles(const float3* Pinit, const float3* Vinit,
                        unsigned nParticles);
    virtual void stepForward(double dt);
    
    void addCollidable(SphereEntity* pCollidable);
    
    SharedBuffer<float3>& P() const {return *m_P[m_currentBuffer];}
    
protected:
    void allocateParticles(unsigned nParticles);
    void cleanupParticles();
    
    SharedBuffer<float3>& prevP() const {return *m_P[1-m_currentBuffer];}
    void swapBuffers() {m_currentBuffer = 1-m_currentBuffer;}
    

    const int m_cudaDevice;
    cudaDeviceProp m_cudaProp;
    
    float3* m_Fd; /// Particle forces (on GPU)
    float3* m_Vd; /// Particle velocities (on GPU)
    SharedBuffer<float3>* m_P[2]; /// Particle positions (double buffered, on GPU)
    
    float3* m_Wd; // Scratch space for reduction operations
    
    int* m_GNd; // Number of particles in each cell of the collision grid
    int* m_Gd; // Collision grid. Each cell has a list of particles
    int m_nMaxCells;
    
    unsigned m_currentBuffer;
    
    int m_nParticles;
    const float m_particleRadius;
    
    SphereEntity* m_pCollidable;
    
#ifdef SANITY_CHECK_COLLISION_GRID
    void sanityCheckCollisionGrid(int nCells, const int3& collDims);
    void dumpCollisionGrid();
    
    int m_nCells;
    int3 m_collDims;
    int* m_GNh;
    int* m_Gh;
#endif
};
    
    
    
// --------------------------------------------------------------------------- 
    
} // END NAMESPACE ZILLION

#endif
