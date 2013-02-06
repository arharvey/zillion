#ifndef _zillion_simulation_h
#define _zillion_simulation_h

#include "sharedBuffer.h"

namespace Zillion {
    
// ---------------------------------------------------------------------------
    
class SimulationCUDA
{
public:
    SimulationCUDA(int cudaDevice, const float3* Pinit, const float3* Vinit,
                   unsigned nParticles, float particleRadius);
    virtual ~SimulationCUDA();
    
    virtual void stepForward(double dt);
    
    SharedBuffer<float3>& P() const {return *m_P[m_currentBuffer];}
    
protected:
    SharedBuffer<float3>& prevP() const {return *m_P[1-m_currentBuffer];}
    void swapBuffers() {m_currentBuffer = 1-m_currentBuffer;}
    

    const int m_cudaDevice;
    cudaDeviceProp m_cudaProp;
    
    float3* m_Fd; /// Particle forces (on GPU)
    float3* m_Vd; /// Particle velocities (on GPU)
    SharedBuffer<float3>* m_P[2]; /// Particle positions (double buffered, on GPU)
    
    float3* m_Wd; // Scratch space for reduction operations
    
    unsigned m_currentBuffer;
    
    unsigned m_nParticles;
    const float m_particleRadius;
};
    
    
    
// --------------------------------------------------------------------------- 
    
} // END NAMESPACE ZILLION

#endif
