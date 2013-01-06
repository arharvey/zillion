#ifndef _zillion_simulation_h
#define _zillion_simulation_h

#include "sharedBuffer.h"

namespace Zillion {
    
// ---------------------------------------------------------------------------
    
class SimulationCUDA
{
public:
    SimulationCUDA(int cudaDevice, const float* Pinit, const float* Vinit,
                   unsigned nParticles, float particleRadius);
    virtual ~SimulationCUDA();
    
    virtual void stepForward(double dt);
    
    SharedBuffer& P() const {return *m_P[m_currentBuffer];}
    
protected:
    SharedBuffer& prevP() const {return *m_P[1-m_currentBuffer];}
    void swapBuffers() {m_currentBuffer = 1-m_currentBuffer;}
    

    const int m_cudaDevice;
    cudaDeviceProp m_cudaProp;
    
    float* m_Fd; /// Particle forces (on GPU)
    float* m_Vd; /// Particle velocities (on GPU)
    SharedBuffer* m_P[2]; /// Particle positions (double buffered, on GPU)
    
    unsigned m_currentBuffer;
    
    unsigned m_nParticles;
    const float m_particleRadius;
};
    
    
    
// --------------------------------------------------------------------------- 
    
} // END NAMESPACE ZILLION

#endif
