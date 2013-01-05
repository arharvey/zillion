#ifndef _zillion_simulation_h
#define _zillion_simulation_h

#include "sharedBuffer.h"

namespace Zillion {
    
// ---------------------------------------------------------------------------
    
class SimulationCUDA
{
public:
    SimulationCUDA(const float* Pinit, unsigned nParticles,
                   float particleRadius);
    virtual ~SimulationCUDA();
    
    SharedBuffer& P(unsigned n) const {return *m_P[n];}
    
protected:
    SharedBuffer* m_P[2];
    unsigned m_nParticles;
    const float m_particleRadius;
};
    
    
    
// --------------------------------------------------------------------------- 
    
} // END NAMESPACE ZILLION

#endif
