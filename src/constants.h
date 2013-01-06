#ifndef _zillion_constants_h
#define _zillion_constants_h

#include "utils.h"

namespace Zillion {
namespace {

const unsigned GRID_DIM = 20;
const float GRID_SIZE = 1.0;
const float PARTICLE_SIZE_RELATIVE_TO_GRID_CELL = 0.9;
    
const GLfloat NEAR = 0.1;
const GLfloat FAR = 1000;
    
const unsigned WIDTH = 1024;
const unsigned HEIGHT = 768;
    
const float FOV = toRadians(60.0);
    
const float GRAVITY = -4;
const float MASS = 1;
const float RESTITUTION = 0.8;
    
} // END NAMESPACE ANONYMOUS
} // END NAMESPACE ZILLION

#endif
