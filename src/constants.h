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
    
const unsigned WIDTH = 800;
const unsigned HEIGHT = 600;
    
const float FOV = toRadians(60.0);
    
    
} // END NAMESPACE ANONYMOUS
} // END NAMESPACE ZILLION

#endif
