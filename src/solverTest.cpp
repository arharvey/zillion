#include "solverTest.h"

#include <iostream>

#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include "cusparseUtils.h"

namespace Zillion {
    
// ---------------------------------------------------------------------------
    
void
solverTest(int cudaDevice)
{
    cusparseHandle_t handle = 0;
    
    std::cout << "Solver test" << std::endl;
    
    // Initialize CUSPARSE library
    
    cusparseCheckStatus( cusparseCreate(&handle) );
    
    
    
    cusparseCheckStatus( cusparseDestroy(handle) );
    handle = 0;
}
       

// ---------------------------------------------------------------------------
    
} // END NAMESPACE ZILLION
