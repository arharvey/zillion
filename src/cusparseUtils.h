#ifndef _zillion_cudaUtils_h
#define _zillion_cudaUtils_h

#include <cstdlib>
#include <iostream>

#include <cusparse_v2.h>

#define cusparseCheckStatus(status) \
Zillion::__cusparseCheckStatus((status), __LINE__, __FILE__)


namespace Zillion {
    
inline
void
__cusparseCheckStatus(cusparseStatus_t status, int nLine, const char* szFile)
{
    if(status != CUSPARSE_STATUS_SUCCESS)
    {
        std::cerr << "CUSPARSE ERROR: "
                  << szFile << " at line " << nLine << std::endl;
        exit(EXIT_FAILURE);
    }
}


// ---------------------------------------------------------------------------


} // END NAMESPACE ZILLION

#endif
