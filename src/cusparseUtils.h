#ifndef _zillion_cusparseUtils_h
#define _zillion_cusparseUtils_h

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
    static const char* szStatusDesc[] =
    {
        "Success",
        "Not initialized",
        "Allocation failed",
        "Invalid value",
        "Architecture mismatch",
        "Mapping error",
        "Execution failed",
        "Internal error",
        "Matrix type not supported"
    };
    
    if(status != CUSPARSE_STATUS_SUCCESS)
    {
        const char* szDesc = "Unknown error";
       
        if(int(status) < sizeof(szStatusDesc)/sizeof(char*))
            szDesc = szStatusDesc[int(status)];
        
        std::cerr << "CUSPARSE ERROR: " << szDesc << std::endl
                  << szFile << " at line " << nLine << std::endl;
        exit(EXIT_FAILURE);
    }
}


// ---------------------------------------------------------------------------


} // END NAMESPACE ZILLION

#endif
