#ifndef _zillion_utils_h
#define _zillion_utils_h

#include <math.h>

#define CHECK_GL() assert(glGetError() == 0)

namespace Zillion {
    
template<class T>
T
toRadians(const T degrees)
{
    return degrees*(M_PI/180.0);
}
   

inline
size_t
inMB(size_t bytes)
{
    return bytes / (1024 * 1024);
}


inline
const char*
yesNo(bool b)
{
    return b  ? "Yes" : "No";
}


} // END NAMESPACE ZILLION

#endif
