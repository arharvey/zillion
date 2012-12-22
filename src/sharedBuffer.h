#ifndef _zillion_sharedBuffer_h
#define _zillion_sharedBuffer_h

#include <GL/glew.h>
#include <cuda_gl_interop.h>

namespace Zillion {

class SharedBuffer
{
public:
    SharedBuffer(GLenum target, size_t nSize, GLenum usage);
    virtual ~SharedBuffer();
    
    float* map();
    void unmap();
    
    void bind();
    
    size_t size() const {return m_nSize;}
    operator float*() const {return m_pDevice;}
    
    
protected:
    GLenum m_target;
    size_t m_nSize;
    
    GLuint m_buffer;
    cudaGraphicsResource* m_pRes;
    float* m_pDevice;
};


} // END NAMESPACE ZILLION

#endif
