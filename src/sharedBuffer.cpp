#include <cuda_runtime.h>

#include "sharedBuffer.h"

namespace Zillion {

SharedBuffer::SharedBuffer(GLenum target, size_t nSize, GLenum usage):
m_target(target), m_nSize(0), m_buffer(0), m_pRes(NULL), m_pDevice(NULL)
{
    glGenBuffers(1, &m_buffer);
    glBindBuffer(m_target, m_buffer);
    glBufferData(m_target, nSize*sizeof(GLfloat), NULL, usage);
    cudaGraphicsGLRegisterBuffer(&m_pRes, m_buffer, cudaGraphicsMapFlagsNone);
}


SharedBuffer::~SharedBuffer()
{
    unmap();
    
    ::cudaGraphicsUnregisterResource(m_pRes);
    glDeleteBuffers(1, &m_buffer);
}
    

float*
SharedBuffer::map()
{
    size_t s;
            
    ::cudaGraphicsMapResources(1, &m_pRes);
    cudaGraphicsResourceGetMappedPointer((void**)&m_pDevice, &s, m_pRes);

    return m_pDevice;
}


void
SharedBuffer::unmap()
{
    if(m_pDevice != NULL)
    {
        ::cudaGraphicsUnmapResources(1, &m_pRes);
        m_pDevice = NULL;
    }
}


void
SharedBuffer::bind()
{
    glBindBuffer(m_target, m_buffer);
}

} // END NAMESPACE ZILLION
