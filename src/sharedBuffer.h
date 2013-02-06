#ifndef _zillion_sharedBuffer_h
#define _zillion_sharedBuffer_h

#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include "cudaUtils.h"

namespace Zillion {

template<class T>
class SharedBuffer
{
public:
    SharedBuffer(GLenum target, size_t nSize, GLenum usage);
    virtual ~SharedBuffer();
    
    T* map();
    void unmap();
    
    void bind();
    
    size_t size() const {return m_nSize;}
    operator T*() const {return m_pDevice;}
    
protected:
    GLenum m_target;
    size_t m_nSize;
    
    GLuint m_buffer;
    cudaGraphicsResource* m_pRes;
    T* m_pDevice;
};


template<class T>
SharedBuffer<T>::SharedBuffer(GLenum target, size_t nSize, GLenum usage):
m_target(target), m_nSize(0), m_buffer(0), m_pRes(NULL), m_pDevice(NULL)
{
    glGenBuffers(1, &m_buffer);
    glBindBuffer(m_target, m_buffer);
    glBufferData(m_target, nSize*sizeof(T), NULL, usage);
    
    cudaCheckError( cudaGraphicsGLRegisterBuffer(&m_pRes,
                                                 m_buffer,
                                                 cudaGraphicsMapFlagsNone) );
};


template<class T>
SharedBuffer<T>::~SharedBuffer()
{
    unmap();
    
    ::cudaGraphicsUnregisterResource(m_pRes);
    glDeleteBuffers(1, &m_buffer);
};
    

template<class T>
T*
SharedBuffer<T>::map()
{
    size_t s;
            
    cudaCheckError( ::cudaGraphicsMapResources(1, &m_pRes) );
    cudaCheckError( cudaGraphicsResourceGetMappedPointer((void**)&m_pDevice,
                                                         &s, m_pRes) );

    return (T*)m_pDevice;
};


template<class T>
void
SharedBuffer<T>::unmap()
{
    if(m_pDevice != NULL)
    {
        ::cudaGraphicsUnmapResources(1, &m_pRes);
        m_pDevice = NULL;
    }
};


template<class T>
void
SharedBuffer<T>::bind()
{
    glBindBuffer(m_target, m_buffer);
};



} // END NAMESPACE ZILLION

#endif
