#ifndef _zillion_spherePrimitive_h
#define _zillion_spherePrimitive_h

#include <GL/glew.h>

namespace Zillion {

/// Initialise a Sphere primitive
class SpherePrimitive
{
public:
    enum Buffer
    {
        kVertex = 0,
        kElement = 1,
        kBuffers
    };
    
    SpherePrimitive(GLfloat radius, unsigned subdivU, unsigned subdivV);
    virtual ~SpherePrimitive();
    
    void bind() const;
    void unbind() const;
    
    void draw() const;
    void drawInstances(unsigned nPts) const;
    
protected:
    const GLfloat m_radius; /// Radius
    const unsigned m_subdivU; /// Subdivisions in U direction
    const unsigned m_subdivV; /// Subdivisions in V direction
    
    GLfloat* m_pVertices; /// Vertices
    GLuint* m_pElements; /// Mesh connectivity
    unsigned m_nElements; /// Number of elements
    
    GLuint m_vao; /// Vertex Array Object
    GLuint m_buffer[kBuffers]; // Buffers
};

} // END NAMESPACE ZILLION

#endif