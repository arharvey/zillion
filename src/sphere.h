#ifndef _zillion_sphere_h
#define _zillion_sphere_h

#include <GL/glew.h>

namespace Zillion {

/// Initialise a Sphere primitive
class Sphere
{
public:
    enum Buffer
    {
        kVertex = 0,
        kElement = 1,
        kBuffers
    };
    
    Sphere(GLfloat radius, unsigned subdivU, unsigned subdivV);
    virtual ~Sphere();
    
    GLuint buffer(Buffer id) const {return m_buffer[id];}
    
    void draw() const;
    void drawInstances(unsigned nPts) const;
    
protected:
    const GLfloat m_radius; /// Radius
    const unsigned m_subdivU; /// Subdivisions in U direction
    const unsigned m_subdivV; /// Subdivisions in V direction
    
    GLfloat* m_pVertices; /// Vertices
    GLuint* m_pElements; /// Mesh connectivity
    unsigned m_nElements; /// Number of elements
    
    GLuint m_buffer[kBuffers]; // Buffers
};

} // END NAMESPACE ZILLION

#endif