#ifndef _zillion_sphere_h
#define _zillion_sphere_h

#include <GL/glew.h>

namespace Zillion {

/// Initialise a Sphere primitive
class Sphere
{
public:
    Sphere(GLfloat radius, unsigned subdivU, unsigned subdivV);
    virtual ~Sphere();
    
    const GLfloat* vertexBuffer() const {return m_pVertices;}
    const GLuint* indexBuffer() const {return m_pElements;}
    
    void draw() const;
    
protected:
    const GLfloat m_radius; /// Radius
    const unsigned m_subdivU; /// Subdivisions in U direction
    const unsigned m_subdivV; /// Subdivisions in V direction
    
    GLfloat* m_pVertices; /// Vertices
    GLuint* m_pElements; /// Mesh connectivity
    unsigned m_nElements; /// Number of elements
    
    GLuint m_vbo; // Vertex buffer object
    GLuint m_ebo; // Element buffer object
};


} // END NAMESPACE ZILLION

#endif