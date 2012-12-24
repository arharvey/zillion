#include <assert.h>
#include <math.h>
#include <iostream>

#include <OpenEXR/ImathVec.h>

#include "spherePrimitive.h"

namespace Zillion {
  
SpherePrimitive::SpherePrimitive(GLfloat radius, unsigned subdivU, unsigned subdivV):
m_radius(radius), m_subdivU(subdivU), m_subdivV(subdivV),
m_pVertices(NULL), m_pElements(NULL), m_nElements(0), m_vao(0)
{
    assert(subdivU >= 3 && subdivV >= 2);
    assert(radius > 0.0);
    
    const unsigned nVertices = 2 + (subdivU * (subdivV-1));
    const unsigned nComp = 8; // 3 x position, 3 x normal, 2 x UV
    
    // Calculate vertices and UVs
    
    m_pVertices = new GLfloat[nVertices * nComp];
    
    GLfloat* vtx = m_pVertices;
    
    vtx[0] = 0.0;
    vtx[1] = -m_radius;
    vtx[2] = 0.0;
    
    vtx[6] = 0.0; // UVs
    vtx[7] = 0.0; //
    
    vtx += nComp;
    
    const GLfloat toAngleU = 2.0*M_PI / GLfloat(m_subdivU);
    const GLfloat toAngleV = M_PI / GLfloat(m_subdivV);
    for(unsigned v = 1; v < m_subdivV; v++)
    {
        const GLfloat phi = GLfloat(v)*toAngleV - M_PI/2.0;
        
        const GLfloat r = m_radius * cos(phi);
        const GLfloat y = m_radius * sin(phi);
        
        for(unsigned u = 0; u < m_subdivU; u++)
        {
            const GLfloat theta = GLfloat(u)*toAngleU;
            
            vtx[0] = r * cos(theta);
            vtx[1] = y;
            vtx[2] = r * sin(theta);
            
            vtx[6] = GLfloat(u)/GLfloat(m_subdivU);
            vtx[7] = GLfloat(v)/GLfloat(m_subdivV);
            
            vtx += nComp;
        }
    }
    
    vtx[0] = 0.0;
    vtx[1] = m_radius;
    vtx[2] = 0.0;
    
    vtx[6] = 0.0;
    vtx[7] = 1.0;
    
    // Since the sphere is centered on the origin, calculating normals is
    // trivial.
    
    vtx = m_pVertices;
    for(unsigned n = 0; n < nVertices; n++)
    {
        Imath::V3f norm(vtx[0], vtx[1], vtx[2]);
        norm.normalize();
        
        vtx[3] = norm.x;
        vtx[4] = norm.y;
        vtx[5] = norm.z;
        
        vtx += nComp;
    }
    
  
    // Mesh connectivity (anti-clockwise winding)

    const unsigned nTriangles = 2*subdivU + 2*(subdivU * (subdivV-2));
    m_nElements = nTriangles*3;
    m_pElements = new GLuint[m_nElements];
    
    GLuint* i = m_pElements;
    
    unsigned start = 1;
    
    for(unsigned u = 0; u < subdivU; u++)
    {
        i[0] = 0; // Bottom vertex
        i[1] = start + u;
        i[2] = start + (u+1)%subdivU;
        
        //std::cout << i[0] << ", " << i[1] << ", " << i[2] << std::endl;
        
        i += 3;
    }
    
    for(unsigned v = 1; v < subdivV-1; v++)
    {
        for(unsigned u = 0; u < subdivU; u++)
        {
            // a, b, c, d form a quad
            const GLuint a = start + u;
            const GLuint b = start + (u+1)%subdivU;
            const GLuint c = b + subdivU;
            const GLuint d = a + subdivU;
            
            i[0] = c;
            i[1] = b;
            i[2] = a;
            
            //std::cout << i[0] << ", " << i[1] << ", " << i[2] << std::endl;
            
            i += 3;
            
            i[0] = d;
            i[1] = c;
            i[2] = a;
            
            //std::cout << i[0] << ", " << i[1] << ", " << i[2] << std::endl;
            
            i += 3;
        }
        
        start += subdivU;
    }
    
    for(unsigned u = 0; u < subdivU; u++)
    {
        i[0] = start + (u+1)%subdivU;
        i[1] = start + u;
        i[2] = nVertices-1; // Top vertex
        
        //std::cout << i[0] << ", " << i[1] << ", " << i[2] << std::endl;
        
        i += 3;
    }
    

    // Upload to graphics card
    glGenVertexArraysAPPLE(1, &m_vao);
    glBindVertexArrayAPPLE(m_vao);
    
    glGenBuffers(kBuffers, m_buffer);
    
    glBindBuffer(GL_ARRAY_BUFFER, m_buffer[kVertex]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*nVertices*nComp,
                    m_pVertices, GL_STATIC_DRAW);
    
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                            nComp*sizeof(GLfloat), 0);
    glEnableVertexAttribArray(0); // Position attribute
    
    
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
                            nComp*sizeof(GLfloat), (void*)(3*sizeof(GLfloat)) );
    glEnableVertexAttribArray(1); // Normal attribute
    
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE,
                            nComp*sizeof(GLfloat), (void*)(6*sizeof(GLfloat)) );
    glEnableVertexAttribArray(2); // UV attribute
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_buffer[kElement]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*nTriangles*3,
                    m_pElements, GL_STATIC_DRAW);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindVertexArrayAPPLE(0);
    
    delete [] m_pElements;
    delete [] m_pVertices;
}


void
SpherePrimitive::bind() const
{
    glBindVertexArrayAPPLE(m_vao);
    
}


void
SpherePrimitive::unbind() const
{
    glBindVertexArrayAPPLE(0);
}


void
SpherePrimitive::draw() const
{
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_buffer[kElement]);
    glDrawElements(GL_TRIANGLES, m_nElements, GL_UNSIGNED_INT, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}


void
SpherePrimitive::drawInstances(unsigned nPts) const
{
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_buffer[kElement]);
    glDrawElementsInstanced(GL_TRIANGLES, m_nElements,
                                      GL_UNSIGNED_INT, 0,
                                      nPts);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}


SpherePrimitive::~SpherePrimitive()
{
    glDeleteBuffers( kBuffers, m_buffer );
    glDeleteVertexArraysAPPLE(1, &m_vao);
}


} // END NAMESPACE ZILLION
