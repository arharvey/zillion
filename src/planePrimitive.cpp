#include "planePrimitive.h"


PlanePrimitive::PlanePrimitive():
m_vao(0), m_plane(Imath::V3f(0.0, 1.0, 0.0), 0.0)
{
    init(Imath::V3f(1.0, 0.0, 0.0));
}


PlanePrimitive::PlanePrimitive(const Imath::Plane3f& p, const Imath::V3f& uDir):
m_vao(0), m_plane(p)
{
    init(uDir);
}


void
PlanePrimitive::init(const Imath::V3f& uDir)
{
    glGetError(); // Reset
    
    glGenVertexArrays(1, &m_vao);
    GLenum error = glGetError();
    if(error != GL_NO_ERROR)
    {
        std::cerr << "ERROR 0x" << std::hex << error << std::endl;   
    }
    
    
    glBindVertexArray(m_vao);
    
    glGenBuffers(kNumBuffers, m_buffer);
    
    const unsigned MAX_INTERSECT = 12;
    
    // Reserve space for dynamic vertex arrays
    
    glBindBuffer(GL_ARRAY_BUFFER, m_buffer[kPosition]);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0); // Position attribute
    
    glBindBuffer(GL_ARRAY_BUFFER, m_buffer[kNormal]);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1); // Normal attribute
    
    glBindBuffer(GL_ARRAY_BUFFER, m_buffer[kUV]);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(2); // UV attribute
    
    glBindBuffer(GL_ARRAY_BUFFER,0);
    glBindVertexArray(0);
    
    const Imath::V3f& n = m_plane.normal;
    
    Imath::V3f o = n * m_plane.distance;
    Imath::V3f v = uDir.normalized() % n;
    Imath::V3f u = n % v;
    
    m_uvXf = Imath::M44f(Imath::M33f(u.x, v.x, 0,
                                     u.y, v.y, 0,
                                     u.z, v.z, 0),
                         Imath::V3f(-o.x, -o.z, 0));
}


PlanePrimitive::~PlanePrimitive()
{
    glDeleteBuffers(kNumBuffers, m_buffer);
    glDeleteVertexArraysAPPLE(1, &m_vao);
}


void
PlanePrimitive::update(const Imath::M44f& Mf, float near, float far)
{
    const unsigned MAX_INTERSECT = 12;
    
    Imath::V4f vertices[MAX_INTERSECT];
    
    const GLfloat s = 100;
    
    vertices[0] = Imath::V4f(-s, 0, s, 1.0);
    vertices[1] = Imath::V4f(s, 0, s, 1.0);
    vertices[2] = Imath::V4f(s, 0, -s, 1.0);
    vertices[3] = Imath::V4f(-s, 0, -s, 1.0);
    
    // Submit geometry
    
    unsigned nVertices = 4;
    
    GLfloat positions[MAX_INTERSECT*3];
    GLfloat normals[MAX_INTERSECT*3];
    GLfloat uvs[MAX_INTERSECT*2];
    
    GLfloat* p = positions;
    for(unsigned n=0; n < nVertices; n++)
    {
        p[0] = vertices[n].x;
        p[1] = vertices[n].y;
        p[2] = vertices[n].z;
        
        p += 3;
    }
    
    p = normals;
    
    const Imath::V3f& norm = m_plane.normal;
    for(unsigned n=0; n < nVertices; n++)
    {
        p[0] = norm.x;
        p[1] = norm.y;
        p[2] = norm.z;
        
        p += 3;
    }
    
    
    p = uvs;
    
    for(unsigned n=0; n < nVertices; n++)
    {
        Imath::V4f uv = vertices[n] * m_uvXf;
        
        p[0] = uv.x;
        p[1] = -uv.y;
        
        //std::cout << p[0] << ", " << p[1] << std::endl;
        
        p += 2;
    }
    
    //std::cout << std::endl;
    glBindVertexArray(m_vao);
    
    glBindBuffer(GL_ARRAY_BUFFER, m_buffer[kPosition]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*nVertices*3,
                    positions, GL_DYNAMIC_DRAW);
    
    glBindBuffer(GL_ARRAY_BUFFER, m_buffer[kNormal]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*nVertices*3,
                    normals, GL_DYNAMIC_DRAW);
    
    glBindBuffer(GL_ARRAY_BUFFER, m_buffer[kUV]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*nVertices*2,
                    uvs, GL_DYNAMIC_DRAW);
    
    glBindBuffer(GL_ARRAY_BUFFER,0);
    glBindVertexArray(0);
}


void
PlanePrimitive::bind() const
{
    glBindVertexArray(m_vao);
}


void
PlanePrimitive::unbind() const
{
    glBindVertexArray(0);
}


void
PlanePrimitive::draw() const
{
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
}