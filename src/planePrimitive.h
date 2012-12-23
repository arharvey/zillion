#ifndef _zillion_planePrimitive_h
#define _zillion_planePrimitive_h

#include <GL/glew.h>

#include <OpenEXR/ImathPlane.h>

class PlanePrimitive
{
public:
    enum Buffer
    {
        kPosition = 0,
        kNormal,
        kUV,
        kNumBuffers
    };
    
    PlanePrimitive();
    PlanePrimitive(const Imath::Plane3f& p,
                   const Imath::V3f& uDir = Imath::V3f(1.0, 0.0, 0.0));
    
    virtual ~PlanePrimitive();
    
    void bind() const;
    void unbind() const;
    
    void update(const Imath::M44f& Mf, float near, float far);
    void draw() const;
    
    
protected:
    void init(const Imath::V3f& uDir);
    
    GLuint m_vao; /// Vertex Array Object
    GLuint m_buffer[kNumBuffers]; /// Vertex buffer objects
    
    Imath::Plane3f m_plane;
    Imath::M44f m_uvXf;
};

#endif
