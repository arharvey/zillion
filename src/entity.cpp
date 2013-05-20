#include "entity.h"

namespace Zillion {
    
// ---------------------------------------------------------------------------

Entity::Entity()
{
    m_xf.makeIdentity();
}


Entity::~Entity()
{
}


// ---------------------------------------------------------------------------


SphereEntity::SphereEntity(float radius):
m_radius(radius)
{
}


bool
SphereEntity::intersect(const Imath::V3f& origin, const Imath::V3f& dir, float& t)
{
    /* We solve the quadratic equation At^2 + Bt + C = 0 with:
     * 
     * A = d.d
     * B = 2(o - c).d
     * C = (o - c).(o - c) - r^2
     * 
     * where
     * 
     * o is ray origin
     * d is ray direction
     * r is sphere radius
     * 
     * See: http://wiki.cgsociety.org/index.php/Ray_Sphere_Intersection
     * 
     */
    
    const float A = dir.dot(dir);
    
    const Imath::V3f disp = origin - xform().translation();
    const float B = 2.0f * disp.dot(dir);
    
    const float C = disp.dot(disp) - m_radius*m_radius;
    
    float D = B*B - 4.0f*A*C;
    if(D > 0.0f)
    {
        // Hit! We are only interested in the closest hit though
        
        D = sqrt(D);
        const float tNear = (-B - D) / (2.0f*A);
        
        if(tNear >= 0.0f)
        {
            t = tNear;
            return true;
        }
    }
    
    return false;
}
    
    
// ---------------------------------------------------------------------------
    
}