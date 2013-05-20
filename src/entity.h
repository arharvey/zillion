#ifndef _zillion_entity_h
#define _zillion_entity_h

#include <OpenEXR/ImathVec.h>
#include <OpenEXR/ImathMatrix.h>

namespace Zillion {

/// An Entity is a draggable object that can be manipulated by the user
class Entity
{
    public:
        Entity();
        virtual ~Entity();
        
        void setXform(const Imath::M44f& xf) {m_xf = xf;}
        const Imath::M44f& xform() const {return m_xf;}
        Imath::M44f& xform() {return m_xf;}
        
        /// Return true on intersection, with t containing hit coefficient.
        virtual bool intersect(const Imath::V3f& origin,
                               const Imath::V3f& dir, float& t) =0;
        
    private:
        Imath::M44f m_xf;
};


// ---------------------------------------------------------------------------

class SphereEntity : public Entity
{
    public:
        SphereEntity(float radius);
        
        void setRadius(float r) {m_radius = r;}
        float radius() const {return m_radius;}
        
        virtual bool intersect(const Imath::V3f& origin,
                               const Imath::V3f& dir, float& t);
        
    private:
        float m_radius;
};


// ---------------------------------------------------------------------------

} // END NAMESPACE ZILLION

#endif
