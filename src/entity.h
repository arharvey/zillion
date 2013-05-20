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
        
        const Imath::M44f& xform() const {return m_xf;}
        
        virtual const Imath::V3f& velocity() const=0;
        
        virtual void updateDynamics(float dt) =0;
        
        /// Return true on intersection, with t containing hit coefficient.
        virtual bool intersect(const Imath::V3f& origin,
                               const Imath::V3f& dir, float& t) =0;
        
    protected:
        Imath::M44f m_xf;
};


// ---------------------------------------------------------------------------

class SphereEntity : public Entity
{
    public:
        SphereEntity(const Imath::V3f& c, float radius);
        
        void setCenter(const Imath::V3f& c) {m_xf.setTranslation(c);}
        const Imath::V3f center() const {return m_xf.translation();}
        
        void setRadius(float r) {m_radius = r;}
        float radius() const {return m_radius;}
        
        const Imath::V3f& velocity() const {return m_velocity;}
        
        virtual void updateDynamics(float dt);
        
        virtual bool intersect(const Imath::V3f& origin,
                               const Imath::V3f& dir, float& t);
        
    private:
        float m_radius;
        
        Imath::V3f m_prevCenter;
        Imath::V3f m_velocity;
};


// ---------------------------------------------------------------------------

} // END NAMESPACE ZILLION

#endif
