#ifndef _zillion_camera_h
#define _zillion_camera_h

#include <OpenEXR/ImathVec.h>
#include <OpenEXR/ImathMatrix.h>

namespace Zillion {
    
class Camera
{
public:
    Camera();
    virtual ~Camera();
    
    virtual const Imath::M44f& cameraTransform() const =0;
    virtual const Imath::M44f& viewTransform() const =0;
    
protected:
};


// ---------------------------------------------------------------------------

class TumbleCamera : public Camera
{
public:
    TumbleCamera();
    
    virtual const Imath::M44f& cameraTransform() const;
    virtual const Imath::M44f& viewTransform() const;
    
    const Imath::V3f& center() const {return m_center;}
    float distance() const {return m_distance;}
    float azimuth() const {return m_azimuth;}
    float altitude() const {return m_altitude;}
    
    void setCenter(const Imath::V3f& c);
    void setDistance(float dist);
    void scaleDistance(float scale);
    void setAzimuth(float radians);
    void setAltitude(float radians);
    
    
protected:
    void invalidate() {m_bValid = false;}
    void ensureValid() const;
    
    Imath::V3f m_center; /// Tumble center
    float m_distance; /// Distance of viewpoint from center
    float m_azimuth; // Rotation in horizontal plane
    float m_altitude; // Rotation in vertical plane
    
    mutable bool m_bValid; /// Transform is valid
    mutable Imath::M44f m_cameraXf; /// Camera transform
    mutable Imath::M44f m_viewXf; /// View transform
};

    
} // END NAMESPACE ZILLION


#endif
