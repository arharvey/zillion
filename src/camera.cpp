#include "utils.h"

#include "camera.h"




namespace Zillion {

Camera::Camera()
{
}


Camera::~Camera()
{
}


// ---------------------------------------------------------------------------

TumbleCamera::TumbleCamera():
m_center(), m_distance(2.0f), m_azimuth(0.0f), m_altitude(0.0f),
m_bValid(false)
{
    
}


const Imath::M44f&
TumbleCamera::cameraTransform() const
{
    ensureValid();
    
    return m_cameraXf;
}


const Imath::M44f&
TumbleCamera::viewTransform() const
{
    ensureValid();
    
    return m_viewXf;
}
    

void
TumbleCamera::setCenter(const Imath::V3f& c)
{
    m_center = c;
    invalidate();
}


void
TumbleCamera::setDistance(float dist)
{
    m_distance = dist;
    invalidate();
}


void
TumbleCamera::scaleDistance(float scale)
{
    m_distance *= scale;
    if(m_distance < 0.1)
        m_distance = 0.1;
    
    invalidate();
}


void
TumbleCamera::setAzimuth(float a)
{
    m_azimuth = a;
    invalidate();
}


void
TumbleCamera::setAltitude(float a)
{
    m_altitude = a;
    invalidate();
}
    

void
TumbleCamera::ensureValid() const
{
    if(m_bValid)
        return;
    
    m_cameraXf.makeIdentity();
    m_cameraXf.translate(Imath::V3f(0.0, 0.0, m_distance));
    m_cameraXf *= Imath::M44f().setEulerAngles(Imath::V3f(-m_altitude, m_azimuth, 0.0) );
    m_cameraXf *= Imath::M44f(Imath::M33f(), m_center);
    
    m_viewXf = m_cameraXf.inverse();
    
    m_bValid = true;
}

} // END NAMESPACE ZILLION
