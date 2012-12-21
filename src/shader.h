#ifndef _zillion_shader_h
#define _zillion_shader_h

#include <GL/glew.h>

namespace Zillion {

class Shader
{
public:
    Shader(const char* szFile, GLuint shaderType);
    
    bool valid() const {return m_bValid;}
    operator bool() const {return valid();}
    
    GLuint id() const {return m_id;}
    
    
protected:
    GLuint  m_id;
    bool m_bValid;
};

} // END NAMESPACE ZILLION

#endif