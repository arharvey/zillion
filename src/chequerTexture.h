#ifndef _zillion_chequerTexture_h
#define _zillion_chequerTexture_h

#include <GL/glew.h>

namespace Zillion {

class ChequerTexture
{
public:
    ChequerTexture(unsigned size);
    virtual ~ChequerTexture();
    
    void bind(GLuint textureUnit) const;
    
protected:
    GLuint m_texture;
};

} // END NAMESPACE ZILLION

#endif
