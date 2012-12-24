#include <iostream>

#include "chequerTexture.h"

namespace Zillion {

ChequerTexture::ChequerTexture(unsigned size):
m_texture(0)
{
    glGenTextures(1, &m_texture);
    
    std::cout << "Texture: " << m_texture << std::endl;
    
    glBindTexture(GL_TEXTURE_2D, m_texture);
        
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    
    // Mipmapping
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
    // A black and white chequerboard
    const GLfloat v0 = 0.8f, v1 = 0.9f;
    
    float pixels[] =
    {
        v0, v0, v0,   v1, v1, v1, 
        v1, v1, v1,   v0, v0, v0
    };
    
    //glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, 2, 2, 0, GL_RGB, GL_FLOAT, NULL);
    //glGenerateMipmap(GL_TEXTURE_2D);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, 2, 2, 0, GL_RGB, GL_FLOAT, pixels);
    
}


ChequerTexture::~ChequerTexture()
{
    glDeleteTextures(1, &m_texture);
}
    

void
ChequerTexture::bind(GLuint textureUnit) const
{
    glActiveTexture( GL_TEXTURE0 + textureUnit );
    glBindTexture(GL_TEXTURE_2D, m_texture);
}


} // END NAMESPACE ZILLION
