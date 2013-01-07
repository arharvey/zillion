#include <iostream>

#include "chequerTexture.h"

namespace Zillion {

ChequerTexture::ChequerTexture(unsigned size):
m_texture(0)
{
    glGenTextures(1, &m_texture);
    
    std::cout << "Texture: " << m_texture << std::endl;
    
    glBindTexture(GL_TEXTURE_2D, m_texture);
    
    // A black and white chequerboard
    
    const GLfloat values[2] = {0.7f, 0.8f};
    
    const unsigned nDim = 1;
    const unsigned nSize = 1<<nDim;
    const unsigned nHalfSize = nSize>>1;
    
    float* pixels = new GLfloat[nSize*nSize*3];
    
    float* p = pixels;
    for(unsigned y = 0; y < nSize; y++)
    {
        for(unsigned x = 0; x < nSize; x++, p += 3)
        {
            GLfloat v = values[(x^y) >> (nDim-1)];
            p[0] = v;
            p[1] = v;
            p[2] = v;
        }
    }
    
    //glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, nSize, nSize, 0, GL_RGB, GL_FLOAT, NULL);
    //glGenerateMipmap(GL_TEXTURE_2D);
    glTexImage2D (GL_TEXTURE_2D, 0, GL_RGB, nSize, nSize, 0, GL_RGB, GL_FLOAT, pixels);
    
    delete [] pixels;
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    
    // Mipmapping
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
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
