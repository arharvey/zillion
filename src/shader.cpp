#include <iostream>
#include <fstream>

#include "shader.h"

namespace Zillion {

Shader::Shader(const char* szFile, GLuint shaderType):
m_id(0), m_bValid(false)
{
    std::ifstream source;
    source.open(szFile);
    
    source.seekg(0, std::ios::end);
    int length = source.tellg();
    source.seekg(0, std::ios::beg);
    
    char* szSource = new char[length+1];
    memset(szSource, 0, length+1);
    
    source.read(szSource, length);
    
    m_id = glCreateShader(shaderType);
    glShaderSource(m_id, 1, (const char**)&szSource, NULL);
    
    glCompileShader(m_id);
    
    GLint status;
    glGetShaderiv(m_id, GL_COMPILE_STATUS, &status);
    
    if(status == GL_TRUE)
        m_bValid = true;
    else
    {
        char buffer[1024];
        glGetShaderInfoLog(m_id, 1024, NULL, buffer);
        
        std::cerr << "Error compiling shader '" << szFile << "':" << std::endl;
        std::cerr << buffer << std::endl;
    }
    
    delete [] szSource;
}

} // END NAMESPACE ZILLION


    