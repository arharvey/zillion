#include "program.h"

Program::Program(unsigned nNumUniforms):
m_program(0), m_pUniforms(NULL)
{
    if(nNumUniforms > 0)
    {
        m_pUniforms = new GLint[nNumUniforms];
        memset(m_pUniforms, 0, sizeof(GLint)*nNumUniforms);
    }
}


Program::~Program()
{
    delete [] m_pUniforms;
}


void
Program::use() const
{
    if(m_program)
        glUseProgram(m_program);
}

void
Program::set(unsigned n, const Imath::M44f& M) const
{
    glUniformMatrix4fv(m_pUniforms[n], 1, GL_FALSE, &(M.x[0][0]));
}


void
Program::set(unsigned n, const Imath::M33f& M) const
{
    glUniformMatrix3fv(m_pUniforms[n], 1, GL_FALSE, &(M.x[0][0]));
}


void
Program::set(unsigned n, const Imath::V3f& v) const
{
    glUniform3f(m_pUniforms[n], v.x, v.y, v.z);
}


bool
Program::link(GLuint program, const char* szName)
{
    glLinkProgram(program);

    GLint status = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if(status != GL_TRUE)
    {
        char buffer[1024];
        glGetProgramInfoLog(program, 1024, NULL, buffer);

        std::cerr << "Error linking program '" << szName << "':" << std::endl;
        std::cerr << buffer << std::endl;
        return false;
    }

    return true;
}


void
Program::initUniformLocations(const char** szUniforms, unsigned nNumUniforms)
{
    for(unsigned n = 0; n < nNumUniforms; n++)
        m_pUniforms[n] = glGetUniformLocation(m_program, szUniforms[n]);
}
