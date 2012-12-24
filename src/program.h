#ifndef _zillion_program_h
#define _zillion_program_h

#include <GL/glew.h>
#include <OpenEXR/ImathMatrix.h>
#include <OpenEXR/ImathVec.h>

class Program
{
public:
    Program(unsigned nNumUniforms);
    virtual ~Program();
    
    void use() const;
    
    GLuint program() const {return m_program;}
    operator bool() const {return m_program != 0;}
    operator GLuint() const {return program();}
    
    void set(unsigned n, const Imath::M44f& M) const;
    void set(unsigned n, const Imath::M33f& M) const;
    void set(unsigned n, const Imath::V3f& v) const;
    void set(unsigned n, int i) const;
    
    
protected:
    static bool link(GLuint program, const char* szName);
    
    void initUniformLocations(const char** szUniforms, unsigned nNumUniforms);
    
    GLuint m_program;
    GLint* m_pUniforms;
};



#endif
