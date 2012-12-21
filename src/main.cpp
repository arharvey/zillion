#include <assert.h>

#include <iostream>

#include <OpenEXR/ImathFrustum.h>

#include <GL/glew.h>
#include "SDL.h"

#include "constants.h"
#include "shader.h"
#include "sphere.h"

namespace Zillion {
  
bool
linkProgram(GLuint program, const char* szDesc)
{
    glLinkProgram(program);
    
    GLint status = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if(status != GL_TRUE)
    {
        char buffer[1024];
        glGetProgramInfoLog(program, 1024, NULL, buffer);
        
        std::cerr << "Error linking program '" << szDesc << "':" << std::endl;
        std::cerr << buffer << std::endl;
        return false;
    }
    
    return true;
}


// ---------------------------------------------------------------------------

bool
run()
{
    const char* szVertexFile =
        "/Users/andrewharvey/dev/zillion/src/vertex.glsl";
    const char* szFragmentFile =
        "/Users/andrewharvey/dev/zillion/src/fragment.glsl";
    
    // Compile OpenGL shaders
    Shader vertexShader(szVertexFile, GL_VERTEX_SHADER);
    if(!vertexShader)
        return false;
    
    Shader fragmentShader(szFragmentFile, GL_FRAGMENT_SHADER);
    if(!fragmentShader)
        return false;
    
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader.id());
    CHECK_GL();
    
    glAttachShader(shaderProgram, fragmentShader.id());
    CHECK_GL();
    
    if(!linkProgram(shaderProgram, "sphere"))
        return false;
    
    glUseProgram(shaderProgram);
    
    {
        // Create VBO for sphere
        Sphere sphere(0.5, 40, 20);
        
        GLint posAttrib = glGetAttribLocation(shaderProgram, "position");
        GLint normAttrib = glGetAttribLocation(shaderProgram, "normal");
       
        glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE,
                                6*sizeof(GLfloat), 0);
        glVertexAttribPointer(normAttrib, 3, GL_FLOAT, GL_FALSE,
                                6*sizeof(GLfloat), (void*)(3*sizeof(GLfloat)) );
        
        glEnableVertexAttribArray(posAttrib);
        glEnableVertexAttribArray(normAttrib);
        
        // Set up camera
        Imath::Frustumf frustum(0.1, 1000.0, FOV, 0.0,
                                float(WIDTH)/float(HEIGHT));
        
        Imath::M44f projXf = frustum.projectionMatrix();
        
        GLint uniProjXf = glGetUniformLocation(shaderProgram, "projectionXf");
        glUniformMatrix4fv(uniProjXf, 1, GL_FALSE, &(projXf.x[0][0]));
        
        
        Imath::M44f modelXf;
        modelXf.makeIdentity();
        modelXf.scale(Imath::V3f(0.1, 0.1, 0.1));
        modelXf.rotate(Imath::V3f(0.0, toRadians(00), 0.0));
        modelXf *= Imath::M44f(Imath::M33f(), Imath::V3f(0.0, 0.0, -1.0));
        
        Imath::M44f viewXf;
        viewXf.makeIdentity();
        viewXf.rotate(
            Imath::V3f(toRadians(0.0), toRadians(0.0), toRadians(0.0)));
        viewXf *= Imath::M44f(Imath::M33f(), Imath::V3f(0.0, 0.0, 0.0));
        
        
        Imath::M44f modelViewXf = modelXf * viewXf.inverse();
        
        const float (*x)[4] = modelViewXf.x;
        Imath::M33f normalXf(   x[0][0], x[0][1], x[0][2],
                                x[1][0], x[1][1], x[1][2],
                                x[2][0], x[2][1], x[2][2]);
        normalXf.invert();
        normalXf.transpose();
        
        GLint uniModelViewXf = glGetUniformLocation(shaderProgram, "modelViewXf");
        glUniformMatrix4fv(uniModelViewXf, 1, GL_FALSE, &(modelViewXf.x[0][0]));
        
        GLint uniNormalXf = glGetUniformLocation(shaderProgram, "normalXf");
        glUniformMatrix3fv(uniNormalXf, 1, GL_FALSE, &(normalXf.x[0][0]));
        
        
        // Lights
        
        Imath::V3f lightDir(-1.0, -1.0, -1.0);
        lightDir *= viewXf;
        lightDir.normalize();
        
        GLint uniLightDir = glGetUniformLocation(shaderProgram, "lightDirWorld");
        glUniform3f(uniLightDir, lightDir.x, lightDir.y, lightDir.z);
        
        glEnable(GL_DEPTH_TEST);
        
        SDL_Event windowEvent;
        while(true)
        {
            if(SDL_PollEvent(&windowEvent))
            {
                if(windowEvent.type == SDL_QUIT)
                    break;

                if(windowEvent.type == SDL_KEYUP &&
                    windowEvent.key.keysym.sym == SDLK_ESCAPE)
                    break;
            }

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            sphere.draw();

            SDL_GL_SwapBuffers();
        }
    }
    
    return true;
}

} // END NAMESPACE ZILLION


// ---------------------------------------------------------------------------

int
main(int argc, char* argv[])
{
	SDL_Init(SDL_INIT_TIMER | SDL_INIT_VIDEO);

	SDL_Surface* pSurface = SDL_SetVideoMode(
								Zillion::WIDTH, Zillion::HEIGHT, 32,
								SDL_HWSURFACE | SDL_DOUBLEBUF | SDL_OPENGL);
	SDL_WM_SetCaption("Zillion", 0);

	glewExperimental = GL_TRUE;
	glewInit();

    bool status = Zillion::run();

	SDL_Quit();

	return status ? 0 : 1;
}
