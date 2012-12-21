#include <assert.h>

#include <iostream>

#include <GL/glew.h>

#include "SDL.h"

#include "shader.h"
#include "sphere.h"


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
    Zillion::Shader vertexShader(szVertexFile, GL_VERTEX_SHADER);
    if(!vertexShader)
        return false;
    
    Zillion::Shader fragmentShader(szFragmentFile, GL_FRAGMENT_SHADER);
    if(!fragmentShader)
        return false;
    
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader.id());
    assert(glGetError() == 0);
    
    glAttachShader(shaderProgram, fragmentShader.id());
    assert(glGetError() == 0);
    
    if(!linkProgram(shaderProgram, "sphere"))
        return false;
    
    glUseProgram(shaderProgram);
    
    {
        // Create VBO for sphere
        Zillion::Sphere sphere(0.5, 40, 20);
        
        GLint posAttrib = glGetAttribLocation(shaderProgram, "position");
        assert(glGetError() == 0);
        
        glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(posAttrib);
        
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


int
main(int argc, char* argv[])
{
	SDL_Init(SDL_INIT_TIMER | SDL_INIT_VIDEO);

	SDL_Surface* pSurface = SDL_SetVideoMode(
								800, 600, 32,
								SDL_HWSURFACE | SDL_DOUBLEBUF | SDL_OPENGL);
	SDL_WM_SetCaption("Zillion", 0);

	glewExperimental = GL_TRUE;
	glewInit();

    bool status = run();

	SDL_Quit();

	return status ? 0 : 1;
}
