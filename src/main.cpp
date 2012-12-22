#include <assert.h>

#include <iostream>

#include <OpenEXR/ImathFrustum.h>

#include <GL/glew.h>
#include "SDL.h"

#include <cuda_runtime.h>

#include "constants.h"
#include "shader.h"
#include "sphere.h"

void
initGrid(float* Pd, const float* P0d, unsigned nPts);

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


void
gridPts(GLfloat* pPts, unsigned nDim, GLfloat size)
{
    const GLfloat _DIM = 1.0/GLfloat(nDim);
    
    GLfloat* pt = pPts;
    for(unsigned k = 0; k < nDim; k++)
    {
        for(unsigned j = 0; j < nDim; j++)
        {
            for(unsigned i = 0; i < nDim; i++)
            {
                pt[0] = (((GLfloat(i)+.5)*_DIM) - 0.5) * size;
                pt[1] = (((GLfloat(j)+.5)*_DIM) - 0.5) * size;
                pt[2] = (((GLfloat(k)+.5)*_DIM) - 0.5) * size;

                pt += 3;
            }
        }
    }
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
        Sphere sphere(0.5, 8, 4);
        
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
        GLint uniModelViewXf = glGetUniformLocation(shaderProgram, "modelViewXf");
        GLint uniNormalXf = glGetUniformLocation(shaderProgram, "normalXf");
        
        GLint uniLightDir = glGetUniformLocation(shaderProgram, "lightDirWorld");
        
        glUniformMatrix4fv(uniProjXf, 1, GL_FALSE, &(projXf.x[0][0]));
        
        // Instanced positions
        
        const unsigned nDimNum = 20;
        const GLfloat size = 1.0;
        
        const unsigned nPts = nDimNum*nDimNum*nDimNum;
        const unsigned sizeP = nPts*3*sizeof(float);
        
        GLfloat* P = new GLfloat[nPts*3];
        gridPts(P, nDimNum, size);
        
        // Instanced positions (using CUDA)
        
        float *Pd[2];
        for(unsigned n = 0; n < 2; n++)
            cudaMalloc((void**)&Pd[n], sizeP);
            
        cudaMemcpy(Pd[0], P, sizeP, cudaMemcpyHostToDevice);
        initGrid(Pd[1], Pd[0], nPts);
        cudaMemcpy(P, Pd[1], sizeP, cudaMemcpyDeviceToHost);
        
        
        glBindBuffer(GL_ARRAY_BUFFER, sphere.buffer(Sphere::kPt));
        
        GLint ptAttrib = glGetAttribLocation(shaderProgram, "pt");
        
        glVertexAttribPointer(ptAttrib, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glVertexAttribDivisorARB(ptAttrib, 1);
        glEnableVertexAttribArray(ptAttrib);
        
        // View
        
        Imath::M44f viewXf;
        viewXf.makeIdentity();
        viewXf.rotate(
            Imath::V3f(toRadians(0.0), toRadians(0.0), toRadians(0.0)));
        viewXf *= Imath::M44f(Imath::M33f(), Imath::V3f(0.0, 0.0, 0.0));
        
        // Lights
        
        Imath::V3f lightDir(-1.0, -1.0, -1.0);
        lightDir *= viewXf;
        lightDir.normalize();
        
        glUniform3f(uniLightDir, lightDir.x, lightDir.y, lightDir.z);
        
        glEnable(GL_DEPTH_TEST);
        
        unsigned nFrameCount = 0;
        Uint64 nStartTick = SDL_GetTicks();
        
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

            GLfloat animRotY = (GLfloat(SDL_GetTicks()) / 1000.0) * 45.0;
        
            Imath::M44f modelXf;
            modelXf.makeIdentity();
            modelXf.scale(Imath::V3f(1.0, 1.0, 1.0));
            modelXf.rotate(Imath::V3f(0.0, toRadians(animRotY), toRadians(animRotY)));
            modelXf *= Imath::M44f(Imath::M33f(), Imath::V3f(0.0, 0.0, -2));

            Imath::M44f modelViewXf = modelXf * viewXf.inverse();

            const float (*x)[4] = modelViewXf.x;
            Imath::M33f normalXf(   x[0][0], x[0][1], x[0][2],
                                    x[1][0], x[1][1], x[1][2],
                                    x[2][0], x[2][1], x[2][2]);
            normalXf.invert();
            normalXf.transpose();

            glUniformMatrix4fv(uniModelViewXf, 1, GL_FALSE, &(modelViewXf.x[0][0]));
            glUniformMatrix3fv(uniNormalXf, 1, GL_FALSE, &(normalXf.x[0][0]));
            
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            sphere.drawInstances(P, nPts);

            SDL_GL_SwapBuffers();
            
            nFrameCount++;
        }
        
        if(nFrameCount > 0)
        {
            Uint64 nTicks = SDL_GetTicks() - nStartTick;
            float fps = float(nFrameCount) / (float(nTicks) / 1000.0);
            
            std::cout << "Speed: " << fps << " fps" << std::endl;
        }
        
        for(unsigned n = 0; n < 2; n++)
            cudaFree(Pd[n]);
    }
    
    return true;
}

} // END NAMESPACE ZILLION


// ---------------------------------------------------------------------------

int
main(int argc, char* argv[])
{
	SDL_Init(SDL_INIT_TIMER | SDL_INIT_VIDEO);

    // Disable vertical sync. Warning: SDL_GL_SWAP_CONTROL is deprecated
    SDL_GL_SetAttribute( SDL_GL_SWAP_CONTROL, 0 );
    
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
