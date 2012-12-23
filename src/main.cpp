#include <assert.h>

#include <iostream>

#include <OpenEXR/ImathFrustum.h>

#include <GL/glew.h>
#include "SDL.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "constants.h"
#include "shader.h"
#include "sphere.h"
#include "sharedBuffer.h"

void
initGrid(float* Pd, const float* P0d, unsigned nPts);

namespace Zillion {

int g_cudaDevice = 0;
    
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


void
printCUDADeviceProperties(const cudaDeviceProp& prop)
{
    std::cout << "Name: " << prop.name << std::endl;
        
    std::cout << "Compute capability: "
            << prop.major << "." << prop.minor << std::endl;

    std::cout << "Clock rate: " 
            << prop.clockRate/1000 << " MHz" << std::endl;

    std::cout << "Integrated: "
            << yesNo(prop.integrated) << std::endl;

    std::cout << "Global memory: " 
            << inMB(prop.totalGlobalMem) << " MB" << std::endl;

    std::cout << "Constant memory: "
            << prop.totalConstMem << " bytes" << std::endl;

    std::cout << "Shared memory per block: "
            << prop.sharedMemPerBlock << " bytes" << std::endl;

    std::cout << "Registers per block: "
            << prop.regsPerBlock << std::endl;

    std::cout << "Max threads per block: "
            << prop.maxThreadsPerBlock << std::endl;

    std::cout << "Max thread dimensions: "
            << prop.maxThreadsDim[0] << ", "
            << prop.maxThreadsDim[1] << ", "
            << prop.maxThreadsDim[2] << std::endl;

    std::cout << "Max grid size: "
            << prop.maxGridSize[0] << ", "
            << prop.maxGridSize[1] << ", "
            << prop.maxGridSize[2] << std::endl;

    std::cout << "Max threads per multi-processor: "
            << prop.maxThreadsPerMultiProcessor << std::endl;

    std::cout << "Async engine count: "
            << prop.asyncEngineCount << std::endl;


    std::cout << "Can map host memory: "
            << yesNo(prop.canMapHostMemory) << std::endl;

    std::cout << "Unified addressing: "
            << yesNo(prop.unifiedAddressing) << std::endl;

    std::cout << "Concurrent kernels: "
            << yesNo(prop.concurrentKernels) << std::endl;
}


bool
initCUDA()
{
    cudaError_t status;

    cudaDeviceProp cudaProp;
    
    // Discover CUDA devices
    
    int nDeviceCount;
    cudaGetDeviceCount(&nDeviceCount);
    
    std::cout << "Found " << nDeviceCount << " CUDA devices" << std::endl;
    std::cout << std::endl;
    
    for(int nDevice = 0; nDevice < nDeviceCount; nDevice++)
    {
        std::cout << "DEVICE " << nDevice << std::endl;
        std::cout << "========" << std::endl;
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, nDevice);
        
        printCUDADeviceProperties(prop);
        
        std::cout << std::endl;
    }
    
    // Request a suitable CUDA device
   
    memset(&cudaProp, 0, sizeof(cudaDeviceProp));
    cudaProp.major = 1;
    cudaProp.minor = 1;
    
    status = cudaChooseDevice(&g_cudaDevice, &cudaProp);
    if(status != cudaSuccess)
    {
        std::cerr << "Unable to find CUDA device" << std::endl;
        return false;
    }
    
    cudaGLSetGLDevice(g_cudaDevice);
    
    std::cout << "Using CUDA device " << g_cudaDevice << std::endl;
    
    std::cout << std::endl;
    
    return true;
}




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
    glAttachShader(shaderProgram, fragmentShader.id());
    
    glBindAttribLocation(shaderProgram, 0, "position");
    glBindAttribLocation(shaderProgram, 1, "normal");
    glBindAttribLocation(shaderProgram, 2, "pt");
    
    if(!linkProgram(shaderProgram, "sphere"))
        return false;
    
    glUseProgram(shaderProgram);
    
    {
        GLint uniProjXf = glGetUniformLocation(shaderProgram, "projectionXf");
        GLint uniModelViewXf = glGetUniformLocation(shaderProgram, "modelViewXf");
        GLint uniNormalXf = glGetUniformLocation(shaderProgram, "normalXf");
        GLint uniLightDir = glGetUniformLocation(shaderProgram, "lightDirWorld");
        
        // Create VBO for sphere
        Sphere sphere(0.5, 8, 4);
        
        // Instanced positions
        
        const unsigned nDimNum = 20;
        const GLfloat size = 1.0;
        
        const unsigned nPts = nDimNum*nDimNum*nDimNum;
        std::cout << "Instancing " << nPts << " objects" << std::endl;
        
        const unsigned sizeP = nPts*3*sizeof(float);
        
        GLfloat* Pinit = new GLfloat[nPts*3];
        gridPts(Pinit, nDimNum, size);
        
        // Instanced positions (using CUDA)
        
        SharedBuffer P0(GL_ARRAY_BUFFER, nPts*3, GL_DYNAMIC_DRAW);
        SharedBuffer P1(GL_ARRAY_BUFFER, nPts*3, GL_DYNAMIC_DRAW);
        
        P0.map();
        P1.map();
        
        cudaMemcpy(P0, Pinit, sizeP, cudaMemcpyHostToDevice);
        initGrid(P1, P0, nPts);
        
        P0.unmap();
        P1.unmap();
        
        // Centers
        GLint centerAttrib = glGetAttribLocation(shaderProgram, "center");
        
        sphere.bind();
        P1.bind();
        glEnableVertexAttribArray(centerAttrib);
        glVertexAttribPointer(centerAttrib, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glVertexAttribDivisorARB(centerAttrib, 1);
        
        // Projection
        
        Imath::Frustumf frustum(0.1, 1000.0, FOV, 0.0,
                                float(WIDTH)/float(HEIGHT));
        
        Imath::M44f projXf = frustum.projectionMatrix();
        
        glUniformMatrix4fv(uniProjXf, 1, GL_FALSE, &(projXf.x[0][0]));
        
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

            
            // View
        
            Imath::M44f viewXf;
            viewXf.makeIdentity();
            viewXf.rotate(
                Imath::V3f(toRadians(0.0), toRadians(0.0), toRadians(0.0)));
            viewXf *= Imath::M44f(Imath::M33f(), Imath::V3f(0.0, 0.0, 0.0));
            
            // Model
            
            GLfloat animRotY = (GLfloat(SDL_GetTicks()) / 1000.0) * 45.0;
        
            Imath::M44f modelXf;
            modelXf.makeIdentity();
            modelXf.scale(Imath::V3f(1.0, 1.0, 1.0));
            modelXf.rotate(Imath::V3f(0.0, toRadians(animRotY), toRadians(animRotY)));
            modelXf *= Imath::M44f(Imath::M33f(), Imath::V3f(0.0, 0.0, -2));

            // ModelView
            
            Imath::M44f modelViewXf = modelXf * viewXf.inverse();

            glUniformMatrix4fv(uniModelViewXf, 1, GL_FALSE, &(modelViewXf.x[0][0]));
            
            // Normal transform
            
            const float (*x)[4] = modelViewXf.x;
            Imath::M33f normalXf(   x[0][0], x[0][1], x[0][2],
                                    x[1][0], x[1][1], x[1][2],
                                    x[2][0], x[2][1], x[2][2]);
            normalXf.invert();
            normalXf.transpose();
            
            glUniformMatrix3fv(uniNormalXf, 1, GL_FALSE, &(normalXf.x[0][0]));
            
            // Lights
        
            Imath::V3f lightDir(-1.0, -1.0, -1.0);
            lightDir *= viewXf;
            lightDir.normalize();

            glUniform3f(uniLightDir, lightDir.x, lightDir.y, lightDir.z);
            
            // Redraw
            
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            sphere.drawInstances(nPts);

            SDL_GL_SwapBuffers();
            
            nFrameCount++;
        }
        
        if(nFrameCount > 0)
        {
            Uint64 nTicks = SDL_GetTicks() - nStartTick;
            float fps = float(nFrameCount) / (float(nTicks) / 1000.0);
            
            std::cout << "Speed: " << fps << " fps" << std::endl;
        }
    }
    
    return true;
}

} // END NAMESPACE ZILLION


// ---------------------------------------------------------------------------


int
main(int argc, char* argv[])
{
    if(!Zillion::initCUDA())
        return 1;
    
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
