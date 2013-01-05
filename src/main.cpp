#include <assert.h>

#include <iostream>

#include <OpenEXR/ImathFrustum.h>

#include <GL/glew.h>
#include "SDL.h"

#include <cuda_runtime.h>

#include "constants.h"
#include "shader.h"
#include "program.h"
#include "planePrimitive.h"
#include "spherePrimitive.h"
#include "camera.h"
#include "chequerTexture.h"

#include "simulationCUDA.h"

const unsigned LEFT_BTN = 1;
const unsigned MIDDLE_BTN = 2;
const unsigned RIGHT_BTN = 4;


namespace Zillion {

int g_cudaDevice = 0;
   

void
gridPts(float* pPts, float nDim, GLfloat cellSize, const Imath::V3f offset)
{
    const float _DIM = 1.0/float(nDim);
    
    float* pt = pPts;
    for(unsigned k = 0; k < nDim; k++)
    {
        for(unsigned j = 0; j < nDim; j++)
        {
            for(unsigned i = 0; i < nDim; i++)
            {
                pt[0] = (((float(i)+.5)*_DIM) - 0.5) * cellSize + offset.x;
                pt[1] = (((float(j)+.5)*_DIM) - 0.5) * cellSize + offset.y;
                pt[2] = (((float(k)+.5)*_DIM) - 0.5) * cellSize + offset.z;

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


// ---------------------------------------------------------------------------

class ParticleProgram : public Program
{
public:
    enum Uniform
    {
        kScale = 0,
        kProjectionXf,
        kModelViewXf,
        kModelViewNormalXf,
        kLightDirWorld,
        kSurfaceColor0,
        kSurfaceColor1,
        kNumUniforms
    };
    
    static const char* szUniforms[kNumUniforms];
    
    ParticleProgram(const std::string& strShaderDir):
    Program(kNumUniforms),
    m_vertex((strShaderDir+"particleVertex.glsl").c_str(), GL_VERTEX_SHADER),
    m_fragment((strShaderDir+"particleFragment.glsl").c_str(), GL_FRAGMENT_SHADER)
    {
        if(!m_vertex || !m_fragment)
            return;
        
        GLuint program = glCreateProgram();
        glAttachShader(program, m_vertex.id());
        glAttachShader(program, m_fragment.id());
        
        glBindAttribLocation(program, 0, "position");
        glBindAttribLocation(program, 1, "normal");
        glBindAttribLocation(program, 2, "uv");
        
        glBindAttribLocation(program, 3, "center");

        if(!link(program, "particle"))
            return;

        // Program successfully linked
        m_program = program;
        
        initUniformLocations(szUniforms, kNumUniforms);
    }
    
    
protected:
    Shader m_vertex;
    Shader m_fragment;
};


const char* ParticleProgram::szUniforms[] = {
    "scale",
    "projectionXf",
    "modelViewXf",
    "normalXf",
    "lightDirWorld",
    "surfaceColor0",
    "surfaceColor1" 
};


// ---------------------------------------------------------------------------

class GroundProgram : public Program
{
public:
    enum Uniform
    {
        kProjectionXf = 0,
        kViewXf,
        kViewNormalXf,
        kLightDirWorld,
        kTex,
        kNumUniforms
    };
    
    static const char* szUniforms[kNumUniforms];
    
    GroundProgram(const std::string& strShaderDir):
    Program(kNumUniforms),
    m_vertex((strShaderDir+"groundVertex.glsl").c_str(), GL_VERTEX_SHADER),
    m_fragment((strShaderDir+"groundFragment.glsl").c_str(), GL_FRAGMENT_SHADER)
    {
        if(!m_vertex || !m_fragment)
            return;
        
        GLuint program = glCreateProgram();
        glAttachShader(program, m_vertex.id());
        glAttachShader(program, m_fragment.id());
        
        glBindAttribLocation(program, 0, "position");
        glBindAttribLocation(program, 1, "normal");
        glBindAttribLocation(program, 2, "uv");

        if(!link(program, "ground"))
            return;

        // Program successfully linked
        m_program = program;
        
        initUniformLocations(szUniforms, kNumUniforms);
    }
    
    
protected:
    Shader m_vertex;
    Shader m_fragment;
};


const char* GroundProgram::szUniforms[] = {
    "projectionXf",
    "viewXf",
    "normalXf",
    "lightDirWorld",
    "tex"
};


// ---------------------------------------------------------------------------

class DomeProgram : public Program
{
public:
    enum Uniform
    {
        kProjectionXf = 0,
        kViewXf,
        kColor0,
        kColor1,
        kColor2,
        kNumUniforms
    };
    
    static const char* szUniforms[kNumUniforms];
    
    DomeProgram(const std::string& strShaderDir):
    Program(kNumUniforms),
    m_vertex((strShaderDir+"domeVertex.glsl").c_str(), GL_VERTEX_SHADER),
    m_fragment((strShaderDir+"domeFragment.glsl").c_str(), GL_FRAGMENT_SHADER)
    {
        if(!m_vertex || !m_fragment)
            return;
        
        GLuint program = glCreateProgram();
        glAttachShader(program, m_vertex.id());
        glAttachShader(program, m_fragment.id());
        
        glBindAttribLocation(program, 0, "position");
        glBindAttribLocation(program, 1, "normal");
        glBindAttribLocation(program, 2, "uv");

        if(!link(program, "dome"))
            return;

        // Program successfully linked
        m_program = program;
        
        initUniformLocations(szUniforms, kNumUniforms);
    }
    
    
protected:
    Shader m_vertex;
    Shader m_fragment;
};


const char* DomeProgram::szUniforms[] = {
    "projectionXf",
    "viewXf",
    "color0",
    "color1",
    "color2"
};



// ---------------------------------------------------------------------------

bool
run()
{
    ChequerTexture chequerTex(2);
    
    // Particle program
    ParticleProgram particleProg("/Users/andrewharvey/dev/zillion/src/");
    if(!particleProg)
        return false;
    
    GroundProgram groundProg("/Users/andrewharvey/dev/zillion/src/");
    if(!groundProg)
        return false;
    
    DomeProgram domeProg("/Users/andrewharvey/dev/zillion/src/");
    if(!domeProg)
        return false;
    
    {
        const unsigned nDimNum = GRID_DIM;
        const float gridSize = GRID_SIZE;
        const float cellSize = gridSize / float(nDimNum);
        const float particleRadius = cellSize * 0.5 *
                                     PARTICLE_SIZE_RELATIVE_TO_GRID_CELL;
        
        // Create VBO for sphere
        PlanePrimitive ground( Imath::Plane3f(Imath::V3f(0.0, 1.0, 0.0), 0.0));
        SpherePrimitive dome(FAR, 100, 50);
        SpherePrimitive sphere(1.0, 8, 4);
       
        // Initialize simulation
        const unsigned nParticles = nDimNum*nDimNum*nDimNum;
        std::cout << "Instancing " << nParticles << " objects" << std::endl;
         
        float* Pinit = new float[nParticles*3];
        gridPts(Pinit, nDimNum, 1.0, Imath::V3f(0.0, 0.75, 0.0));
        
        SimulationCUDA sim(Pinit, nParticles, particleRadius);
        
        delete [] Pinit;
         
        // Centers
        GLint centerAttrib = glGetAttribLocation(particleProg, "center");
      
        sphere.bind();
        sim.P(1).bind();
        
        glEnableVertexAttribArray(centerAttrib);
        glVertexAttribPointer(centerAttrib, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glVertexAttribDivisorARB(centerAttrib, 1);
        sphere.unbind();
        
        // Projection
        
        Imath::Frustumf frustum(NEAR, FAR, FOV, 0.0,
                                float(WIDTH)/float(HEIGHT));
        
        Imath::M44f projXf = frustum.projectionMatrix();
        
        // Initialise shader programs
        
        particleProg.use();
        particleProg.set(ParticleProgram::kProjectionXf, projXf);
        particleProg.set(ParticleProgram::kScale, particleRadius);
        
        groundProg.use();
        groundProg.set(GroundProgram::kProjectionXf, projXf);
        groundProg.set(GroundProgram::kTex, 0);
        
        domeProg.use();
        domeProg.set(DomeProgram::kProjectionXf, projXf);
        domeProg.set(DomeProgram::kColor0, Imath::V3f(0, 0, 0));
        domeProg.set(DomeProgram::kColor1, Imath::V3f(0.5, 0.6, 1.0));
        domeProg.set(DomeProgram::kColor2, Imath::V3f(0, 0, 0));
        
        glCullFace(GL_BACK);
        
        unsigned nFrameCount = 0;
        Uint64 nStartTick = SDL_GetTicks();
        
        TumbleCamera camera;
        camera.setCenter(Imath::V3f(0.0, 0.5, 0.0));
        camera.setDistance(3.0f);
        camera.setAltitude(toRadians(0));
        camera.setAzimuth(toRadians(0));
        
        unsigned mouseButton = 0;
        
        SDL_Event windowEvent;
        while(true)
        {
            if(SDL_PollEvent(&windowEvent))
            {
                if(windowEvent.type == SDL_QUIT)
                    break;

                if(windowEvent.type == SDL_KEYUP)
                {
                    if(windowEvent.key.keysym.sym == SDLK_ESCAPE)
                        break;
                }
                
                if(windowEvent.type == SDL_MOUSEBUTTONDOWN)
                {
                    switch(windowEvent.button.button)
                    {
                        case SDL_BUTTON_LEFT:
                            mouseButton |= LEFT_BTN;
                            break;
                            
                        case SDL_BUTTON_MIDDLE:
                            mouseButton |= MIDDLE_BTN;
                            break;
                            
                        case SDL_BUTTON_RIGHT:
                            mouseButton |= RIGHT_BTN;
                            break;
                    };
                }
                
                if(windowEvent.type == SDL_MOUSEBUTTONUP)
                {
                    switch(windowEvent.button.button)
                    {
                        case SDL_BUTTON_LEFT:
                            mouseButton &= ~LEFT_BTN;
                            break;
                            
                        case SDL_BUTTON_MIDDLE:
                            mouseButton &= ~MIDDLE_BTN;
                            break;
                            
                        case SDL_BUTTON_RIGHT:
                            mouseButton &= ~RIGHT_BTN;
                            break;
                            
                    };
                }
                
                if(windowEvent.type == SDL_MOUSEMOTION)
                {
                    
                    if(windowEvent.motion.state == SDL_BUTTON(SDL_BUTTON_LEFT))
                    {
                        
                        float altitudeDelta = float(windowEvent.motion.yrel) *
                                                toRadians(10) / 50.0;
                        
                        float azimuthDelta = float(-windowEvent.motion.xrel) *
                                                toRadians(10) / 50.0;
                        
                        camera.setAltitude(camera.altitude() + altitudeDelta);
                        camera.setAzimuth(camera.azimuth() + azimuthDelta);
                    }
                    else
                    if(windowEvent.motion.state == SDL_BUTTON(SDL_BUTTON_RIGHT))
                    {
                        float delta = float(windowEvent.motion.xrel)/100.0 * 0.25f;
                        
                        if(delta < 0.0)
                            delta = 1.0-delta;
                        else
                            delta = 1.0/(1.0+delta);
                        
                        camera.scaleDistance(delta);
                    }
                            
                }
            }
            
            // View
            
            Imath::M44f viewXf = camera.viewTransform();
            
            // View Normal transform
            
            const float (*x)[4];
            x = viewXf.x;
            Imath::M33f viewNormalXf(  x[0][0], x[0][1], x[0][2],
                                       x[1][0], x[1][1], x[1][2],
                                       x[2][0], x[2][1], x[2][2]);
            viewNormalXf.invert();
            viewNormalXf.transpose();
            
            // Model
            
            Imath::M44f modelXf;
            modelXf.makeIdentity();
            
            // ModelView
            
            Imath::M44f modelViewXf = modelXf * viewXf;
            
            // ModelView Normal transform
            
            x = modelViewXf.x;
            Imath::M33f modelViewNormalXf(  x[0][0], x[0][1], x[0][2],
                                            x[1][0], x[1][1], x[1][2],
                                            x[2][0], x[2][1], x[2][2]);
            modelViewNormalXf.invert();
            modelViewNormalXf.transpose();
            
            // Lights
        
            Imath::V3f lightDir(-1.0, -1.0, -1.0);
            lightDir *= viewNormalXf;
            lightDir.normalize();
            
            // Start new frame
            
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            
            // Sky dome

            glDisable(GL_DEPTH_TEST);
            glDisable(GL_CULL_FACE); // Make sure we can see inside the sphere
            
            domeProg.use();

            Imath::M44f domeViewXf = viewXf;
            domeViewXf[3][0] = 0.0;
            domeViewXf[3][1] = 0.0; 
            domeViewXf[3][2] = 0.0;

            domeProg.set(DomeProgram::kViewXf, domeViewXf);

            dome.bind();
            dome.draw();
            dome.unbind();
            
            
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_CULL_FACE);
            
            // Ground plane
            
            groundProg.use();
            groundProg.set(GroundProgram::kViewXf, viewXf);
            groundProg.set(GroundProgram::kViewNormalXf, viewNormalXf);
            groundProg.set(GroundProgram::kLightDirWorld, lightDir);
            
            chequerTex.bind(0);
            
            ground.update(viewXf * projXf, NEAR, FAR);
            ground.bind();
            ground.draw();
            ground.unbind();
              
            // Particle system
            
            particleProg.use();
            particleProg.set(ParticleProgram::kModelViewXf, modelViewXf);
            particleProg.set(ParticleProgram::kModelViewNormalXf, modelViewNormalXf);
            particleProg.set(ParticleProgram::kLightDirWorld, lightDir);
            
            sphere.bind();
            sphere.drawInstances(nParticles);
            sphere.unbind();
            
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_CULL_FACE);
            
            // Make visible
            
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
