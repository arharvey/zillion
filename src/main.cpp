#include <assert.h>

#include <iostream>

#include <OpenEXR/ImathFrustum.h>

#include <GL/glew.h>
#include <GL/glfw.h>

#include <cuda_runtime.h>

#include "constants.h"
#include "shader.h"
#include "program.h"
#include "planePrimitive.h"
#include "spherePrimitive.h"
#include "camera.h"
#include "chequerTexture.h"

#include "simulationCUDA.h"

#include "minMaxTest.h"

const unsigned LEFT_BTN = 1;
const unsigned MIDDLE_BTN = 2;
const unsigned RIGHT_BTN = 4;


namespace Zillion {

int g_cudaDevice = 0;


void
initPositions(float3* pPts, float nDim, GLfloat size, int nMaxParticles,
              const Imath::V3f offset, const float jitter)
{
    srand(19);
    
    const float _DIM = 1.0/float(nDim);
    
    int nParticles = 0;
    
    float3* pt = pPts;
    for(unsigned k = 0; k < nDim; k++)
    {
        for(unsigned j = 0; j < nDim; j++)
        {
            for(unsigned i = 0; i < nDim; i++)
            {
                pt->x = (((float(i)+.5+frand()*jitter)*_DIM) - 0.5) * size + offset.x;
                pt->y = (((float(j)+.5+frand()*jitter)*_DIM) - 0.5) * size + offset.y;
                pt->z = (((float(k)+.5+frand()*jitter)*_DIM) - 0.5) * size + offset.z;
                
                pt++;
                nParticles++;
                
                if(nMaxParticles > 0 && nParticles > nMaxParticles)
                    return;
            }
        }
    }
}


void
initColors(float3* pColor, int nParticles, const Imath::V3f& tint, float mix)
{
    srand(31);
    
    float3* C = pColor;
    for(int n = 0; n < nParticles; n++)
    {
        Imath::V3f CC(frand(), frand(), frand());
        CC *= 1.0f / std::max(CC.x, std::max(CC.y, CC.z));
        CC = mix*CC + (1.0f-mix)*tint;

        C->x = CC.x;
        C->y = CC.y;
        C->z = CC.z;

        C++;
    }
}


void
initVelocities(float3* pVel, const float3* pPts, unsigned N, float scale, const Imath::V3f& center)
{
    for(unsigned n = 0; n < N; n++)
    {
        const float3& P = pPts[n];
        Imath::V3f vel = Imath::V3f(P.x, P.y, P.z) - center;
        vel.normalize();
        vel *= scale;
        
        float3& V = pVel[n];
        V.x = vel.x;
        V.y = vel.y;
        V.z = vel.z;
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

    std::cout << "Multiprocessor count: "
            << prop.multiProcessorCount << std::endl;
    
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
        glBindAttribLocation(program, 4, "color");

        glBindFragDataLocation(program, 0, "outColor" );
        
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
    "lightDirWorld"
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

        glBindFragDataLocation(program, 0, "outColor" );
        
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

        glBindFragDataLocation(program, 0, "outColor" );
        
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


unsigned g_mouseButtons = 0;
int g_mouseX = 0, g_mouseY = 0, g_mousePrevX = 0, g_mousePrevY = 0;
int g_mouseRelativeX = 0, g_mouseRelativeY = 0;
bool g_bMouseMoved = false;

void GLFWCALL
onMouseButtonPressed(int button, int action)
{
    switch(button)
    {
        case GLFW_MOUSE_BUTTON_LEFT:
        {
            if(action == GLFW_PRESS)
                g_mouseButtons |= LEFT_BTN;
            else
                g_mouseButtons &= ~LEFT_BTN;
            
            break;
        }
            
        
        case GLFW_MOUSE_BUTTON_RIGHT:
        {
            if(action == GLFW_PRESS)
                g_mouseButtons |= RIGHT_BTN;
            else
                g_mouseButtons &= ~RIGHT_BTN;
            
            break;
        }
    }
}


void GLFWCALL 
onMouseMoved(int x, int y)
{
    g_mousePrevX = g_mouseX;
    g_mousePrevY = g_mouseY;
    
    g_mouseX = x;
    g_mouseY = y;
    
    g_mouseRelativeX = g_mouseX - g_mousePrevX;
    g_mouseRelativeY = g_mouseY - g_mousePrevY;
    
    g_bMouseMoved = true;
}


void
init()
{    
    glfwGetMousePos(&g_mousePrevX, &g_mousePrevY);
    g_mouseX = g_mousePrevX;
    g_mouseY = g_mousePrevY;

    glfwSetMousePosCallback(onMouseMoved);
    
    glfwSetMouseButtonCallback(onMouseButtonPressed);
}


bool
run()
{
    ChequerTexture chequerTex(2);
    
    const char* szShadersDir = "../src/";
    
    // Particle program
    ParticleProgram particleProg(szShadersDir);
    if(!particleProg)
        return false;
    
    GroundProgram groundProg(szShadersDir);
    if(!groundProg)
        return false;
    
    DomeProgram domeProg(szShadersDir);
    if(!domeProg)
        return false;
    
    {
        const unsigned nDimNum = GRID_DIM;
        const float gridSize = GRID_SIZE;
        const float cellSize = gridSize / float(nDimNum);
        const float particleRadius = cellSize * 0.5 *
                                     PARTICLE_SIZE_RELATIVE_TO_GRID_CELL;
        
        // Create VBO for sphere
        PlanePrimitive ground( Imath::Plane3f(Imath::V3f(0.0, 2.0, 0.0), 0.0));
        SpherePrimitive dome(FAR, 100, 50);
        SpherePrimitive sphere(1.0, 12, 6);
       
        // Initialize simulation
        const unsigned nParticles = nDimNum*nDimNum*nDimNum;
        
        std::cout << "Instancing " << nParticles << " objects" << std::endl;
         
        float3* Pinit = new float3[nParticles];
        
        const Imath::V3f offset(0.0, 1.25, 0.0);
        const float jitter = 0.5*(1.0-PARTICLE_SIZE_RELATIVE_TO_GRID_CELL);
        
        initPositions(Pinit, nDimNum, GRID_SIZE, 0, offset, jitter);
               
        float3* Vinit = new float3[nParticles];
        initVelocities(Vinit, Pinit, nParticles, 0.0, Imath::V3f(0.0, 0.25, 0.0));
        
        SimulationCUDA sim(g_cudaDevice, Pinit, Vinit, nParticles, particleRadius);
        
        delete [] Pinit;
        delete [] Vinit;
        
        // Initialise shader programs
        
        particleProg.use();
        particleProg.set(ParticleProgram::kScale, particleRadius);
        
        sphere.bind();
        GLuint colorBuffer;
        glGenBuffers(1, &colorBuffer);
    
        float3* Cinit = new float3[nParticles];
        initColors(Cinit, nParticles, Imath::V3f(0.9, 0.9, 1.0), 0.5);
        
        glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*nParticles*3,
                        Cinit, GL_STATIC_DRAW);

        delete [] Cinit;
        
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(4); // Color attribute
    
        GLint colorAttrib = glGetAttribLocation(particleProg, "color");
        glEnableVertexAttribArray(colorAttrib);
        glVertexAttribPointer(colorAttrib, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glVertexAttribDivisorARB(colorAttrib, 1);
        
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        sphere.unbind();
        
        groundProg.use();
        groundProg.set(GroundProgram::kTex, 0);
        
        domeProg.use();
        domeProg.set(DomeProgram::kColor0, Imath::V3f(0, 0, 0));
        domeProg.set(DomeProgram::kColor1, Imath::V3f(0.5, 0.6, 1.0));
        domeProg.set(DomeProgram::kColor2, Imath::V3f(0, 0, 0));
        
        glCullFace(GL_BACK);
        
        unsigned nFrameCount = 0;
        double startTime = glfwGetTime();
        double prevTime = startTime;
        
        TumbleCamera camera;
        camera.setCenter(Imath::V3f(0.0, 0.5, 0.0));
        camera.setDistance(3.0f);
        camera.setAltitude(toRadians(0));
        camera.setAzimuth(toRadians(0));
        
        int windowWidth = WIDTH, windowHeight = HEIGHT;
        while( glfwGetWindowParam(GLFW_OPENED) && nFrameCount < 1e8)
        {
            int currentWindowWidth = 0, currentWindowHeight = 0;
            glfwGetWindowSize(&currentWindowWidth, &currentWindowHeight);
            
            if(currentWindowWidth != windowWidth ||
                    currentWindowHeight != windowHeight)
            {
                windowWidth = currentWindowWidth;
                windowHeight = currentWindowHeight;

                glfwSetWindowSize(windowWidth, windowHeight);
                glViewport(0, 0, windowWidth, windowHeight);
            }
                
            if(glfwGetKey(GLFW_KEY_ESC) == GLFW_PRESS)
                    break;
             
       
            if(g_bMouseMoved)
            {
                if(g_mouseButtons & LEFT_BTN)
                {

                    float altitudeDelta = float(g_mouseRelativeY) *
                                            toRadians(10) / 50.0;

                    float azimuthDelta = float(-g_mouseRelativeX) *
                                            toRadians(10) / 50.0;

                    camera.setAltitude(camera.altitude() + altitudeDelta);
                    camera.setAzimuth(camera.azimuth() + azimuthDelta);
                }
                else
                if(g_mouseButtons & RIGHT_BTN)
                {
                    float delta = float(g_mouseRelativeX)/100.0 * 0.25f;

                    if(delta < 0.0)
                        delta = 1.0-delta;
                    else
                        delta = 1.0/(1.0+delta);

                    camera.scaleDistance(delta);
                }
                
                g_bMouseMoved = false;
            }
            
            
            // Update projection
        
            Imath::Frustumf frustum(NEAR, FAR, FOV, 0.0,
                                    float(windowWidth)/float(windowHeight));

            Imath::M44f projXf = frustum.projectionMatrix();
            
            particleProg.use();
            particleProg.set(ParticleProgram::kProjectionXf, projXf);

            groundProg.use();
            groundProg.set(GroundProgram::kProjectionXf, projXf);

            domeProg.use();
            domeProg.set(DomeProgram::kProjectionXf, projXf);
        
            
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
            
            

            Imath::M44f domeViewXf = viewXf;
            domeViewXf[3][0] = 0.0;
            domeViewXf[3][1] = 0.0; 
            domeViewXf[3][2] = 0.0;

            domeProg.use();
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
            
            sim.P().bind(); // Particle positions
            GLint centerAttrib = glGetAttribLocation(particleProg, "center");
            glEnableVertexAttribArray(centerAttrib);
            glVertexAttribPointer(centerAttrib, 3, GL_FLOAT, GL_FALSE, 0, 0);
            glVertexAttribDivisorARB(centerAttrib, 1);
            
            sphere.drawInstances(nParticles);
            sphere.unbind();
            
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_CULL_FACE);
            
            // Make visible
            
            glfwSwapBuffers();
            
            // Step the simulation forward
            const double currentTime = glfwGetTime();
            const double dt = std::min(currentTime - prevTime, 1.0/60.0);
            
            for(int n = 0; n < ITERATIONS; n++)
                sim.stepForward(dt/ITERATIONS);
              
            prevTime = currentTime;
            
            nFrameCount++;
        }
        
        if(nFrameCount > 0)
        {
            std::cout << "Frames rendered: " << nFrameCount << std::endl;
            
            double elapsed = glfwGetTime() - startTime;
            double fps = double(nFrameCount) / elapsed;
            
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
    
    //Zillion::minTest(7*1024*1024, 23, Zillion::g_cudaDevice);
    //return 0;
    
    
	glfwInit();

    glfwOpenWindowHint( GLFW_OPENGL_VERSION_MAJOR, 3 );
    glfwOpenWindowHint( GLFW_OPENGL_VERSION_MINOR, 2 );
    glfwOpenWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );
    
    glfwOpenWindow( Zillion::WIDTH, Zillion::HEIGHT, 0, 0, 0, 0, 32, 0, GLFW_WINDOW );
    glfwSetWindowTitle( "Zillion" );
    glfwSwapInterval(0);
    
	glewExperimental = GL_TRUE;
	glewInit();

    Zillion::init();
    bool status = Zillion::run();

	glfwTerminate();

	return status ? 0 : 1;
}
