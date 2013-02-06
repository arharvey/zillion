cmake_minimum_required (VERSION 2.8)
project (zillion)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

set (CMAKE_VERBOSE_MAKEFILE TRUE)
find_package(CUDA REQUIRED)
set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch compute_11")

FILE (GLOB zillion_src src/*.cpp src/*.h src/*.cu)

cuda_add_executable (zillion ${zillion_src})

find_package (OpenGL REQUIRED)
include_directories (${OPENGL_INCLUDE_DIR})
target_link_libraries (zillion ${OPENGL_LIBRARIES})

find_package (GLEW REQUIRED)
include_directories (${GLEW_INCLUDE_DIR})
target_link_libraries (zillion ${GLEW_LIBRARY})

find_package (GLFW REQUIRED)
include_directories (${GLFW_INCLUDE_DIRS})
target_link_libraries (zillion ${GLFW_LIBRARIES})

# Required by GLFW on Linux
if(${CMAKE_SYSTEM_NAME} STREQUAL Linux)
    find_library(XRANDR_LIBRARY DOC "Absolute path to Xrandr library."
    NAMES Xrandr
    HINTS
    PATHS
    /usr/local/lib
    /usr/lib
    )
    target_link_libraries (zillion ${XRANDR_LIBRARY})
endif()

# Do not need IlmThread
set (ILMBASE_CUSTOM TRUE)
set (ILMBASE_CUSTOM_LIBRARIES Half Iex Imath)

find_package (IlmBase REQUIRED)
include_directories (${ILMBASE_INCLUDE_DIR})
target_link_libraries (zillion ${ILMBASE_LIBRARIES})

