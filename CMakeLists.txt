cmake_minimum_required (VERSION 2.8)
project (zillion)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

FILE (GLOB zillion_src src/*.cpp src/*.h)

add_executable (zillion ${zillion_src})

find_package (OpenGL REQUIRED)
include_directories (${OPENGL_INCLUDE_DIR})
target_link_libraries (zillion ${OPENGL_LIBRARIES})

find_package (GLEW REQUIRED)
include_directories (${GLEW_INCLUDE_DIR})
target_link_libraries (zillion ${GLEW_LIBRARY})

find_package (SDL REQUIRED)
include_directories (${SDL_INCLUDE_DIR})
target_link_libraries (zillion ${SDL_LIBRARY})

# Do not need IlmThread
set (ILMBASE_CUSTOM TRUE)
set (ILMBASE_CUSTOM_LIBRARIES Half Iex Imath)

find_package (ILMBASE REQUIRED)
include_directories (${ILMBASE_INCLUDE_DIR})
target_link_libraries (zillion ${ILMBASE_LIBRARIES})

