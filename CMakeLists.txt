cmake_minimum_required (VERSION 2.8)
project (zillion)

FILE (GLOB zillion_src src/*.cpp)

add_executable (zillion ${zillion_src})