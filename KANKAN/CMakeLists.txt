
cmake_minimum_required(VERSION 3.18)

project(KANKAN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_kankan src/bindings.cpp)

target_include_directories(_kankan PRIVATE include)

install(TARGETS _kankan DESTINATION kankan)
