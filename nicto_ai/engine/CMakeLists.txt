cmake_minimum_required(VERSION 3.18)
project(nicto_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11
find_package(pybind11 REQUIRED)

# Find PyTorch (optional, for tensor interop)
find_package(Torch QUIET)

# Source files
set(SOURCES
    cpp/bindings.cpp
)

# Create the module
pybind11_add_module(nicto_engine ${SOURCES})

target_include_directories(nicto_engine PRIVATE ${CMAKE_SOURCE_DIR}/cpp)

if(Torch_FOUND)
    target_link_libraries(nicto_engine PRIVATE ${TORCH_LIBRARIES})
    target_compile_definitions(nicto_engine PRIVATE USE_TORCH)
endif()

# Optimization flags
if(MSVC)
    target_compile_options(nicto_engine PRIVATE /O2 /std:c++17)
else()
    target_compile_options(nicto_engine PRIVATE -O3 -std=c++17 -fPIC)
endif()

# Install
install(TARGETS nicto_engine DESTINATION nicto_ai/engine)
