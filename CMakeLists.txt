
cmake_minimum_required(VERSION 3.18...4.0)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)
  # LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(pybind11 CONFIG REQUIRED)
# find_package(CUDAToolkit REQUIRED)

pybind11_add_module(_core
  src/main.cpp
  src/fft.cpp
)
target_link_libraries(_core PRIVATE pybind11::module)

target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

# The install directory is the output (wheel) directory
install(TARGETS _core DESTINATION cuda_poly)
