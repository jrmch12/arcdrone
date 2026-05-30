cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_FLAGS -std=c++17)
project(_jax_gsplat LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

pybind11_add_module(
    _jax_gsplat
    ${CMAKE_CURRENT_LIST_DIR}/lib/ffi.cu
    ${CMAKE_CURRENT_LIST_DIR}/lib/ops.cu
    ${CMAKE_CURRENT_LIST_DIR}/lib/kernels/forward.cu
)

if(DEFINED SKBUILD)
    set(JAX_GSPLAT_INSTALL_DEST ${SKBUILD_PLATLIB_DIR})
else()
    set(JAX_GSPLAT_INSTALL_DEST ${CMAKE_CURRENT_LIST_DIR})
endif()

install(TARGETS _jax_gsplat LIBRARY DESTINATION ${JAX_GSPLAT_INSTALL_DEST})
