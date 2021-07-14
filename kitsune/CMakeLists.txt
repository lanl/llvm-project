#
# Copyright (c) 2020 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under
# the LLVM license.
#
cmake_minimum_required(VERSION 3.13.4)

# RPATH settings on macOS do not affect INSTALL_NAME.
if (POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

# Include file check macros honor CMAKE_REQUIRED_LIBRARIES.
if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

# option() honors normal variables.
if (POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

set(KITSUNE_ENABLED TRUE
    CACHE
    BOOL
    "Enable Kitsune and Tapir features to the LLVM infrastructure."
)

set(KITSUNE_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(KITSUNE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(KITSUNE_BINARY_DIR  ${CMAKE_CURRENT_BINARY_DIR})


message(STATUS "Kitsune source directory: ${KITSUNE_SOURCE_DIR}")
message(STATUS "Kitsune include directory: ${KITSUNE_INCLUDE_DIR}")
message(STATUS "Kitsune binary directory: ${KITSUNE_BINARY_DIR}")

option(KITSUNE_INCLUDE_TESTS
  "Enable Kitsune tests." FALSE) # TODO: temp patch ${LLVM_INCLUDE_TESTS})
option(KITSUNE_BUILD_EXAMPLES
  "Build Kitsune example programs by default." ${KITSUNE_INCLUDE_TESTS})

# Add Kitsune-centric modules to cmake path.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

include(AddKitsune)

option(KITSUNE_ENABLE_WERROR
  "Fail kitsune builds if a warning is triggered." OFF)

# Runtime backend targets.

find_package(CUDAToolkit)
option(KITSUNE_ENABLE_CUDA
  "Enable Kitsune-centric CUDA support." ${CUDAToolkit_FOUND})
if (KITSUNE_ENABLE_CUDA)
  option(KITSUNE_ENABLE_CUDA_TARGET
    "Enable the CUDA runtime ABI target." ${CUDAToolkit_FOUND})
  if (KITSUNE_ENABLE_CUDA_TARGET)
    option(KITSUNE_CUDA_LIBRARY_DIR
    "Enable the CUDA runtime ABI target." ${CUDAToolkit_LIBRARY_DIR})
    set(KITSUNE_EXTRA_CUDA_LINK_LIBS ""
      CACHE
      STRING "Extra link libraries for CUDA ABI runtime target."
    )
  endif()

  option(KITSUNE_ENABLE_KITCUDA_TARGET
    "Enable Kitsune's CUDA runtime ABI target." ${CUDAToolkit_FOUND})
  if (KITSUNE_ENABLE_KITCUDA_TARGET)
    set(KITSUNE_EXTRA_KITCUDA_LINK_LIBS ""
      CACHE
      STRING "Extra link libraries for Kitsune's CUDA ABI runtime target."
    )
    add_subdirectory(lib/CudaRT)
  endif()
endif()

find_package(OpenCL)
# FIXME: Re-enable OpenCL default once code gen pieces are complete.
option(KITSUNE_ENABLE_OPENCL_TARGET
  "Enable SPIRV+OpenCL backend target." FALSE) # ${OpenCL_FOUND}
if (KITSUNE_ENABLE_OPENCL_TARGET)
	add_definitions(-DKITSUNE_ENABLE_OPENCL_TARGET)
  set(OPENCL_EXTRA_LINK_LIBS "-lkitopencl"
    CACHE
    STRING "Extra link libraries for OpenCL runtime target."
  )
  add_subdirectory(lib/OpenCL)
endif()

find_package(OpenCilkRTS)
option(KITSUNE_ENABLE_OPENCILKRTS_TARGET
  "Enable OpenCilk runtime backend target." ${OpenCilk_FOUND})
if (KITSUNE_ENABLE_OPENCILKRTS_TARGET)
  set(OPENCILKRTS_EXTRA_LINK_LIBS "-lopencilk-personality-cpp"
    CACHE
    STRING "Extra link libraries for the OpenCilk runtime target."
  )
endif()

find_package(Qthreads)
option(KITSUNE_ENABLE_QTHREADS_TARGET
  "Enable Qthreads runtime backend target." ${Qthreads_FOUND})
if (Qthreads_FOUND)
  message(STATUS "Qthreads runtime target enabled.")
  set(QTHREADS_EXTRA_LINK_LIBS "-ldl"
    CACHE
    STRING "Additional libraries needed by Qthreads backend runtime."
  )
endif()

find_package(Realm)
option(KITSUNE_ENABLE_REALM_TARGET
  "Enable Realm runtime backend target." ${Realm_FOUND})
if (Realm_FOUND)
  message(STATUS "Realm runtime target enabled.")
  set(REALM_EXTRA_LINK_LIBS  "-lpthread -ldl -lrt"
    CACHE
    STRING "Additional libraries needed by Realm runtime target.")
  add_subdirectory(lib/Realm)
endif()

if ("openmp" IN_LIST LLVM_ENABLE_PROJECTS)
  message(STATUS "adding support for libomp target code generation...")
  option(KITSUNE_ENABLE_OPENMP "Enable OpenMP/libomp runtime backend target." ON)
  set(OPENMP_EXTRA_LINK_LIBS "")
else()
  option(KITSUNE_ENABLE_OPENMP_TARGET "Enable OpenMP/libomp runtime backend target." OFF)
endif()

# Special kitsune modes for clang.
find_package(Kokkos)
option(KITSUNE_ENABLE_KOKKOS
  "Enable Kokkos support." ${Kokkos_FOUND})
if (Kokkos_FOUND)
  set(KOKKOS_EXTRA_LINK_LIBS "-ldl")
endif()

if (KITSUNE_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()

if (KITSUNE_BUILD_EXAMPLES)
  add_kitsune_subdirectory(examples)
endif()

message(STATUS "building kitsune configuration header...")
message(STATUS "   kitsune/config will live at ${LLVM_BINARY_DIR}/include/")
configure_file(
  ${KITSUNE_SOURCE_DIR}/include/kitsune/config.h.cmake
  ${LLVM_BINARY_DIR}/include/kitsune/config.h
)
