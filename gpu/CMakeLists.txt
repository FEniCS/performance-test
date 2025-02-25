cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)  # Boost CONFIG mode
endif()

set(CMAKE_CXX_EXTENSIONS OFF)

option(nvidia "Build for NVidia GPUs" off)
option(amd "Build for AMD GPUs"	off)

# Scalar type
set(SCALAR_TYPE "float64" CACHE STRING "Set the floating point type")

if(nvidia AND amd)
  message(FATAL_ERROR "You cannot select both amd and nvidia GPU builds. Only select one option:  nvidia, amd")
endif()

project(mat_free C CXX)
if(amd)
# MI250X = gfx90a
# MI300X = gfx942
# Radeon7900 = gfx1100
  set(CMAKE_HIP_ARCHITECTURES gfx90a gfx942 gfx1100)
  enable_language(HIP)
elseif(nvidia)
  # See https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# A100 = 80
# RTX6000Ada = 89
# GH200 = 90
  set(CMAKE_CUDA_ARCHITECTURES 80 89 90)
  set(CMAKE_CUDA_STANDARD 20)
  enable_language(CUDA)
endif()

if(nvidia)
  message(STATUS "Compiling for NVIDIA GPUs")
elseif(amd)
  message(STATUS "Compiling for AMD GPUs")
else()
  message(FATAL_ERROR "No target selected for the build. Please select one of the options: nvidia, amd")
endif()

set(CMAKE_CXX_FLAGS "-Ofast -march=native ${CMAKE_CXX_FLAGS} -DNDEBUG -Wall")
set(CMAKE_C_FLAGS "-Ofast -march=native ${CMAKE_C_FLAGS} -DNDEBUG -Wall")

if (SCALAR_TYPE STREQUAL "float32")
  add_definitions(-DSCALAR_TYPE=float)
elseif(SCALAR_TYPE STREQUAL "float64")
  add_definitions(-DSCALAR_TYPE=double)
else()
  message(FATAL_ERROR "Scalar type ${SCALAR_TYPE} not supported.")
endif()

if (CMAKE_HIP_COMPILER_VERSION GREATER_EQUAL 18) # See https://rocm.docs.amd.com/projects/llvm-project/en/docs-6.2.1/LLVM/clang/html/ReleaseNotes.html#c-specific-potentially-breaking-changes
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fclang-abi-compat=17")
endif()
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -std=c++20 -Wall -Ofast -DNDEBUG -munsafe-fp-atomics")

if (DEFINED ENV{CRAYPE_VERSION})
   message("Detected CrayPE - adding MPI flags")
   set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Ofast -march=native -I/opt/cray/pe/mpich/8.1.27/ofi/gnu/9.1/include")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -march=native -std=c++20 -O3 -DNDEBUG --extended-lambda --expt-relaxed-constexpr")

# ------------------------------------------------------------------------------
# General configuration
find_package(MPI)
find_package(Basix REQUIRED)
find_package(DOLFINX REQUIRED)

if(DEFINED ENV{BOOST_ROOT} OR DEFINED BOOST_ROOT)
  set(Boost_NO_SYSTEM_PATHS on)
endif()
find_package(Boost 1.70 REQUIRED program_options)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_custom_command(
  OUTPUT poisson.c
  COMMAND ffcx --scalar_type ${SCALAR_TYPE} --sum_factorization ${CMAKE_CURRENT_SOURCE_DIR}/poisson.py
  VERBATIM
  DEPENDS poisson.py
  COMMENT "Compile poisson.py using FFCx"
)

if(amd)
    find_package(rocthrust REQUIRED CONFIG)
endif()

add_executable(${PROJECT_NAME} main.cpp poisson.c)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_link_libraries(${PROJECT_NAME} dolfinx Basix::basix Boost::program_options)
if(amd)
  target_link_libraries(${PROJECT_NAME} roc::rocthrust)
endif()

if(amd)
  add_definitions(-DUSE_HIP=1)
  set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
elseif(nvidia)
  add_definitions(-DUSE_CUDA=1)
  set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
endif()
