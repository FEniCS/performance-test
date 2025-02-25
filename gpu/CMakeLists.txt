cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)  # Boost CONFIG mode
endif()

set(CMAKE_CXX_EXTENSIONS OFF)

option(profiling "Enable ROCm scoped profiling" off)
option(omnitrace "Enable AMD omnitrace profiling" off)
option(cpu "Build for CPUs"	off)
option(nvidia "Build for NVidia GPUs" off)
option(amd "Build for AMD GPUs"	off)

# Slicing
option(SLICED "Used sliced algorithm" off)
set(SLICE_SIZE "2" CACHE STRING "Set slice size. Has effect only if slicing is enabled.")

# Optimisation setting for MI250/MI300
option(AMDMI_NOUNROLL "Don't unroll geometry loop in sliced kernel (optimise for MI250/MI300)" off)

# Scalar type
set(SCALAR_TYPE "float64" CACHE STRING "Set the floating point type")

if(cpu AND nvidia OR cpu AND amd)
  message(FATAL_ERROR "You cannot select both a cpu build and a GPU (amd or nvidia) build. Only select one of the three options: cpu, nvidia, amd")
endif()
if(nvidia AND amd)
  message(FATAL_ERROR "You cannot select both amd and nvidia GPU builds. Only select one of the three options: cpu, nvidia, amd")
endif()

project(mat_free C CXX)
if(amd)
  set(CMAKE_HIP_ARCHITECTURES gfx90a gfx942 gfx1100)
  enable_language(HIP)
elseif(nvidia)
  # See https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
  set(CMAKE_CUDA_ARCHITECTURES 86 89 90a)  # RTX A4000
  set(CMAKE_CUDA_STANDARD 20)
  enable_language(CUDA)
endif()

if(omnitrace AND NOT profiling)
  set(profiling on)
endif()
message(STATUS "Profiling is ${profiling}")
message(STATUS "Omnitrace labelling ${omnitrace}")

if(cpu)
  message(STATUS "Compiling for CPUs")
elseif(nvidia)
  message(STATUS "Compiling for NVIDIA GPUs")
elseif(amd)
  message(STATUS "Compiling for AMD GPUs")
else()
  message(FATAL_ERROR "No target selected for the build. Please select one of the three options: cpu, nvidia, amd")
endif()

set(CMAKE_CXX_FLAGS "-Ofast -march=native ${CMAKE_CXX_FLAGS} -DNDEBUG -Wall")
set(CMAKE_C_FLAGS "-Ofast -march=native ${CMAKE_C_FLAGS} -DNDEBUG -Wall")


if (SLICED)
  add_definitions(-DUSE_SLICED)
  add_definitions(-DSLICE_SIZE=${SLICE_SIZE})
  message("Slicing enabled: Slice size: ${SLICE_SIZE}")
  if (AMDMI_NOUNROLL)
    add_definitions(-DAMDMI_DNOUNROLL)
    message("Adding '#pragma unroll 1' to sliced kernel for MI250/MI300 optimisation")
  endif()
endif()

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

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -march=native -std=c++20 -O3 -DNDEBUG --extended-lambda --expt-relaxed-constexpr -lineinfo")

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

# Set the GPU to compile for
# if(amd)
#   set(GPU_TARGETS "gfx90a" CACHE STRING "GPU targets to compile for")
# endif()
# elseif(nvidia)
#   set(GPU_TARGETS "Ampere" CACHE STRING "GPU targets to compile for")
# endif()

add_custom_command(
  OUTPUT poisson.c
  COMMAND ffcx --scalar_type ${SCALAR_TYPE} --sum_factorization ${CMAKE_CURRENT_SOURCE_DIR}/poisson.py
  VERBATIM
  DEPENDS poisson.py
  COMMENT "Compile poisson.py using FFCx"
)

# If using ROCm and HIP, rocThrust requires rocPRIM
if(amd)
    # find_package(hip REQUIRED)
    # find_package(rocprim REQUIRED CONFIG)
    find_package(rocthrust REQUIRED CONFIG)
endif()

if(profiling)
	find_library(ROCTRACER_LIBRARY NAMES roctracer64)
	find_library(ROCTX64_LIBRARY NAMES roctx64)
	find_library(ROCM_SMI64_LIBRARY NAMES rocm_smi64)
	if(NOT omnitrace)
		add_definitions(-DROCM_TRACING=1)
	endif()
	add_definitions(-DROCM_SMI=1)
endif()

if(omnitrace)
	if(NOT DEFINED OMNITRACE_DIR)
		set(OMNITRACE_DIR "/usr")
	endif()
	find_path(OMNITRACE_INCLUDE omnitrace/user.h "${OMNITRACE_DIR}/include")
	include_directories(${OMNITRACE_INCLUDE})
	find_library(OMNITRACE_LIBRARY NAMES omnitrace HINTS "${OMNITRACE_DIR}/lib")
	find_library(OMNITRACE_USER_LIBRARY NAMES omnitrace-user HINTS "${OMNITRACE_DIR}/lib")
	add_definitions(-DOMNITRACE=1)
endif()

include_directories("../../")

add_executable(${PROJECT_NAME} main.cpp poisson.c)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_link_libraries(${PROJECT_NAME} dolfinx Basix::basix Boost::program_options)
if(amd)
  target_link_libraries(${PROJECT_NAME} roc::rocthrust)
endif()

if(amd)
  add_definitions(-DUSE_HIP=1)  # For pmg library
  set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
elseif(nvidia)
  add_definitions(-DUSE_CUDA=1) # For pmg library
  set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
elseif(cpu)
  add_definitions(-DCPU_ONLY=1) # For pmg library
endif()


if(profiling)
	target_link_libraries(${PROJECT_NAME} ${ROCTX64_LIBRARY} ${ROCTRACER_LIBRARY} ${ROCM_SMI64_LIBRARY})
endif()

if(omnitrace)
	target_link_libraries(${PROJECT_NAME} ${OMNITRACE_USER_LIBRARY} ${OMNITRACE_LIBRARY})
endif()
