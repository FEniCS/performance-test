cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME dolfinx-scaling-test)
project(${PROJECT_NAME})

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Get DOLFIN configuration data (DOLFINConfig.cmake must be in
# DOLFIN_CMAKE_CONFIG_PATH)
find_package(Basix REQUIRED)
find_package(DOLFINX REQUIRED)

string(FIND "x-$ENV{PETSC_ARCH}" "complex" PETSC_SCALAR_COMPLEX)

message(STATUS ${PETSC_SCALAR_COMPLEX})

if(${PETSC_SCALAR_COMPLEX} EQUAL -1)
  set(SCALAR_TYPE "double")
else()
  set(SCALAR_TYPE "double complex")
endif()

set(CMAKE_BUILD_TYPE "Release")

# Compile UFL files
add_custom_command(
  OUTPUT Elasticity.c
  COMMAND ffcx --scalar_type ${SCALAR_TYPE} ${CMAKE_CURRENT_SOURCE_DIR}/Elasticity.ufl
  DEPENDS Elasticity.ufl
)
add_custom_command(
  OUTPUT Poisson.c
  COMMAND ffcx --scalar_type ${SCALAR_TYPE} ${CMAKE_CURRENT_SOURCE_DIR}/Poisson.ufl
  DEPENDS Poisson.ufl
)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Executable
add_executable(${PROJECT_NAME} main.cpp mesh.cpp elasticity_problem.cpp poisson_problem.cpp
${CMAKE_CURRENT_BINARY_DIR}/Elasticity.c
${CMAKE_CURRENT_BINARY_DIR}/Poisson.c)

# Find Boost program_options
if(DEFINED ENV{BOOST_ROOT} OR DEFINED BOOST_ROOT)
  set(Boost_NO_SYSTEM_PATHS on)
endif()
set(Boost_USE_MULTITHREADED $ENV{BOOST_USE_MULTITHREADED})
set(Boost_VERBOSE TRUE)
find_package(Boost 1.70 REQUIRED program_options)

# Target libraries
target_link_libraries(${PROJECT_NAME} dolfinx Boost::program_options)

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
