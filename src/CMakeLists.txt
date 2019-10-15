cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME dolfin-scaling-test)
project(${PROJECT_NAME})

# Get DOLFIN configuration data (DOLFINConfig.cmake must be in
# DOLFIN_CMAKE_CONFIG_PATH)
find_package(DOLFIN REQUIRED)
include(${DOLFIN_USE_FILE})

set(CMAKE_BUILD_TYPE "Release")

# set(CMAKE_CXX_FLAGS "-g")

find_package(Boost COMPONENTS program_options REQUIRED)

# Compile UFL files
add_custom_command(
  OUTPUT Elasticity.c
  COMMAND ffc Elasticity.ufl
  DEPENDS Elasticity.ufl
)
add_custom_command(
  OUTPUT Poisson.c
  COMMAND ffc Poisson.ufl
  DEPENDS Poisson.ufl
)

# Executable
add_executable(${PROJECT_NAME} main.cpp mesh.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/Elasticity.c
  ${CMAKE_CURRENT_BINARY_DIR}/Poisson.c)

# Target libraries
target_link_libraries(${PROJECT_NAME} dolfin ${Boost_PROGRAM_OPTIONS_LIBRARY})
