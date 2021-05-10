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

set(CMAKE_BUILD_TYPE "Release")

# Compile UFL files

foreach (X RANGE 1 8)
  add_custom_command(
    OUTPUT Poisson${X}.c
    COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/Poisson${X}.ufl
    DEPENDS Poisson${X}.ufl
  )
  add_custom_command(
    OUTPUT Elasticity${X}.c
    COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/Elasticity${X}.ufl
    DEPENDS Elasticity${X}.ufl
  )
endforeach()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Executable
add_executable(${PROJECT_NAME} main.cpp mesh.cpp elasticity_problem.cpp poisson_problem.cpp
${CMAKE_CURRENT_BINARY_DIR}/Elasticity1.c
${CMAKE_CURRENT_BINARY_DIR}/Elasticity2.c
${CMAKE_CURRENT_BINARY_DIR}/Poisson1.c 
${CMAKE_CURRENT_BINARY_DIR}/Poisson2.c
${CMAKE_CURRENT_BINARY_DIR}/Poisson3.c)

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
