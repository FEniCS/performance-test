cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME dolfinx-scaling-test)
project(${PROJECT_NAME})

include(GNUInstallDirs)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Use C++17
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Get DOLFIN configuration data (DOLFINConfig.cmake must be in
# DOLFIN_CMAKE_CONFIG_PATH)
find_package(DOLFINX REQUIRED)

# set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-Ofast -march=native ${CMAKE_CXX_FLAGS} -DNDEBUG -Wall")
set(CMAKE_C_FLAGS "-Ofast -march=native ${CMAKE_C_FLAGS} -DNDEBUG -Wall")

# Compile UFL files
add_custom_command(
  OUTPUT Poisson.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/Poisson.py
  DEPENDS Poisson.py
)

add_custom_command(
  OUTPUT Elasticity.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/Elasticity.py
  DEPENDS Elasticity.py
)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Executable
add_executable(${PROJECT_NAME} main.cpp mesh.cpp elasticity_problem.cpp cgpoisson_problem.cpp poisson_problem.cpp mem.cpp
${CMAKE_CURRENT_BINARY_DIR}/Elasticity.c
${CMAKE_CURRENT_BINARY_DIR}/Poisson.c)

# Find Boost program_options
if(DEFINED ENV{BOOST_ROOT} OR DEFINED BOOST_ROOT)
  set(Boost_NO_SYSTEM_PATHS on)
endif()
# set(Boost_USE_MULTITHREADED $ENV{BOOST_USE_MULTITHREADED})
set(Boost_VERBOSE TRUE)
find_package(Boost 1.70 REQUIRED program_options)

# Target libraries
target_link_libraries(${PROJECT_NAME} dolfinx Boost::program_options pthread)

message(STATUS ${CMAKE_CXX_FLAGS})
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
