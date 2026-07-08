cmake_minimum_required(VERSION 3.16)

project(dolfinx-scaling-test)

include(GNUInstallDirs)

find_package(DOLFINX REQUIRED)
find_package(Threads REQUIRED)

# Compile UFL files
add_custom_command(
  OUTPUT Poisson.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/Poisson.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Poisson.py
  VERBATIM
)

add_custom_command(
  OUTPUT Elasticity.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/Elasticity.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Elasticity.py
  VERBATIM
)

# Find Boost program_options
if(DEFINED ENV{BOOST_ROOT} OR DEFINED BOOST_ROOT)
  set(Boost_NO_SYSTEM_PATHS ON)
endif()
set(Boost_VERBOSE TRUE)
find_package(Boost 1.70 REQUIRED program_options)

add_executable(
  ${PROJECT_NAME}
  main.cpp
  mesh.cpp
  elasticity_problem.cpp
  cgpoisson_problem.cpp
  poisson_problem.cpp
  mem.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/Elasticity.c
  ${CMAKE_CURRENT_BINARY_DIR}/Poisson.c
)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)

set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES
    CXX_EXTENSIONS OFF
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

set(COMPILE_FLAGS -O3 -ffast-math -g -Wall)
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_FLAGS})

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(
  ${PROJECT_NAME}
  dolfinx::dolfinx
  Boost::program_options
  Threads::Threads
)

message(STATUS "Compile flags: ${COMPILE_FLAGS}")

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
