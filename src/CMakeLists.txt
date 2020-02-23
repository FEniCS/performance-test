cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME dolfinx-scaling-test)
project(${PROJECT_NAME})

# Get DOLFIN configuration data (DOLFINConfig.cmake must be in
# DOLFIN_CMAKE_CONFIG_PATH)
find_package(DOLFINX REQUIRED)

set(CMAKE_BUILD_TYPE "Release")

# Compile UFL files
add_custom_command(
  OUTPUT Elasticity.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/Elasticity.ufl
  DEPENDS Elasticity.ufl
)
add_custom_command(
  OUTPUT Poisson.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/Poisson.ufl
  DEPENDS Poisson.ufl
)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Executable
add_executable(${PROJECT_NAME} main.cpp mesh.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/Elasticity.c
  ${CMAKE_CURRENT_BINARY_DIR}/Poisson.c)

# Target libraries
target_link_libraries(${PROJECT_NAME} dolfinx ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
# target_compile_options(${PROJECT_NAME} PRIVATE -Wall -pedantic -Werror)

set_source_files_properties(main.cpp PROPERTIES COMPILE_FLAGS "-Wall -Wextra -pedantic -Werror")

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
