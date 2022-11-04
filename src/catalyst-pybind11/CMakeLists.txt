cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME catalyst_pybind)
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

find_package(pybind11 REQUIRED)

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Python3 COMPONENTS REQUIRED Interpreter Development)
find_package(pybind11 CONFIG)

# pybind11 method:
pybind11_add_module(${PROJECT_NAME} CatalystAdaptor-pybind.cpp)

option(USE_CATALYST "Build example with Catalyst enabled" ON)
if (USE_CATALYST)
  find_package(catalyst REQUIRED PATHS "${ParaView_DIR}/catalyst")
  target_compile_definitions(${PROJECT_NAME}
    PRIVATE "PARAVIEW_IMPL_DIR=\"${ParaView_CATALYST_DIR}\"" USE_CATALYST=1)
  target_link_libraries(${PROJECT_NAME} PRIVATE catalyst::catalyst)
endif()

# Target libraries
target_link_libraries(${PROJECT_NAME} PRIVATE dolfinx)

# install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
