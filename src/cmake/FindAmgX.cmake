# Find AmgX library

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_path(AMGX_INCLUDE_DIR NAME amgx_c.h
  PATHS ${AMGX_DIR} $ENV{AMGX_DIR}
  PATH_SUFFIXES include)
find_library(AMGX_LIBRARIES NAME amgxsh
  PATHS ${AMGX_DIR} $ENV{AMGX_DIR}
  PATH_SUFFIXES lib lib64)

mark_as_advanced(AMGX_FOUND
  AMGX_INCLUDE_DIR AMGX_LIBRARIES)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AmgX
  REQUIRED_VARS AMGX_INCLUDE_DIR AMGX_LIBRARIES)

if(AmgX_FOUND AND NOT TARGET AmgX)
  add_library(AmgX INTERFACE IMPORTED)
  target_link_libraries(AmgX INTERFACE MPI::MPI_CXX)
  target_link_libraries(AmgX INTERFACE CUDA::cudart CUDA::cublas)

  set_target_properties(AmgX PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${AMGX_INCLUDE_DIR}")
  target_link_libraries(AmgX INTERFACE ${AMGX_LIBRARIES})
endif()

