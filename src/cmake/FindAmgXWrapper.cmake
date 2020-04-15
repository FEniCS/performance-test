# Find AmgXWrapper library

find_package(AmgX REQUIRED)
find_package(PETSc REQUIRED)

find_path(AMGXWRAPPER_INCLUDE_DIR NAME AmgXSolver.hpp
  PATHS ${AMGX_DIR} $ENV{AMGX_DIR}
  PATH_SUFFIXES include)
find_library(AMGXWRAPPER_LIBRARIES NAME AmgXWrapper
  PATHS ${AMGX_DIR} $ENV{AMGX_DIR}
  PATH_SUFFIXES lib lib64)

mark_as_advanced(AMGXWRAPPER_FOUND
  AMGXWRAPPER_INCLUDE_DIR AMGXWRAPPER_LIBRARIES)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AmgXWrapper
  REQUIRED_VARS AMGXWRAPPER_INCLUDE_DIR AMGXWRAPPER_LIBRARIES)

if(AmgXWrapper_FOUND AND NOT TARGET AmgXWrapper)
  add_library(AmgXWrapper INTERFACE IMPORTED)
  target_link_libraries(AmgXWrapper INTERFACE AmgX)
  target_link_libraries(AmgXWrapper INTERFACE PETSC::petsc)

  set_target_properties(AmgXWrapper PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${AMGXWRAPPER_INCLUDE_DIR}")
  target_link_libraries(AmgXWrapper INTERFACE ${AMGXWRAPPER_LIBRARIES})
endif()

