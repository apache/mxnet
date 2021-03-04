# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

# ----------
# FindACL
# ----------
#
# Finds the Arm Compute Library
# https://arm-software.github.io/ComputeLibrary/latest/
#
# This module defines the following variables:
#
#   ACL_FOUND          - True if ACL was found
#   ACL_INCLUDE_DIRS   - include directories for ACL
#   ACL_LIBRARIES      - link against this library to use ACL
#
# The module will also define two cache variables:
#
#   ACL_INCLUDE_DIR    - the ACL include directory
#   ACL_LIBRARY        - the path to the ACL library
#

# Use ACL_ROOT_DIR environment variable to find the library and headers
find_path(ACL_INCLUDE_DIR
  NAMES arm_compute/graph.h
  PATHS ENV ACL_ROOT_DIR
  NO_DEFAULT_PATH
  )

find_library(ACL_LIBRARY
  NAMES arm_compute
  PATHS ENV ACL_ROOT_DIR
  PATH_SUFFIXES build
  NO_DEFAULT_PATH
  )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ACL DEFAULT_MSG
  ACL_INCLUDE_DIR
  ACL_LIBRARY
)

mark_as_advanced(
  ACL_LIBRARY
  ACL_INCLUDE_DIR
  )

# Find the extra libraries and include dirs
if(ACL_FOUND)
  find_path(ACL_EXTRA_INCLUDE_DIR
    NAMES half/half.hpp
    PATHS ENV ACL_ROOT_DIR
    PATH_SUFFIXES include
    )

  find_library(ACL_GRAPH_LIBRARY
    NAMES arm_compute_graph
    PATHS ENV ACL_ROOT_DIR
    PATH_SUFFIXES build
    )

  find_library(ACL_CORE_LIBRARY
    NAMES arm_compute_core
    PATHS ENV ACL_ROOT_DIR
    PATH_SUFFIXES build
    )

  list(APPEND ACL_INCLUDE_DIRS
    ${ACL_INCLUDE_DIR} ${ACL_EXTRA_INCLUDE_DIR})
  list(APPEND ACL_LIBRARIES
    ${ACL_LIBRARY} ${ACL_GRAPH_LIBRARY} ${ACL_CORE_LIBRARY})
endif()


