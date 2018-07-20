# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Find the MKLML (subset of MKL) libraries
#
# The following variables are optionally searched for defaults
#
# MKLROOT: Base directory where all MKL/MKLML components are found
#
# The following are set after configuration is done:
#
# MKLML_FOUND
# MKLML_INCLUDE_DIRS
# MKLML_LIBRARIES

if($ENV{MKLROOT})
  file(TO_CMAKE_PATH "$ENV{MKLROOT}" MKLROOT)
  message(STATUS "MKLROOT=${MKLROOT}")
endif()

set(MKLML_INCLUDE_SEARCH_PATHS
    ${MKLML_INCLUDE_SEARCH_PATHS}

    "${MKLROOT}"
    "${MKLROOT}/include"

    "${PROJECT_SOURCE_DIR}/3rdparty/MKLML/include"
    )

# ---[ Find libraries
if(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(PATH_SUFFIXES lib lib/ia32)
else()
  set(PATH_SUFFIXES lib lib/intel64)
endif()

set(MKLML_LIB_SEARCH_PATHS
    ${MKLML_LIB_SEARCH_PATHS}

    "${MKLROOT}"

    "${PROJECT_SOURCE_DIR}/3rdparty/MKLML/lib"
    )

find_path(MKLML_INCLUDE_DIR
          NAMES mkl_blas.h
          PATHS ${MKLML_INCLUDE_SEARCH_PATHS})

set(LOOKED_FOR
    MKLML_INCLUDE_DIR
    )

set(MKLML_LIBS iomp5)

if(WIN32)
  list(APPEND MKLML_LIBS mklml)
elseif(APPLE)
  list(APPEND MKLML_LIBS mklml)
else()
  list(APPEND MKLML_LIBS mklml_gnu)
endif()

foreach(__lib ${MKLML_LIBS})
  set(__mkl_lib "${__lib}")
  string(TOUPPER ${__mkl_lib} __mkl_lib_upper)

  # add static windows libs first
  set(__mkl_lib_names lib${__mkl_lib}.lib lib${__mkl_lib}md.lib)

  find_library(${__mkl_lib_upper}_LIBRARY
               NAMES ${__mkl_lib_names} ${__mkl_lib}
               PATHS ${MKLML_LIB_SEARCH_PATHS}
               PATH_SUFFIXES ${PATH_SUFFIXES}
               )
  mark_as_advanced(${__mkl_lib_upper}_LIBRARY)

  list(APPEND LOOKED_FOR ${__mkl_lib_upper}_LIBRARY)
  list(APPEND MKLML_LIBRARIES ${${__mkl_lib_upper}_LIBRARY})
endforeach()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKLML DEFAULT_MSG ${LOOKED_FOR})

if(MKLML_FOUND)
  set(MKLML_INCLUDE_DIRS "${MKLML_INCLUDE_DIR}")

  mark_as_advanced(${LOOKED_FOR})

  message(STATUS "Found MKLML (include: ${MKLML_INCLUDE_DIRS}, libraries: ${MKLML_LIBRARIES})")
endif()
