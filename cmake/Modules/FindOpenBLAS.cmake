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

# Find the OpenBLAS libraries
#
# The following variables are optionally searched for defaults
#
# OpenBLAS_HOME:        Base directory where all OpenBLAS components are found
# OpenBLAS_NEED_LAPACK: Whether need lapack libraries
#
# The following are set after configuration is done:
#
# OpenBLAS_FOUND
# OpenBLAS_LAPACK_FOUND
# OpenBLAS_INCLUDE_DIRS
# OpenBLAS_LIBRARIES

if(OpenBLAS_FOUND)
  return()
endif()

if(CMAKE_CROSSCOMPILING)
  set(OpenBLAS_INCLUDE_SEARCH_PATHS
      ${OpenBLAS_INCLUDE_SEARCH_PATHS}

      "$ENV{CROSS_ROOT}"
      "${CROSS_ROOT}"
      )
endif()

set(OpenBLAS_INCLUDE_SEARCH_PATHS
    ${OpenBLAS_INCLUDE_SEARCH_PATHS}

    "$ENV{OpenBLAS_HOME}"
    "${OpenBLAS_HOME}"

    /usr
    /usr/include/openblas
    /usr/include/openblas-base
    /usr/local
    /usr/local/include/openblas
    /usr/local/include/openblas-base
    /opt/OpenBLAS
    /usr/local/opt/openblas

    "${PROJECT_SOURCE_DIR}/3rdparty/OpenBLAS"
    "${PROJECT_SOURCE_DIR}/thirdparty/OpenBLAS"
    )

if(CMAKE_CROSSCOMPILING)
  set(Open_BLAS_LIB_SEARCH_PATHS
      ${Open_BLAS_LIB_SEARCH_PATHS}

      "$ENV{CROSS_ROOT}"
      "${CROSS_ROOT}"
      )
endif()

set(OpenBLAS_LIB_SEARCH_PATHS
    ${OpenBLAS_LIB_SEARCH_PATHS}

    "$ENV{OpenBLAS_HOME}"
    "${OpenBLAS_HOME}"

    /
    /lib/openblas-base
    /usr
    /usr/lib/openblas-base
    /usr/local/
    /opt/OpenBLAS
    /usr/local/opt/openblas

    "${PROJECT_SOURCE_DIR}/3rdparty/OpenBLAS"
    "${PROJECT_SOURCE_DIR}/thirdparty/OpenBLAS"
    )

find_path(OpenBLAS_INCLUDE_DIR
          NAMES cblas.h
          PATHS ${OpenBLAS_INCLUDE_SEARCH_PATHS}
          PATH_SUFFIXES include)

set(OpenBLAS_LIB_NAMES openblas libopenblas.dll.a libopenblas.dll)

if(CMAKE_CROSSCOMPILING OR MSVC)
  message(STATUS "Will try to link to OpenBLAS statically")
  set(OpenBLAS_LIB_NAMES libopenblas.a ${OpenBLAS_LIB_NAMES})
endif()

# For some reason setting this is really important and on windows the library is not found even given exact file name
if(MSVC)
  set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

find_library(OpenBLAS_LIBRARY
             NAMES ${OpenBLAS_LIB_NAMES}
             PATHS ${OpenBLAS_LIB_SEARCH_PATHS}
             PATH_SUFFIXES lib lib64)

set(LOOKED_FOR
    OpenBLAS_INCLUDE_DIR
    OpenBLAS_LIBRARY
    )

set(OpenBLAS_LAPACK_FOUND)
set(OpenBLAS_LAPACK_LIBRARY)
set(OpenBLAS_LAPACK_INCLUDE_DIR)

if(OpenBLAS_NEED_LAPACK)
  message(STATUS "Looking for LAPACK support...")

  # we need another variable (starting with __) because cmake will not overwrite it if already set
  find_path(__OpenBLAS_LAPACK_INCLUDE_DIR
            NAMES lapacke.h
            PATHS ${OpenBLAS_INCLUDE_SEARCH_PATHS})

  # OpenBLAS does not have a separate LAPACK library: https://github.com/xianyi/OpenBLAS/issues/296
  # LAPACK if present in OpenBLAS build is included into libopenblas
  set(__OpenBLAS_LAPACK_LIBRARY ${OpenBLAS_LIBRARY})

  set(CMAKE_REQUIRED_LIBRARIES ${__OpenBLAS_LAPACK_LIBRARY})
  include(CheckFunctionExists)
  check_function_exists("cheev_" LAPACK_FOUND)

  if(LAPACK_FOUND)
    set(OpenBLAS_LAPACK_FOUND True)
    set(OpenBLAS_LAPACK_INCLUDE_DIR ${__OpenBLAS_LAPACK_INCLUDE_DIR})
    set(OpenBLAS_LAPACK_LIBRARY ${__OpenBLAS_LAPACK_LIBRARY})

    set(LOOKED_FOR
        ${LOOKED_FOR}
        OpenBLAS_LAPACK_INCLUDE_DIR
        OpenBLAS_LAPACK_LIBRARY
        )
    message(STATUS "LAPACK found")
  else()
    message(WARNING "OpenBlas has not been compiled with LAPACK support, LAPACK functionality will not be available")
  endif()

endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenBLAS DEFAULT_MSG ${LOOKED_FOR})

if(OpenBLAS_FOUND)
  set(OpenBLAS_INCLUDE_DIRS "${OpenBLAS_INCLUDE_DIR}" "${OpenBLAS_LAPACK_INCLUDE_DIR}")
  set(OpenBLAS_LIBRARIES "${OpenBLAS_LIBRARY}" "${OpenBLAS_LAPACK_LIBRARY}")

  mark_as_advanced(${LOOKED_FOR})

  message(STATUS "Found OpenBLAS (include: ${OpenBLAS_INCLUDE_DIRS}, libraries: ${OpenBLAS_LIBRARIES})")
endif()
