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

file(TO_CMAKE_PATH "$ENV{OpenBLAS_HOME}" OpenBLAS_HOME)
file(TO_CMAKE_PATH "$ENV{OpenBLAS}" OpenBLAS_DIR)
file(TO_CMAKE_PATH "$ENV{CROSS_ROOT}" CROSS_ROOT)

if(CMAKE_CROSSCOMPILING)
  set(OpenBLAS_INCLUDE_SEARCH_PATHS
      ${CROSS_ROOT}
      ${CROSS_ROOT}/include
      )
endif()

set(OpenBLAS_INCLUDE_SEARCH_PATHS
    ${OpenBLAS_INCLUDE_SEARCH_PATHS}

    ${OpenBLAS_HOME}
    ${OpenBLAS_HOME}/include

    /usr/include
    /usr/include/openblas
    /usr/include/openblas-base
    /usr/local/include
    /usr/local/include/openblas
    /usr/local/include/openblas-base
    /opt/OpenBLAS/include
    /usr/local/opt/openblas/include

    ${PROJECT_SOURCE_DIR}/3rdparty/OpenBLAS/include
    ${PROJECT_SOURCE_DIR}/thirdparty/OpenBLAS/include
    )

if(CMAKE_CROSSCOMPILING)
  set(Open_BLAS_LIB_SEARCH_PATHS
      ${CROSS_ROOT}
      ${CROSS_ROOT}/include
      )
endif()

set(OpenBLAS_LIB_SEARCH_PATHS
    ${OpenBLAS_LIB_SEARCH_PATHS}

    ${OpenBLAS_DIR}
    ${OpenBLAS_DIR}/lib
    ${OpenBLAS_DIR}/lib64
    ${OpenBLAS_HOME}
    ${OpenBLAS_HOME}/lib
    ${OpenBLAS_HOME}/lib64

    /lib/
    /lib/openblas-base
    /lib64/
    /usr/lib
    /usr/lib/openblas-base
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
    /opt/OpenBLAS/lib
    /usr/local/opt/openblas/lib

    ${PROJECT_SOURCE_DIR}/3rdparty/OpenBLAS/lib
    ${PROJECT_SOURCE_DIR}/thirdparty/OpenBLAS/lib
    )

find_path(OpenBLAS_INCLUDE_DIR
          NAMES cblas.h
          PATHS ${OpenBLAS_INCLUDE_SEARCH_PATHS})
find_library(OpenBLAS_LIBRARY
             NAMES openblas libopenblas.dll.a libopenblas.dll
             PATHS ${OpenBLAS_LIB_SEARCH_PATHS})

set(LOOKED_FOR
    OpenBLAS_INCLUDE_DIR
    OpenBLAS_LIBRARY
    )

if(OpenBLAS_NEED_LAPACK)
  message(STATUS "Looking for lapack support...")

  find_path(OpenBLAS_LAPACK_INCLUDE_DIR
            NAMES lapacke.h
            PATHS ${OpenBLAS_INCLUDE_SEARCH_PATHS})

  if(UNIX AND NOT APPLE)
    # lapack if present in OpenBLAS build is included into the libopenblas. But it requires gfortran to be linked
    # dynamically.
    # OpenBLAS does not have a separate lapack library: https://github.com/xianyi/OpenBLAS/issues/296
    # Static linking goes with openblas, but fortran needs to be linked dynamically:
    # https://github.com/xianyi/OpenBLAS/issues/460#issuecomment-61293128
    find_library(OpenBLAS_LAPACK_LIBRARY
                 NAMES gfortran
                 PATHS ${OpenBLAS_LIB_SEARCH_PATHS})
  else()
    set(OpenBLAS_LAPACK_LIBRARY ${OpenBLAS_LIBRARY})
  endif()

  set(CMAKE_REQUIRED_LIBRARIES ${OpenBLAS_LIBRARY})
  include(CheckFunctionExists)
  check_function_exists("cheev_" LAPACK_FOUND)

  if(LAPACK_FOUND)
    set(OpenBLAS_LAPACK_FOUND True)

    set(LOOKED_FOR
        ${LOOKED_FOR}
        OpenBLAS_LAPACK_INCLUDE_DIR
        OpenBLAS_LAPACK_LIBRARY
        )
    message(STATUS "Lapack found")
  else()
    set(OpenBLAS_LAPACK_FOUND False)
    message(WARNING "OpenBlas has not been compiled with Lapack support, lapack functionality will not be available")
  endif()

endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenBLAS DEFAULT_MSG ${LOOKED_FOR})

if(OpenBLAS_FOUND)
  set(OpenBLAS_INCLUDE_DIRS ${OpenBLAS_INCLUDE_DIR} ${OpenBLAS_LAPACK_INCLUDE_DIR})
  set(OpenBLAS_LIBRARIES ${OpenBLAS_LIBRARY} ${OpenBLAS_LAPACK_LIBRARY})

  mark_as_advanced(${LOOKED_FOR})

  message(STATUS "Found OpenBLAS (include: ${OpenBLAS_INCLUDE_DIRS}, libraries: ${OpenBLAS_LIBRARIES})")
endif()
