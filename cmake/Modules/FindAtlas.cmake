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

# Find the Atlas (and Lapack) libraries
#
# The following variables are optionally searched for defaults
#
# Atlas_ROOT_DIR:    Base directory where all Atlas components are found
# Atlas_NEED_LAPACK: Whether need LAPACK libraries
#
# The following are set after configuration is done:
#
# Atlas_FOUND
# Atlas_LAPACK_FOUND
# Atlas_INCLUDE_DIRS
# Atlas_LIBRARIES

if(Atlas_FOUND)
  return()
endif()

if(CMAKE_CROSSCOMPILING)
  set(Atlas_INCLUDE_SEARCH_PATHS
      ${Atlas_INCLUDE_SEARCH_PATHS}

      "$ENV{CROSS_ROOT}"
      "${CROSS_ROOT}"
      )
endif()

set(Atlas_INCLUDE_SEARCH_PATHS
    ${Atlas_INCLUDE_SEARCH_PATHS}

    "$ENV{Atlas_ROOT_DIR}"
    "${Atlas_ROOT_DIR}"

    /usr/include/atlas
    /usr/include/atlas-base
    )

if(${CMAKE_CROSSCOMPILING})
  set(Atlas_LIB_SEARCH_PATHS
      ${Atlas_LIB_SEARCH_PATHS}

      "$ENV{CROSS_ROOT}"
      "${CROSS_ROOT}"
      )
endif()

set(Atlas_LIB_SEARCH_PATHS
    ${Atlas_LIB_SEARCH_PATHS}

    "$ENV{Atlas_ROOT_DIR}"
    "${Atlas_ROOT_DIR}"

    /usr/lib/atlas
    /usr/lib/atlas-base
    )

find_path(Atlas_CBLAS_INCLUDE_DIR
          NAMES cblas.h
          PATHS ${Atlas_INCLUDE_SEARCH_PATHS}
          PATH_SUFFIXES include)

find_library(Atlas_CBLAS_LIBRARY
             NAMES ptcblas_r ptcblas cblas_r cblas
             PATHS ${Atlas_LIB_SEARCH_PATHS}
             PATH_SUFFIXES lib)
find_library(Atlas_BLAS_LIBRARY
             NAMES atlas_r atlas
             PATHS ${Atlas_LIB_SEARCH_PATHS}
             PATH_SUFFIXES lib)

set(LOOKED_FOR
    Atlas_CBLAS_INCLUDE_DIR

    Atlas_CBLAS_LIBRARY
    Atlas_BLAS_LIBRARY
    )

set(Atlas_LAPACK_FOUND)
set(Atlas_CLAPACK_INCLUDE_DIR)
set(Atlas_LAPACK_LIBRARY)

if(Atlas_NEED_LAPACK)
  message(STATUS "Looking for LAPACK support...")

  # we need another variables (starting with __) because cmake will not overwrite it if already set
  find_path(__Atlas_CLAPACK_INCLUDE_DIR
            NAMES clapack.h
            PATHS ${Atlas_INCLUDE_SEARCH_PATHS})
  find_library(__Atlas_LAPACK_LIBRARY
               NAMES lapack_r lapack lapack_atlas
               PATHS ${Atlas_LIB_SEARCH_PATHS})

  set(CMAKE_REQUIRED_LIBRARIES ${Atlas_LAPACK_LIBRARY})
  include(CheckFunctionExists)
  check_function_exists("cgees_" LAPACK_FOUND)

  if(LAPACK_FOUND)
    set(Atlas_LAPACK_FOUND True)
    set(Atlas_CLAPACK_INCLUDE_DIR ${__Atlas_CLAPACK_INCLUDE_DIR})
    set(Atlas_LAPACK_LIBRARY ${__Atlas_LAPACK_LIBRARY})

    set(LOOKED_FOR
        ${LOOKED_FOR}
        Atlas_CLAPACK_INCLUDE_DIR
        Atlas_LAPACK_LIBRARY)

    message(STATUS "LAPACK found")
  else()
    message(WARNING "LAPACK with Atlas could not be found, LAPACK functionality will not be available")
  endif()

endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Atlas DEFAULT_MSG ${LOOKED_FOR})

if(Atlas_FOUND)
  set(Atlas_INCLUDE_DIRS "${Atlas_CBLAS_INCLUDE_DIR}" "${Atlas_CLAPACK_INCLUDE_DIR}")
  set(Atlas_LIBRARIES "${Atlas_LAPACK_LIBRARY}" "${Atlas_CBLAS_LIBRARY}" "${Atlas_BLAS_LIBRARY}")

  mark_as_advanced(${LOOKED_FOR})

  message(STATUS "Found Atlas (include: ${Atlas_INCLUDE_DIRS}, libraries: ${Atlas_LIBRARIES})")
endif()
