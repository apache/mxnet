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
# MKL_FOUND
# MKL_INCLUDE_DIR
# MKL_LIBRARIES
# MKL_USE_INTEL_OMP

if(MKL_FOUND)
  return()
endif()

set(MKLML_INCLUDE_SEARCH_PATHS
    ${MKLML_INCLUDE_SEARCH_PATHS}

    "$ENV{MKLROOT}"
    "${MKLROOT}"

    "${PROJECT_SOURCE_DIR}/3rdparty/MKLML"

    /usr
    /usr/local
    )

# ---[ Find libraries
if(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(PATH_SUFFIXES lib lib/ia32)
else()
  set(PATH_SUFFIXES lib lib/intel64)
endif()

set(MKLML_LIB_SEARCH_PATHS
    ${MKLML_LIB_SEARCH_PATHS}

    "$ENV{MKLROOT}"
    "${MKLROOT}"

    "${PROJECT_SOURCE_DIR}/3rdparty/MKLML"

    /usr
    /usr/local
    )

find_path(MKLML_INCLUDE_DIR
          NAMES mkl_blas.h
          PATHS ${MKLML_INCLUDE_SEARCH_PATHS}
          PATH_SUFFIXES include)

set(LOOKED_FOR
    MKLML_INCLUDE_DIR
    )

set(MKLML_LIBRARIES)

# Find Intel OpenMP
set(MKL_USE_INTEL_OMP)

find_library(IOMP5_LIBRARY
             NAMES iomp5 libiomp5.lib libiomp5md.lib
             PATHS ${MKLML_LIB_SEARCH_PATHS}
             PATH_SUFFIXES ${PATH_SUFFIXES}
             )
mark_as_advanced(IOMP5_LIBRARY)

if(IOMP5_LIBRARY)
  list(APPEND LOOKED_FOR IOMP5_LIBRARY)
  list(APPEND MKLML_LIBRARIES ${IOMP5_LIBRARY})

  set(MKL_USE_INTEL_OMP True)
endif()

# add static windows libs first
set(__MKL_LIB_NAMES libmklml.lib libmklmlmd.lib)

if(MKL_USE_INTEL_OMP)
  list(APPEND __MKL_LIB_NAMES mklml_intel)
else()
  list(APPEND __MKL_LIB_NAMES mklml_gnu)
endif()

list(APPEND __MKL_LIB_NAMES mklml)

mark_as_advanced(__MKL_LIB_NAMES)

message("MKLML_LIBRARY=${MKLML_LIBRARY}")
message("__MKL_LIB_NAMES=${__MKL_LIB_NAMES}")
message("MKLML_LIB_SEARCH_PATHS=${MKLML_LIB_SEARCH_PATHS}")
message("PATH_SUFFIXES=${PATH_SUFFIXES}")

find_library(MKLML_LIBRARY
             NAMES ${__MKL_LIB_NAMES}
             PATHS ${MKLML_LIB_SEARCH_PATHS}
             PATH_SUFFIXES ${PATH_SUFFIXES}
             )

mark_as_advanced(MKLML_LIBRARY)

message("AFTER: MKLML_LIBRARY=${MKLML_LIBRARY}")

list(APPEND LOOKED_FOR MKLML_LIBRARY)
list(APPEND MKLML_LIBRARIES ${MKLML_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKLML DEFAULT_MSG ${LOOKED_FOR})

if(MKLML_FOUND)
  set(MKL_FOUND ${MKLML_FOUND})
  set(MKL_LIBRARIES ${MKLML_LIBRARIES})
  set(MKL_INCLUDE_DIR "${MKLML_INCLUDE_DIR}")

  mark_as_advanced(${LOOKED_FOR})

  message(STATUS "Found MKLML (include: ${MKL_INCLUDE_DIR}, libraries: ${MKL_LIBRARIES})")
endif()
