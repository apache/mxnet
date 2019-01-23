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
#
# Finds the OpenBLAS library.
#
# The following variables are set after configuration is done:
#
# - OpenBLAS_FOUND
# - OpenBLAS_INCLUDE_DIRS
# - OpenBLAS_LIBRARIES
#
# This script will try to find the OpenBLAS library using the following RESULT_VARs in this order:
#
# 1 - Use find_package(OpenBLAS) in Config mode.
# 2 - Find the files manually
#
# At each step that was just described, in order to guarantee that the library was found correctly,
# this script tries to compile this simple program:
#
#   #include <cblas.h>
#   int main() {
#     cblas_sasum(0, (float*)0, 0);
#     return 0;
#   }
#
# If this simple program can't be compiled with the OpenBLAS library discovered in the previous
# step, then it assumes the step failed and moves on to the next step.
#
# To control where OpenBLAS should be searched for, one can set the environment variable
# `OpenBLAS_HOME` point to the installation directory of OpenBLAS:
#
#   set OpenBLAS_HOME=c:\mxnet

include(CheckCSourceCompiles)
function(check_openblas_compiles RESULT_VAR)
  message(STATUS "Testing a simple OpenBLAS program with: (${RESULT_VAR}, "
                                                          "includes: ${OpenBLAS_INCLUDE_DIRS}, "
                                                          "libs: ${OpenBLAS_LIBRARIES})")
  set(CMAKE_REQUIRED_INCLUDES "${OpenBLAS_INCLUDE_DIRS}")
  set(CMAKE_REQUIRED_LIBRARIES "${OpenBLAS_LIBRARIES}")

  check_c_source_compiles("
      #include <cblas.h>
      int main() { cblas_sasum(0, (float*)0, 0); return 0; }
  " OPENBLAS_CAN_COMPILE)

  unset(CMAKE_REQUIRED_INCLUDES CACHE)
  unset(CMAKE_REQUIRED_LIBRARIES CACHE)

  if (NOT ${CAN_COMPILE})
    message(WARNING "Couldn't compile a simple program to test for OpenBLAS presence using the '${RESULT_VAR}' RESULT_VAR. "
            "To verify the error from the compiler, run cmake with the flag `--debug-trycompile`. "
            "Check that the include directories are correct and contain the 'cblas.h' file: ${OpenBLAS_INCLUDE_DIRS} "
            "Check that the library directory also contains the compiled library: '${OpenBLAS_LIBRARIES}'.")
  else()
    message(STATUS "Found OpenBLAS via ${RESULT_VAR} (include: ${OpenBLAS_INCLUDE_DIRS})")
    message(STATUS "Found OpenBLAS via ${RESULT_VAR} (lib: ${OpenBLAS_LIBRARIES})")
  endif()

  set(${RESULT_VAR} ${OPENBLAS_CAN_COMPILE} PARENT_SCOPE)
  unset(OPENBLAS_CAN_COMPILE CACHE)
endfunction()

macro(unset_openblas_variables)
  unset(OpenBLAS_FOUND)
  unset(OpenBLAS_FOUND CACHE)
  unset(OpenBLAS_LIBRARIES)
  unset(OpenBLAS_LIBRARIES CACHE)
  unset(OpenBLAS_INCLUDE_DIRS)
  unset(OpenBLAS_INCLUDE_DIRS CACHE)
endmacro()

# Try config-mode
find_package(OpenBLAS
             NO_MODULE  # Forcing Config mode, otherwise this would be an infinite loop.
             PATHS $ENV{OpenBLAS_HOME}
             )
if (OpenBLAS_FOUND)
  check_openblas_compiles(OpenBLAS_CONFIG_MODE)
  if (OpenBLAS_CONFIG_MODE)
    return()
  else()
    unset_openblas_variables()
  endif()
endif()

# Try finding the files manually
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
    "${OpenBLAS_ROOT}"

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
    "${OpenBLAS_ROOT}"

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

find_path(OpenBLAS_INCLUDE_DIRS
          NAMES cblas.h
          PATHS ${OpenBLAS_INCLUDE_SEARCH_PATHS}
          PATH_SUFFIXES include include/openblas)

set(OpenBLAS_LIB_NAMES openblas libopenblas)

if(CMAKE_CROSSCOMPILING)
  message(STATUS "Will try to link to OpenBLAS statically")
  set(OpenBLAS_LIB_NAMES libopenblas.a ${OpenBLAS_LIB_NAMES})
endif()

if(MSVC)
  # The OpenBLAS distributed in SourceForge has this non-standard file extension: *.a
  # From the README there, it shows:
  #   libopenblas.dll       The shared library for Visual Studio and GCC.
  #   libopenblas.a         The static library. Only work with GCC.
  #   libopenblas.dll.a     The import library for Visual Studio.
  # We have to use `libopenblas.dll.a` since this is the library file compiled for Visual
  # Studio. This might not be ideal for Windows users since they will also need the DLL to be on
  # their PATH, but that's how this distribution of OpenBLAS is.
  list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES .dll.a)
endif()

find_library(OpenBLAS_LIBRARIES
             NAMES ${OpenBLAS_LIB_NAMES}
             PATHS ${OpenBLAS_LIB_SEARCH_PATHS}
             PATH_SUFFIXES lib lib64)

check_openblas_compiles(OpenBLAS_MANUAL_MODE)
if (NOT OpenBLAS_MANUAL_MODE)
  unset_openblas_variables()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenBLAS DEFAULT_MSG OpenBLAS_INCLUDE_DIRS OpenBLAS_LIBRARIES)
