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

file(TO_CMAKE_PATH "$ENV{OpenBLAS_HOME}" OpenBLAS_HOME)
file(TO_CMAKE_PATH "$ENV{OpenBLAS}" OpenBLAS_DIR)

SET(Open_BLAS_INCLUDE_SEARCH_PATHS
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
  ${OpenBLAS_HOME}
  ${OpenBLAS_HOME}/include
)

SET(Open_BLAS_LIB_SEARCH_PATHS
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
        ${OpenBLAS_DIR}
        ${OpenBLAS_DIR}/lib
        ${OpenBLAS_HOME}
        ${OpenBLAS_HOME}/lib
 )

FIND_PATH(OpenBLAS_INCLUDE_DIR NAMES cblas.h PATHS ${Open_BLAS_INCLUDE_SEARCH_PATHS})
# the Julia's private OpenBLAS is named as `libopenblas64_.so` on x86-64 Linux
FIND_LIBRARY(OpenBLAS_LIB NAMES openblas64_ openblas PATHS ${Open_BLAS_LIB_SEARCH_PATHS})
IF(NOT OpenBLAS_LIB)
	FIND_FILE(OpenBLAS_LIB NAMES libopenblas.dll.a PATHS ${Open_BLAS_LIB_SEARCH_PATHS})
ENDIF()

SET(OpenBLAS_FOUND ON)

#    Check include files
IF(NOT OpenBLAS_INCLUDE_DIR)
    SET(OpenBLAS_FOUND OFF)
    MESSAGE(STATUS "Could not find OpenBLAS include. Turning OpenBLAS_FOUND off")
ENDIF()

#    Check libraries
IF(NOT OpenBLAS_LIB)
    SET(OpenBLAS_FOUND OFF)
    MESSAGE(STATUS "Could not find OpenBLAS lib. Turning OpenBLAS_FOUND off")
ENDIF()

IF (OpenBLAS_FOUND)
  IF (NOT OpenBLAS_FIND_QUIETLY)
    MESSAGE(STATUS "Found OpenBLAS libraries: ${OpenBLAS_LIB}")
    MESSAGE(STATUS "Found OpenBLAS include: ${OpenBLAS_INCLUDE_DIR}")
  ENDIF (NOT OpenBLAS_FIND_QUIETLY)
ELSE (OpenBLAS_FOUND)
  IF (OpenBLAS_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find OpenBLAS")
  ENDIF (OpenBLAS_FIND_REQUIRED)
ENDIF (OpenBLAS_FOUND)

MARK_AS_ADVANCED(
    OpenBLAS_INCLUDE_DIR
    OpenBLAS_LIB
    OpenBLAS
)

#   Check ILP64 data model for the case of Julia self-shipped `libopenblas64_.so`
SET(detect_interface64_src "
    #include <string.h>
    char* openblas_get_config64_(void)\;
    int main() {
        return strstr(openblas_get_config64_(), \"USE64BITINT\") == NULL\;
    }
")
FILE(WRITE "${CMAKE_CURRENT_BINARY_DIR}/detect_interface64.c" ${detect_interface64_src})
TRY_RUN(
    OpenBLAS_INTERFACE64 compile_detect_interface64
    "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/detect_interface64.c"
    LINK_LIBRARIES ${OpenBLAS_LIB}
)
IF(OpenBLAS_INTERFACE64 EQUAL 0)
    add_definitions(-DOPENBLAS_INTERFACE64=1)  # see julia/deps/cblas.h
ENDIF(OpenBLAS_INTERFACE64 EQUAL 0)
FILE(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/detect_interface64.c")
