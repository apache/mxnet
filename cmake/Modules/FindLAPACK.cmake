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

file(TO_CMAKE_PATH "$ENV{LAPACK_HOME}" LAPACK_HOME)
file(TO_CMAKE_PATH "$ENV{LAPACK}" LAPACK_DIR)

SET(LAPACK_INCLUDE_SEARCH_PATHS
  /usr/include
  /usr/include/LAPACK
  /usr/include/LAPACK-base
  /usr/local/include
  /usr/local/include/LAPACK
  /usr/local/include/LAPACK-base
  /opt/LAPACK/include
  /usr/local/opt/LAPACK/include
  ${PROJECT_SOURCE_DIR}/3rdparty/LAPACK/include
  ${PROJECT_SOURCE_DIR}/thirdparty/LAPACK/include
  ${LAPACK_HOME}
  ${LAPACK_HOME}/include
)

SET(LAPACK_LIB_SEARCH_PATHS
        /lib/
        /lib/LAPACK-base
        /lib64/
        /usr/lib
        /usr/lib/LAPACK-base
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /opt/LAPACK/lib
        /usr/local/opt/LAPACK/lib
        ${PROJECT_SOURCE_DIR}/3rdparty/LAPACK/lib
        ${PROJECT_SOURCE_DIR}/thirdparty/LAPACK/lib
        ${LAPACK_DIR}
        ${LAPACK_DIR}/lib
        ${LAPACK_HOME}
        ${LAPACK_HOME}/lib
 )

FIND_PATH(LAPACK_INCLUDE_DIR NAMES cblas.h PATHS ${LAPACK_INCLUDE_SEARCH_PATHS})
FIND_LIBRARY(LAPACK_LIB NAMES lapack PATHS ${LAPACK_LIB_SEARCH_PATHS})
IF(NOT LAPACK_LIB)
	FIND_FILE(LAPACK_LIB NAMES liblapack.dll.a PATHS ${LAPACK_LIB_SEARCH_PATHS})
ENDIF()

SET(LAPACK_FOUND ON)

#    Check include files
IF(NOT LAPACK_INCLUDE_DIR)
    SET(LAPACK_FOUND OFF)
    MESSAGE(STATUS "Could not find LAPACK include. Turning LAPACK_FOUND off")
ENDIF()

#    Check libraries
IF(NOT LAPACK_LIB)
    SET(LAPACK_FOUND OFF)
    MESSAGE(STATUS "Could not find LAPACK lib. Turning LAPACK_FOUND off")
ENDIF()

IF (LAPACK_FOUND)
  IF (NOT LAPACK_FIND_QUIETLY)
    MESSAGE(STATUS "Found LAPACK libraries: ${LAPACK_LIB}")
    MESSAGE(STATUS "Found LAPACK include: ${LAPACK_INCLUDE_DIR}")
  ENDIF (NOT LAPACK_FIND_QUIETLY)
ELSE (LAPACK_FOUND)
  IF (LAPACK_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find LAPACK")
  ENDIF (LAPACK_FIND_REQUIRED)
ENDIF (LAPACK_FOUND)

MARK_AS_ADVANCED(
    LAPACK_INCLUDE_DIR
    LAPACK_LIB
    LAPACK
)

