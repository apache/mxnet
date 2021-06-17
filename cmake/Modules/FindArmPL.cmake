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

# Find the ARM Performance Libraries
#
# The following are set after configuration is done:
#  ArmPL_FOUND
#  ArmPL_INCLUDE_DIR
#  ArmPL_LIBRARIES

file(TO_CMAKE_PATH "$ENV{ArmPL_HOME}" ArmPL_HOME)

SET(ArmPL_INCLUDE_SEARCH_PATHS
        /opt/arm/armpl_21.0_gcc-8.2/include_lp64_mp
        ${ArmPL_HOME}/include_lp64_mp
)

SET(ArmPL_LIB_SEARCH_PATHS
	/opt/arm/armpl_21.0_gcc-8.2/lib
        ${ArmPL_HOME}/lib
)

FIND_PATH(ArmPL_INCLUDE_DIR NAMES armpl.h PATHS ${ArmPL_INCLUDE_SEARCH_PATHS})
FIND_LIBRARY(ArmPL_LIB NAMES armpl_lp64_mp PATHS ${ArmPL_LIB_SEARCH_PATHS})
FIND_LIBRARY(MATH_LIB NAMES amath PATHS ${ArmPL_LIB_SEARCH_PATHS})
FIND_LIBRARY(STRING_LIB NAMES astring PATHS ${ArmPL_LIB_SEARCH_PATHS})
SET(ArmPL_LIBRARIES
	${ArmPL_LIB}
	${MATH_LIB}
	${STRING_LIB}
	/usr/local/gcc-8.5.0/lib64/libgfortran.so
	/usr/lib/aarch64-linux-gnu/libm.so
)

SET(ArmPL_FOUND ON)

#    Check include files
IF(NOT ArmPL_INCLUDE_DIR)
    SET(ArmPL_FOUND OFF)
    MESSAGE(STATUS "Could not find ArmPL include. Turning ArmPL_FOUND off")
ENDIF()

#    Check libraries
IF(NOT ArmPL_LIBRARIES)
    SET(ArmPL_FOUND OFF)
    MESSAGE(STATUS "Could not find ArmPL lib. Turning ArmPL_FOUND off")
ENDIF()

IF (ArmPL_FOUND)
  IF (NOT ArmPL_FIND_QUIETLY)
	  MESSAGE(STATUS "Found ArmPL libraries: ${ArmPL_LIBRARIES}")
    MESSAGE(STATUS "Found ArmPL include: ${ArmPL_INCLUDE_DIR}")
  ENDIF (NOT ArmPL_FIND_QUIETLY)
ELSE (ArmPL_FOUND)
  IF (ArmPL_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find ArmPL")
  ENDIF (ArmPL_FIND_REQUIRED)
ENDIF (ArmPL_FOUND)

MARK_AS_ADVANCED(
    ArmPL_FOUND
    ArmPL_INCLUDE_DIR
    ArmPL_LIBRARIES
)

