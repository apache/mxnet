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

# Find the Apple Accelerate framework
#
# The following are set after configuration is done:
#  Accelerate_FOUND
#  Accelerate_INCLUDE_DIRS
#  Accelerate_LIBRARIES

file(TO_CMAKE_PATH "$ENV{Accelerate_HOME}" Accelerate_HOME)
set(Accelerate_INCLUDE_SEARCH_PATHS
  /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/Current
  /System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current
  ${Accelerate_HOME}
)

find_path(Accelerate_CBLAS_INCLUDE_DIR NAMES cblas.h PATHS ${Accelerate_INCLUDE_SEARCH_PATHS} PATH_SUFFIXES Headers)

set(LOOKED_FOR
    Accelerate_CBLAS_INCLUDE_DIR
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Accelerate DEFAULT_MSG ${LOOKED_FOR})

if(Accelerate_FOUND)
  set(Accelerate_INCLUDE_DIR ${Accelerate_CBLAS_INCLUDE_DIR})
  set(Accelerate_LIBRARIES "-framework Accelerate")
  mark_as_advanced(${LOOKED_FOR})

  message(STATUS "Found Accelerate (include: ${Accelerate_CBLAS_INCLUDE_DIR}, library: ${Accelerate_BLAS_LIBRARY})")
endif(Accelerate_FOUND)

