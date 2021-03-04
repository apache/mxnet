#===============================================================================
# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

# FindLevelZero

# Result Variables
#
# This module defines the following variables:
#
#   LevelZero_FOUND          - True if Level Zero was found
#   LevelZero_INCLUDE_DIRS   - include directories for Level Zero
#   LevelZero_LIBRARIES      - link against this library to use Level Zero
#
# The module will also define two cache variables:
#
#   LevelZero_INCLUDE_DIR    - the Level Zero include directory
#   LevelZero_LIBRARY        - the path to the Level Zero library
#

find_path(LevelZero_INCLUDE_DIR
    NAMES
        level_zero/ze_api.h
    PATH_SUFFIXES
        include)

find_library(LevelZero_LIBRARY
    NAMES level_zero
    PATHS
    PATH_SUFFIXES
        lib/x64
        lib
        lib64)

set(LevelZero_LIBRARIES ${LevelZero_LIBRARY})
set(LevelZero_INCLUDE_DIRS ${LevelZero_INCLUDE_DIR})

# XXX: Library is optional - the package is reported as found even without it.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    LevelZero
    FOUND_VAR LevelZero_FOUND
    REQUIRED_VARS LevelZero_INCLUDE_DIR)

mark_as_advanced(
    LevelZero_INCLUDE_DIR
    LevelZero_LIBRARY)

if(LevelZero_FOUND AND NOT TARGET LevelZero::LevelZero)
    add_library(LevelZero::LevelZero UNKNOWN IMPORTED)
    set_target_properties(LevelZero::LevelZero PROPERTIES
        IMPORTED_LOCATION "${LevelZero_LIBRARY}")
    set_target_properties(LevelZero::LevelZero PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${LevelZero_INCLUDE_DIRS}")
endif()
