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

# Find the Cython compiler.
#
# This code sets the following variables:
#
#  CYTHON_EXECUTABLE
#
# See also UseCython.cmake

#=============================================================================
# Copyright 2011 Kitware, Inc.
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
#=============================================================================

# Use the Cython executable that lives next to the Python executable
# if it is a local installation.

if(PACKAGE_FIND_VERSION_MAJOR EQUAL 3)
  set(CYTHON_EXE_NAMES cython3 cython.bat cython)
  message(STATUS " Looking for Cython version 3")
else()
  set(CYTHON_EXE_NAMES cython.bat cython cython3)
endif()

if(PYTHONINTERP_FOUND)
  get_filename_component( _python_path ${PYTHON_EXECUTABLE} PATH )
  find_program(CYTHON_EXECUTABLE
    NAMES ${CYTHON_EXE_NAMES}
    HINTS ${_python_path}
    )
else()
  find_program(CYTHON_EXECUTABLE NAMES ${CYTHON_EXE_NAMES})
endif()

include( FindPackageHandleStandardArgs )
find_package_handle_standard_args(Cython DEFAULT_MSG CYTHON_EXECUTABLE)

if(CYTHON_FOUND)
  message(STATUS "Found Cython (executable: ${CYTHON_EXECUTABLE})")
  mark_as_advanced( CYTHON_EXECUTABLE )
endif()

