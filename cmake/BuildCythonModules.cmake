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

function(add_cython_modules python_version)
  unset(PYTHON_EXECUTABLE CACHE)
  set(PYTHONINTERP_FOUND FALSE)
  find_package(PythonInterp ${python_version} EXACT)
  if(PYTHONINTERP_FOUND)
    find_program(CYTHON_EXECUTABLE NAMES cython)
    if(CYTHON_EXECUTABLE)
      add_custom_command(COMMAND ${CMAKE_COMMAND} POST_BUILD
                          -E env MXNET_LIBRARY_PATH=${CMAKE_BINARY_DIR}/libmxnet.so
                          ${PYTHON_EXECUTABLE} setup.py build_ext --inplace --with-cython
                          TARGET mxnet
                          WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/python")
      message("-- Cython modules for python${python_version} will be built")
      set(PYTHON${python_version}_FOUND 1 PARENT_SCOPE)
    else()
      message(FATAL_ERROR "-- Cython not found")
    endif()
  else()
    set(PYTHON${python_version}_FOUND 0 PARENT_SCOPE)
  endif()
endfunction()
