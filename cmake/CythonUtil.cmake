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

################################################################################################
# Build a cython module
#
# Usage:
#   mxnet_external_build_cython(<python major version>)
#
function(mxnet_build_cython_module python_version)
  string(REGEX REPLACE "@" ";" PROP_MXNET_INCLUDE_DIRECTORIES "${MXNET_INCLUDE_DIRECTORIES}")
  string(REGEX REPLACE "@" ";" PROP_MXNET_INTERFACE_LINK_LIBRARIES "${MXNET_INTERFACE_LINK_LIBRARIES}")

  foreach(var ${PROP_MXNET_INCLUDE_DIRECTORIES})
    include_directories(${var})
  endforeach()

  unset(PYTHONLIBS_FOUND)
  unset(PYTHON_LIBRARIES)
  unset(PYTHON_INCLUDE_PATH)
  unset(PYTHON_INCLUDE_DIRS)
  unset(PYTHON_DEBUG_LIBRARIES)
  unset(PYTHONLIBS_VERSION_STRING)
  unset(PYTHONINTERP_FOUND)
  unset(PYTHON_EXECUTABLE)
  unset(PYTHON_VERSION_STRING)
  unset(PYTHON_VERSION_MAJOR)
  unset(PYTHON_VERSION_MINOR)
  unset(PYTHON_VERSION_PATCH)

  if(python_version EQUAL 2)
    set(Python_ADDITIONAL_VERSIONS 2.7 2.6 2.5 2.4 2.3 2.2 2.1 2.0)
  elseif(python_version EQUAL 3)
    set(Python_ADDITIONAL_VERSIONS 3.7 3.6 3.5 3.4 3.3 3.2 3.1 3.0)
  else()
    message(FATAL_ERROR "Nov alid python_version set (must be 2 or 3)")
    return()
  endif()

  set(python_libs_version ${python_version})
  include(${MXNET_ROOT_DIR}/cmake/UseCython.cmake)  # set from mxnet_external_build_cython

  if(NOT CYTHON${python_version}_FOUND)
    message(WARNING " Could not build cython target for Python ${python_version}")
    return()
  endif()

  set(CYTHON_SUBDIR ".")

  file(GLOB_RECURSE CYTHON_SOURCE "${MXNET_ROOT_DIR}/python/mxnet/cython/*.pyx")

  if(NOT MXNET_LIB_LOCATION)
    set(MXNET_LIB_LOCATION mxnet)
  endif()

  set(CYTHON_CXX_SOURCE "")
  foreach(cy_file ${CYTHON_SOURCE})
    set_source_files_properties(${cy_file} PROPERTIES CYTHON_IS_CXX TRUE)
    list(APPEND CYTHON_CXX_SOURCE ${cy_file_generated})
    get_filename_component(cy_module ${cy_file} NAME_WE)
    # We need cmake to have different target names for python 2 and 3
    set(cy_module_name ${cy_module})
    cython_add_module(${cy_module_name} "${CYTHON_SUBDIR}" ${cy_file})
    set_target_properties(${cy_module_name}
      PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CYTHON_SUBDIR}/"
      INTERFACE_LINK_LIBRARIES "${PROP_MXNET_INTERFACE_LINK_LIBRARIES}"
      )
    target_link_libraries(${cy_module_name} ${MXNET_LIB_LOCATION})
  endforeach()
endfunction()


################################################################################################
# Spawn external CMakeLists.txt in order to build a particular cython/python version
#
# The spawn approach is because we need to detect and build with both python version 2 and 3
# This is not osmething that a single process of cmake can deal with, so we launch
# a cmake process for the cython build 2, then for 3, passing it our relevant config
#
# Usage:
#   mxnet_external_build_cython(<python major version>)
#
function(mxnet_external_build_cython python_major_version)
  set(PMV ${python_major_version})

  if(CYTHON_WITHOUT_MXNET_TARGET)
    set(CYTHON_DEPS "")
    set(CYTHON_MXNET_LIB_LOCATION "")
  else()
    set(CYTHON_DEPS mxnet)
    set(CYTHON_MXNET_LIB_LOCATION $<TARGET_LINKER_FILE:mxnet>)
  endif()

  file(GLOB_RECURSE CYTHON_SOURCE "python/mxnet/cython/*.pyx")

  get_cmake_property(CACHE_VARS CACHE_VARIABLES)
  foreach(_cache_var ${CACHE_VARS})
    #message(STATUS "${_cache_var}=${${_cache_var}}")
    get_property(CACHE_VAR_HELPSTRING CACHE ${_cache_var} PROPERTY HELPSTRING)
    if(NOT _cache_var MATCHES "CMAKE_EXTRA_GENERATOR_.+"
      AND NOT _cache_var MATCHES "FIND_PACKAGE_MESSAGE_.+"
      )
      if(_cache_var MATCHES "USE_.+"
        OR _cache_var MATCHES "CMAKE_MODULE_.+"
        OR CACHE_VAR_HELPSTRING STREQUAL "No help, variable specified on the command line."
        )
        get_property(_cache_var_type CACHE ${_cache_var} PROPERTY TYPE)
        if(_cache_var_type STREQUAL "UNINITIALIZED")
          set(_cache_var_type)
        else()
          set(_cache_var_type :${_cache_var_type})
        endif()
        set(CMAKE_ARGS "${CMAKE_ARGS} -D${_cache_var}${_cache_var_type}=\"${${_cache_var}}\"")
      endif()
    endif()
  endforeach()

  get_property(PROP_MXNET_INCLUDE_DIRECTORIES TARGET mxnet PROPERTY INCLUDE_DIRECTORIES)
  string(REGEX REPLACE "\;" "@" MXNET_INCLUDE_DIRECTORIES "${PROP_MXNET_INCLUDE_DIRECTORIES}")

  get_property(PROP_MXNET_INTERFACE_LINK_LIBRARIES TARGET mxnet PROPERTY INTERFACE_LINK_LIBRARIES)
  string(REGEX REPLACE "\;" "@" MXNET_INTERFACE_LINK_LIBRARIES "${PROP_MXNET_INTERFACE_LINK_LIBRARIES}")

  set(CYTHON_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/cython/cy${PMV})

  set(_config_cleanup_files "")
  set(_config_cleanup_files_ex "")

  foreach(_file ${CYTHON_SOURCE})
    get_filename_component(_cy_module ${_file} NAME_WE)
    list(APPEND _config_cleanup_files  "${CYTHON_BINARY_DIR}/${_cy_module}.so")
    list(APPEND _config_cleanup_files  "${CYTHON_BINARY_DIR}/${_cy_module}.cxx")
    list(APPEND _config_cleanup_files  "${CYTHON_BINARY_DIR}/${_cy_module}.c")
  endforeach()

  # Clear some cmake-generated files
  list(APPEND _config_cleanup_files_ex "${CYTHON_BINARY_DIR}/CMakeCache.txt")
  list(APPEND _config_cleanup_files_ex "${CYTHON_BINARY_DIR}/Makefile")

  # Get current cleanup files
  get_directory_property(CLEANUP_FILES ADDITIONAL_MAKE_CLEAN_FILES)
  list(APPEND CLEANUP_FILES ${_config_cleanup_files})
  list(APPEND CLEANUP_FILES ${_config_cleanup_files_ex})
  # Set new list of cleanup files
  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "${CLEANUP_FILES}")

  add_custom_target(build-time-make-cython-directory${PMV} ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CYTHON_BINARY_DIR})

  add_custom_target(${PROJECT_NAME}_ConfigCython${PMV} ALL
    ${CMAKE_COMMAND}
    ${CMAKE_ARGS}
    -G "${CMAKE_GENERATOR}"
    -DCMAKE_MODULE_PATH="${CMAKE_MODULE_PATH}"
    -DMXNET_INCLUDE_DIRECTORIES="${MXNET_INCLUDE_DIRECTORIES}"
    -DMXNET_INTERFACE_LINK_LIBRARIES="${MXNET_INTERFACE_LINK_LIBRARIES}"
    -DMXNET_LIB_LOCATION=${CYTHON_MXNET_LIB_LOCATION}
    -DMXNET_ROOT_DIR=${CMAKE_CURRENT_SOURCE_DIR}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cython/cy${PMV}
    WORKING_DIRECTORY ${CYTHON_BINARY_DIR}
    DEPENDS ${CYTHON_DEPS} build-time-make-cython-directory${PMV}
    )

  add_custom_target(${PROJECT_NAME}_BuildCython${PMV} ALL
    ${CMAKE_COMMAND}
    --build ${CYTHON_BINARY_DIR}
    --config ${CMAKE_BUILD_TYPE}
    WORKING_DIRECTORY ${CYTHON_BINARY_DIR}
    DEPENDS ${PROJECT_NAME}_ConfigCython${PMV}
    )
endfunction()
