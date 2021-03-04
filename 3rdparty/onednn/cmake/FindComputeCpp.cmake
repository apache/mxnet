#.rst:
# FindComputeCpp
#---------------
#
#   Copyright 2016-2018 Codeplay Software Ltd.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use these files except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

#########################
#  FindComputeCpp.cmake
#########################
#
#  Tools for finding and building with ComputeCpp.
#
#  User must define ComputeCpp_DIR pointing to the ComputeCpp
#  installation.
#
#  Latest version of this file can be found at:
#    https://github.com/codeplaysoftware/computecpp-sdk

include(FindPackageHandleStandardArgs)

# Check that a supported host compiler can be found
if(CMAKE_COMPILER_IS_GNUCXX)
    # Require at least gcc 4.8
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8)
      message(FATAL_ERROR
        "host compiler - gcc version must be at least 4.8")
    endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # Require at least clang 3.6
    if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 3.6)
      message(FATAL_ERROR
        "host compiler - clang version must be at least 3.6")
    endif()
endif()

set(COMPUTECPP_USER_FLAGS "" CACHE STRING "User flags for compute++")
mark_as_advanced(COMPUTECPP_USER_FLAGS)

set(COMPUTECPP_BITCODE "spir64" CACHE STRING
  "Bitcode type to use as SYCL target in compute++")
mark_as_advanced(COMPUTECPP_BITCODE)

find_package(OpenCL REQUIRED)

# Find ComputeCpp package

if(DEFINED ComputeCpp_DIR)
  set(computecpp_find_hint ${ComputeCpp_DIR})
elseif(DEFINED ENV{COMPUTECPP_DIR})
  set(computecpp_find_hint $ENV{COMPUTECPP_DIR})
elseif(DEFINED ENV{COMPUTECPPROOT})
  # oneDNN change: extend computecpp_find_hint
  set(computecpp_find_hint $ENV{COMPUTECPPROOT})
endif()

# Used for running executables on the host
set(computecpp_host_find_hint ${computecpp_find_hint})

if(CMAKE_CROSSCOMPILING)
  # ComputeCpp_HOST_DIR is used to find executables that are run on the host
  if(DEFINED ComputeCpp_HOST_DIR)
    set(computecpp_host_find_hint ${ComputeCpp_HOST_DIR})
  elseif(DEFINED ENV{COMPUTECPP_HOST_DIR})
    set(computecpp_host_find_hint $ENV{COMPUTECPP_HOST_DIR})
  endif()
endif()

find_program(ComputeCpp_DEVICE_COMPILER_EXECUTABLE compute++
  HINTS ${computecpp_host_find_hint}
  PATH_SUFFIXES bin)

find_program(ComputeCpp_INFO_EXECUTABLE computecpp_info
  HINTS ${computecpp_host_find_hint}
  PATH_SUFFIXES bin)

find_library(COMPUTECPP_RUNTIME_LIBRARY
  NAMES ComputeCpp ComputeCpp_vs2015
  HINTS ${computecpp_find_hint}
  PATH_SUFFIXES lib
  DOC "ComputeCpp Runtime Library")

find_library(COMPUTECPP_RUNTIME_LIBRARY_DEBUG
  NAMES ComputeCpp ComputeCpp_vs2015_d
  HINTS ${computecpp_find_hint}
  PATH_SUFFIXES lib
  DOC "ComputeCpp Debug Runtime Library")

find_path(ComputeCpp_INCLUDE_DIRS
  NAMES "CL/sycl.hpp"
  HINTS ${computecpp_find_hint}/include
  DOC "The ComputeCpp include directory")
get_filename_component(ComputeCpp_INCLUDE_DIRS ${ComputeCpp_INCLUDE_DIRS} ABSOLUTE)

get_filename_component(computecpp_canonical_root_dir "${ComputeCpp_INCLUDE_DIRS}/.." ABSOLUTE)
set(ComputeCpp_ROOT_DIR "${computecpp_canonical_root_dir}" CACHE PATH
    "The root of the ComputeCpp install")

execute_process(COMMAND ${ComputeCpp_INFO_EXECUTABLE} "--dump-version"
  OUTPUT_VARIABLE ComputeCpp_VERSION
  RESULT_VARIABLE ComputeCpp_INFO_EXECUTABLE_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT ComputeCpp_INFO_EXECUTABLE_RESULT EQUAL "0")
  message(FATAL_ERROR "Package version - Error obtaining version!")
endif()

execute_process(COMMAND ${ComputeCpp_INFO_EXECUTABLE} "--dump-is-supported"
  OUTPUT_VARIABLE COMPUTECPP_PLATFORM_IS_SUPPORTED
  RESULT_VARIABLE ComputeCpp_INFO_EXECUTABLE_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT ComputeCpp_INFO_EXECUTABLE_RESULT EQUAL "0")
  message(FATAL_ERROR "platform - Error checking platform support!")
else()
  mark_as_advanced(COMPUTECPP_PLATFORM_IS_SUPPORTED)
  if (COMPUTECPP_PLATFORM_IS_SUPPORTED)
    message(STATUS "platform - your system can support ComputeCpp")
  else()
    message(WARNING "platform - your system CANNOT support ComputeCpp")
  endif()
endif()

execute_process(COMMAND ${ComputeCpp_INFO_EXECUTABLE}
  "--dump-device-compiler-flags"
  OUTPUT_VARIABLE COMPUTECPP_DEVICE_COMPILER_FLAGS
  RESULT_VARIABLE ComputeCpp_INFO_EXECUTABLE_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS "-sycl-target ${COMPUTECPP_BITCODE}")

if(NOT ComputeCpp_INFO_EXECUTABLE_RESULT EQUAL "0")
  message(FATAL_ERROR "compute++ flags - Error obtaining compute++ flags!")
else()
  mark_as_advanced(COMPUTECPP_COMPILER_FLAGS)
endif()

set(ComputeCpp_LIBRARIES "${COMPUTECPP_RUNTIME_LIBRARY}")

find_package_handle_standard_args(ComputeCpp
  FOUND_VAR ComputeCpp_FOUND
  REQUIRED_VARS ComputeCpp_ROOT_DIR
                ComputeCpp_DEVICE_COMPILER_EXECUTABLE
                ComputeCpp_INFO_EXECUTABLE
                COMPUTECPP_RUNTIME_LIBRARY
                COMPUTECPP_RUNTIME_LIBRARY_DEBUG
                ComputeCpp_INCLUDE_DIRS
                ComputeCpp_LIBRARIES
  VERSION_VAR ComputeCpp_VERSION)
mark_as_advanced(ComputeCpp_ROOT_DIR
                 ComputeCpp_DEVICE_COMPILER_EXECUTABLE
                 ComputeCpp_INFO_EXECUTABLE
                 COMPUTECPP_RUNTIME_LIBRARY
                 COMPUTECPP_RUNTIME_LIBRARY_DEBUG
                 ComputeCpp_INCLUDE_DIRS
                 ComputeCpp_VERSION)

if(NOT ComputeCpp_FOUND)
  return()
endif()

if(CMAKE_CROSSCOMPILING)
  if(NOT SDK_DONT_USE_TOOLCHAIN)
    list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS --gcc-toolchain=${SDK_TOOLCHAIN_DIR})
  endif()
  list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS --sysroot=${SDK_SYSROOT_DIR})
  list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS -target ${SDK_TARGET_TRIPLE})
endif()

separate_arguments(COMPUTECPP_DEVICE_COMPILER_FLAGS)
list(REMOVE_ITEM COMPUTECPP_DEVICE_COMPILER_FLAGS "-emit-llvm")
message(STATUS "compute++ flags - ${COMPUTECPP_DEVICE_COMPILER_FLAGS}")

if(NOT TARGET OpenCL::OpenCL)
  add_library(OpenCL::OpenCL UNKNOWN IMPORTED)
  set_target_properties(OpenCL::OpenCL PROPERTIES
    IMPORTED_LOCATION             "${OpenCL_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${OpenCL_INCLUDE_DIRS}"
  )
endif()

if(NOT TARGET ComputeCpp::ComputeCpp)
  add_library(ComputeCpp::ComputeCpp UNKNOWN IMPORTED)
  set_target_properties(ComputeCpp::ComputeCpp PROPERTIES
    IMPORTED_LOCATION_DEBUG           "${COMPUTECPP_RUNTIME_LIBRARY_DEBUG}"
    IMPORTED_LOCATION_RELWITHDEBINFO  "${COMPUTECPP_RUNTIME_LIBRARY_DEBUG}"
    IMPORTED_LOCATION                 "${COMPUTECPP_RUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES     "${ComputeCpp_INCLUDE_DIRS}"
    IMPORTED_LINK_INTERFACE_LIBRARIES "OpenCL::OpenCL"
  )
endif()

# This property allows targets to specify that their sources should be
# compiled with the integration header included after the user's
# sources, not before (e.g. when an enum is used in a kernel name, this
# is not technically valid SYCL code but can work with ComputeCpp)
define_property(
  TARGET PROPERTY COMPUTECPP_INCLUDE_AFTER
  BRIEF_DOCS "Include integration header after user source"
  FULL_DOCS "Changes compiler arguments such that the source file is
  actually the integration header, and the .cpp file is included on
  the command line so that it is seen by the compiler first. Enables
  non-standards-conformant SYCL code to compile with ComputeCpp."
)
define_property(
  TARGET PROPERTY INTERFACE_COMPUTECPP_FLAGS
  BRIEF_DOCS "Interface compile flags to provide compute++"
  FULL_DOCS  "Set additional compile flags to pass to compute++ when compiling
  any target which links to this one."
)
define_property(
  SOURCE PROPERTY COMPUTECPP_SOURCE_FLAGS
  BRIEF_DOCS "Source file compile flags for compute++"
  FULL_DOCS  "Set additional compile flags for compiling the SYCL integration
  header for the given source file."
)

####################
#   __build_ir
####################
#
#  Adds a custom target for running compute++ and adding a dependency for the
#  resulting integration header.
#
#  TARGET : Name of the target.
#  SOURCE : Source file to be compiled.
#  COUNTER : Counter included in name of custom target. Different counter
#       values prevent duplicated names of custom target when source files with
#       the same name, but located in different directories, are used for the
#       same target.
#
function(__build_ir)
  set(options)
  set(one_value_args
    TARGET
    SOURCE
    COUNTER
  )
  set(multi_value_args)
  cmake_parse_arguments(SDK_BUILD_IR
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  get_filename_component(sourceFileName ${SDK_BUILD_IR_SOURCE} NAME)

  # Set the path to the integration header.
  # Intel MKL-DNN change: do not strip nested directories in target names
  # set(outputSyclFile ${CMAKE_CURRENT_BINARY_DIR}/${sourceFileName}.sycl)
  # set(depFileName ${CMAKE_CURRENT_BINARY_DIR}/${sourceFileName}.sycl.d)
  string(REPLACE "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}"
    outputSyclFile "${SDK_BUILD_IR_SOURCE}.sycl")
  string(REPLACE "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}"
    depFileName "${SDK_BUILD_IR_SOURCE}.sycl.d")

  get_filename_component(outputSyclDir ${outputSyclFile} PATH)
  file(MAKE_DIRECTORY ${outputSyclDir})

  get_target_property(include_directories
    ${SDK_BUILD_IR_TARGET} INCLUDE_DIRECTORIES)
  get_directory_property(compile_definitions
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMPILE_DEFINITIONS)

  string(REGEX REPLACE "([^;]+)" "-I\\1" generated_include_directories
    "${include_directories}")
  string(REGEX REPLACE "([^;]+)" "-D\\1" generated_compile_definitions
    "${compile_definitions}")

  # Obtain language standard of the file
  set(device_compiler_cxx_standard)
  get_target_property(targetCxxStandard ${SDK_BUILD_IR_TARGET} CXX_STANDARD)
  if (targetCxxStandard MATCHES 17)
    set(device_compiler_cxx_standard "-std=c++1z")
  elseif (targetCxxStandard MATCHES 14)
    set(device_compiler_cxx_standard "-std=c++14")
  elseif (targetCxxStandard MATCHES 11)
    set(device_compiler_cxx_standard "-std=c++11")
  elseif (targetCxxStandard MATCHES 98)
    message(FATAL_ERROR "SYCL applications cannot be compiled using C++98")
  else ()
    set(device_compiler_cxx_standard "")
  endif()

  get_property(source_compile_flags
    SOURCE ${SDK_BUILD_IR_SOURCE}
    PROPERTY COMPUTECPP_SOURCE_FLAGS
  )
  if(source_compile_flags)
    list(APPEND target_compile_flags ${source_compile_flags})
  endif()

  set(COMPUTECPP_DEVICE_COMPILER_FLAGS
    ${device_compiler_cxx_standard}
    ${COMPUTECPP_DEVICE_COMPILER_FLAGS}
    ${COMPUTECPP_USER_FLAGS}
    ${target_compile_flags}
  )
  separate_arguments(COMPUTECPP_DEVICE_COMPILER_FLAGS)

  set(ir_dependencies ${SDK_BUILD_IR_SOURCE})
  get_target_property(target_libraries ${SDK_BUILD_IR_TARGET} LINK_LIBRARIES)
  if(target_libraries)
    foreach(library ${target_libraries})
      # oneDNN change: skip link flags
      if(NOT ${library} MATCHES "^-.*")
        list(APPEND ir_dependencies ${library})
      endif()
    endforeach()
  endif()

  # Depfile support was only added in CMake 3.7
  # CMake throws an error if it is unsupported by the generator (i. e. not ninja)
  if((NOT CMAKE_VERSION VERSION_LESS 3.7.0) AND
          CMAKE_GENERATOR MATCHES "Ninja")
    file(RELATIVE_PATH relOutputFile ${CMAKE_BINARY_DIR} ${outputSyclFile})
    set(generate_depfile -MMD -MF ${depFileName} -MT ${relOutputFile})
    set(enable_depfile DEPFILE ${depFileName})
  endif()

  # Add custom command for running compute++
  add_custom_command(
    OUTPUT ${outputSyclFile}
    COMMAND ${ComputeCpp_DEVICE_COMPILER_EXECUTABLE}
            ${COMPUTECPP_DEVICE_COMPILER_FLAGS}
            ${device_compiler_includes}
            ${generated_include_directories}
            ${generated_compile_definitions}
            -o ${outputSyclFile}
            -c ${SDK_BUILD_IR_SOURCE}
            ${generate_depfile}
    DEPENDS ${ir_dependencies}
    IMPLICIT_DEPENDS CXX ${SDK_BUILD_IR_SOURCE}
    ${enable_depfile}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Building ComputeCpp integration header file ${outputSyclFile}")

  # Name: (user-defined name)_(source file)_(counter)_ih
  set(headerTargetName
    ${SDK_BUILD_IR_TARGET}_${sourceFileName}_${SDK_BUILD_IR_COUNTER}_ih)

  if(NOT MSVC)
    # Add a custom target for the generated integration header
    add_custom_target(${headerTargetName} DEPENDS ${outputSyclFile})
    add_dependencies(${SDK_BUILD_IR_TARGET} ${headerTargetName})
  endif()

  # This property can be set on a per-target basis to indicate that the
  # integration header should appear after the main source listing
  get_property(includeAfter TARGET ${SDK_BUILD_IR_TARGET}
      PROPERTY COMPUTECPP_INCLUDE_AFTER)

  if(includeAfter)
    # Change the source file to the integration header - e.g.
    # g++ -c source_file_name.cpp.sycl
    get_target_property(current_sources ${SDK_BUILD_IR_TARGET} SOURCES)
    # Remove absolute path to source file
    list(REMOVE_ITEM current_sources ${SDK_BUILD_IR_SOURCE})
    # Remove relative path to source file
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" ""
      rel_source_file ${SDK_BUILD_IR_SOURCE}
    )
    list(REMOVE_ITEM current_sources ${rel_source_file})
    # Add SYCL header to source list
    list(APPEND current_sources ${outputSyclFile})
    set_property(TARGET ${SDK_BUILD_IR_TARGET}
      PROPERTY SOURCES ${current_sources})
    # CMake/gcc don't know what language a .sycl file is, so tell them
    set_property(SOURCE ${outputSyclFile} PROPERTY LANGUAGE CXX)
    set(includedFile ${SDK_BUILD_IR_SOURCE})
    set(cppFile ${outputSyclFile})
  else()
    set_property(SOURCE ${outputSyclFile} PROPERTY HEADER_FILE_ONLY ON)
    set(includedFile ${outputSyclFile})
    set(cppFile ${SDK_BUILD_IR_SOURCE})
  endif()

  # Force inclusion of the integration header for the host compiler
  if(MSVC)
    # Group SYCL files inside Visual Studio
    source_group("SYCL" FILES ${outputSyclFile})

    if(includeAfter)
      # Allow the source file to be edited using Visual Studio.
      # It will be added as a header file so it won't be compiled.
      set_property(SOURCE ${SDK_BUILD_IR_SOURCE} PROPERTY HEADER_FILE_ONLY true)
    endif()

    # Add both source and the sycl files to the VS solution.
    target_sources(${SDK_BUILD_IR_TARGET} PUBLIC ${SDK_BUILD_IR_SOURCE} ${outputSyclFile})

    # NOTE: The Visual Studio generators parse compile flags differently,
    # hence the different argument syntax
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
      set(forceIncludeFlags "/FI\"${includedFile}\" /TP")
    else()
      set(forceIncludeFlags /FI ${includedFile} /TP)
    endif()
  else()
      set(forceIncludeFlags "-include ${includedFile} -x c++")
  endif()

  set_property(
    SOURCE ${cppFile}
    APPEND_STRING PROPERTY COMPILE_FLAGS "${forceIncludeFlags}"
  )

endfunction(__build_ir)

#######################
#  add_sycl_to_target
#######################
#
#  Adds a SYCL compilation custom command associated with an existing
#  target and sets a dependancy on that new command.
#
#  TARGET : Name of the target to add SYCL to.
#  SOURCES : Source files to be compiled for SYCL.
#
function(add_sycl_to_target)
  set(options)
  set(one_value_args
    TARGET
  )
  set(multi_value_args
    SOURCES
  )
  cmake_parse_arguments(SDK_ADD_SYCL
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  set(fileCounter 0)
  # Add custom target to run compute++ and generate the integration header
  foreach(sourceFile ${SDK_ADD_SYCL_SOURCES})
    if(NOT IS_ABSOLUTE ${sourceFile})
      set(sourceFile "${CMAKE_CURRENT_SOURCE_DIR}/${sourceFile}")
    endif()
    __build_ir(
      TARGET     ${SDK_ADD_SYCL_TARGET}
      SOURCE     ${sourceFile}
      COUNTER    ${fileCounter}
    )
    MATH(EXPR fileCounter "${fileCounter} + 1")
  endforeach()
  set_property(TARGET ${SDK_ADD_SYCL_TARGET}
    APPEND PROPERTY LINK_LIBRARIES ComputeCpp::ComputeCpp)
  set_property(TARGET ${SDK_ADD_SYCL_TARGET}
    APPEND PROPERTY INTERFACE_LINK_LIBRARIES ComputeCpp::ComputeCpp)
endfunction(add_sycl_to_target)
