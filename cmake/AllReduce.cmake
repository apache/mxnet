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

# Finds Google Protocol Buffers library and compilers and extends
# the standard cmake script with version and python generation support

find_package( Protobuf REQUIRED )
include_directories(SYSTEM ${PROTOBUF_INCLUDE_DIR})

# Make sure protoc version is greater than 3.0.0
if(EXISTS ${PROTOBUF_PROTOC_EXECUTABLE})
  message(STATUS "Found PROTOBUF Compiler: ${PROTOBUF_PROTOC_EXECUTABLE}")
else()
  message(FATAL_ERROR "Could not find PROTOBUF Compiler")
endif()

set(PROTOBUF_GENERATE_CPP_APPEND_PATH TRUE)

################################################################################################
# Usage:
#   allreduce_protobuf_generate_cpp(<output_dir> <srcs_var> <hdrs_var> <work_path> <proto_files>)
################################################################################################
function(allreduce_protobuf_generate_cpp output_dir srcs_var hdrs_var work_path proto_path)
  if(NOT ARGN)
    message(SEND_ERROR "Error: allreduce_protobuf_generate_cpp() called without any proto files")
    return()
  endif()

  if(PROTOBUF_GENERATE_CPP_APPEND_PATH)
    # Create an include path for each file specified
    foreach(fil ${ARGN})
      get_filename_component(abs_fil ${fil} ABSOLUTE)
      get_filename_component(abs_path ${abs_fil} PATH)
      list(FIND _protoc_include ${abs_path} _contains_already)
      if(${_contains_already} EQUAL -1)
        list(APPEND _protoc_include -I ${abs_path})
      endif()
    endforeach()
  else()
    set(_protoc_include -I ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  set(${srcs_var})
  set(${hdrs_var})
  foreach(fil ${ARGN})
    get_filename_component(abs_fil ${fil} ABSOLUTE)
    get_filename_component(fil_we ${fil} NAME_WE)
	  string(REPLACE ${work_path}/ "" o_fil ${abs_fil})
	  string(REPLACE "${fil_we}.proto" "" o_fil_path ${o_fil})

    list(APPEND ${srcs_var} "${o_fil_path}/${fil_we}.pb.cc")
    list(APPEND ${hdrs_var} "${o_fil_path}/${fil_we}.pb.h")

    add_custom_command(
      OUTPUT "${o_fil_path}/${fil_we}.pb.cc"
             "${o_fil_path}/${fil_we}.pb.h"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${output_dir}"
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --cpp_out ${output_dir} ${o_fil} --proto_path ${proto_path}
      DEPENDS ${abs_fil}
	    WORKING_DIRECTORY ${work_path}
      COMMENT "Running C++ protocol buffer compiler on ${o_fil}" VERBATIM )
  endforeach()

  set_source_files_properties(${${srcs_var}} ${${hdrs_var}} PROPERTIES GENERATED TRUE)
  set(${srcs_var} ${${srcs_var}} PARENT_SCOPE)
  set(${hdrs_var} ${${hdrs_var}} PARENT_SCOPE)
endfunction()
