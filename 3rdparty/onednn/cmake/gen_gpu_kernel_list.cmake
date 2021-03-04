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

# Parses kernel names from OpenCL files and updates KER_LIST_EXTERN and
# KER_LIST_ENTRIES variables with the parsed kernel names
function(parse_kernels ker_name ker_path)
    set(entries "${KER_LIST_ENTRIES}")

    file(READ ${ker_path} contents)
    string(REGEX MATCHALL
        "kernel[ \n]+void[ \n]+([a-z0-9_]+)"
        kernels ${contents})
    set(cur_ker_names)
    foreach(k ${kernels})
        string(REGEX REPLACE ".*void[ \n]+" "" k ${k})
        list(APPEND cur_ker_names ${k})
        list(FIND unique_ker_names ${k} index)
        if (${index} GREATER -1)
            message(WARNING "Kernel name is not unique: ${k}")
        endif()
        set(entries "${entries}\n        { \"${k}\", ${ker_name}_kernel },")
    endforeach()

    set(KER_LIST_EXTERN
        "${KER_LIST_EXTERN}\nextern const char *${ker_name}_kernel[];"
        PARENT_SCOPE)
    set(KER_LIST_ENTRIES "${entries}" PARENT_SCOPE)

    set(unique_ker_names "${unique_ker_names};${cur_ker_names}"
        PARENT_SCOPE)
endfunction()

function(gen_gpu_kernel_list ker_list_templ ker_list_src ker_sources)
    set(_sources "${SOURCES}")

    set(KER_LIST_EXTERN)
    set(KER_LIST_ENTRIES)

    set(unique_ker_names)

    foreach(ker_path ${ker_sources})
        get_filename_component(ker_name ${ker_path} NAME_WE)
        set(gen_file "${CMAKE_CURRENT_BINARY_DIR}/${ker_name}_kernel.cpp")
        # XXX: incremental build does not work with headers from kernel code
        add_custom_command(
            OUTPUT ${gen_file}
            COMMAND ${CMAKE_COMMAND}
                -DKER_FILE="${ker_path}"
                -DGEN_FILE="${gen_file}"
                -DKER_INC_DIR="${PROJECT_SOURCE_DIR}/src"
                -P ${PROJECT_SOURCE_DIR}/cmake/gen_gpu_kernel.cmake
            DEPENDS ${ker_path}
        )
        list(APPEND _sources "${gen_file}")
        parse_kernels(${ker_name} ${ker_path})
    endforeach()

    configure_file("${ker_list_templ}" "${ker_list_src}")
    set(SOURCES "${_sources}" PARENT_SCOPE)
endfunction()
