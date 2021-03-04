#===============================================================================
# Copyright 2017-2020 Intel Corporation
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

file(GLOB_RECURSE HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/*.hpp
    )
file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )
list(APPEND SOURCES ${TEST_THREAD})
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

if(BENCHDNN_USE_RDPMC)
    add_definitions(-DBENCHDNN_USE_RDPMC)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    append_if(WIN32 CMAKE_CXX_FLAGS "-Qprec-div -Qprec-sqrt")
    append_if(UNIX  CMAKE_CXX_FLAGS "-prec-div -prec-sqrt -fp-model precise")
endif()

if(UNIX AND NOT APPLE)
    find_library(LIBRT rt)
endif()
register_exe(benchdnn "${SOURCES}" "" "${LIBRT}")

file(COPY inputs DESTINATION .)

if(WIN32 AND (NOT DNNL_BUILD_FOR_CI))
    string(REPLACE  ";" "\;" PATH "${CTESTCONFIG_PATH};$ENV{PATH}")
    configure_file(
        "${PROJECT_SOURCE_DIR}/cmake/run_with_env.bat.in"
        "${PROJECT_BINARY_DIR}/run_with_env.bat"
    )
endif()

function(register_benchdnn_test engine driver test_file)
    if (ARGC GREATER 3)
        message(ERROR "Incorrect use of function")
    endif()

    string(REPLACE "test_" "test_benchdnn_" target_name ${test_file})
    set(cmd "-v1 --engine=${engine} --${driver} --batch=${test_file}")
    set(benchdnn_target ${target_name}_${engine})

    if(DNNL_BUILD_FOR_CI)
        string(REPLACE " " ";" cmd "benchdnn ${cmd}")
        add_test(${benchdnn_target} ${cmd})
    else()
        string(REPLACE " " ";" cmd "$<TARGET_FILE:benchdnn> ${cmd}")

        if(WIN32)
            set(cmd "cmd;/c;${PROJECT_BINARY_DIR}/run_with_env.bat;${cmd}")
            set(ARGV2 "cmd;/c;${PROJECT_BINARY_DIR}/run_with_env.bat;${ARGV2}")
        endif()

        add_custom_target(${benchdnn_target}
            COMMAND ${cmd}
            DEPENDS benchdnn
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )

        set_target_properties(${benchdnn_target} PROPERTIES
            EXCLUDE_FROM_DEFAULT_BUILD TRUE)
        maybe_configure_windows_test(${benchdnn_target} TARGET)

        # Create non-suffixed target for compatibility
        if (engine STREQUAL "cpu")
            add_custom_target(${target_name} DEPENDS ${benchdnn_target})
            maybe_configure_windows_test(${target_name} TARGET)
        endif()
    endif()
endfunction()

function(register_all_tests engine driver test_files)
    foreach(test_file ${test_files})
        register_benchdnn_test(${engine} ${driver} ${test_file})
    endforeach()
endfunction()

# The following section is responsible to register test_benchdnn_ targets
# depending on DNNL_TEST_SET value.
set(has_gpu false)
if(NOT DNNL_GPU_RUNTIME STREQUAL "NONE")
    set(has_gpu true)
endif()

# Very sad CMake 2.8.11 does not support continue() command.
file(GLOB all_drivers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/inputs inputs/*)
foreach(driver ${all_drivers})
    set(driver_dir ${CMAKE_CURRENT_SOURCE_DIR}/inputs/${driver})
    # Collect input files in groups
    file(GLOB test_files_large_cpu RELATIVE ${driver_dir}
            inputs/${driver}/test_*_large$)
    file(GLOB test_files_gpu_ci RELATIVE ${driver_dir}
            inputs/${driver}/test_*_gpu_ci)
    file(GLOB test_files_gpu RELATIVE ${driver_dir} inputs/${driver}/test_*_gpu)
    file(GLOB test_files_ci RELATIVE ${driver_dir} inputs/${driver}/test_*_ci)
    file(GLOB test_files_cpu RELATIVE ${driver_dir} inputs/${driver}/test_*)

    # Register ci input files for ci only
    if(DNNL_TEST_SET EQUAL DNNL_TEST_SET_CI)
        # gpu_ci files may happen if cpu coverage can not be used on gpu
        # Filter out gpu_ci inputs from ci
        foreach(test_file ${test_files_gpu_ci})
            string(REPLACE "${test_file}" "" test_files_ci "${test_files_ci}")
        endforeach()

        # use gpu_ci if not empty
        if(has_gpu)
            if(test_files_gpu_ci)
                register_all_tests(gpu "${driver}" "${test_files_gpu_ci}")
            else()
                register_all_tests(gpu "${driver}" "${test_files_ci}")
            endif()
        endif()
        register_all_tests(cpu "${driver}" "${test_files_ci}")
    endif()

    # Register nightly input files
    if(DNNL_TEST_SET GREATER DNNL_TEST_SET_CI)
        ## Filter out gpu, large cpu and invalid inputs from cpu
        foreach(test_file ${test_files_large_cpu} ${test_files_gpu_ci}
                ${test_files_gpu} ${test_files_ci} "test_conv_all;")
            string(REPLACE "${test_file}" "" test_files_cpu "${test_files_cpu}")
        endforeach()

        if(NOT DNNL_USE_CLANG_SANITIZER)
            register_all_tests(cpu "${driver}" "${test_files_large_cpu}")
        endif()
        if(has_gpu)
            register_all_tests(gpu "${driver}" "${test_files_gpu}")
        endif()
        register_all_tests(cpu "${driver}" "${test_files_cpu}")
    endif()
endforeach()
