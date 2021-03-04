#===============================================================================
# Copyright 2016-2020 Intel Corporation
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

# propagate EXAMPLE specific flags
append(CMAKE_C_FLAGS "${CMAKE_EXAMPLE_CCXX_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_EXAMPLE_CCXX_FLAGS}")

# propagate sanitizer flags
append(CMAKE_C_FLAGS "${CMAKE_CCXX_SANITIZER_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_SANITIZER_FLAGS}")

# propagate nowarn flags
append(CMAKE_C_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")

include_directories(${PROJECT_SOURCE_DIR}/include)

set_if(UNIX LIBM m)

file(GLOB sources *.cpp *.c)

foreach(src ${sources})
    get_filename_component(src_name ${src} NAME)
    string(REGEX REPLACE "[_\\.]" "-" example_name ${src_name})
    if(${example_name} MATCHES "^(cpu)")
        # Example name contains cpu-
        register_exe(mkldnn-compat-${example_name} ${src} "test" ${LIBM})
    else()
        register_exe(mkldnn-compat-${example_name} ${src} "" ${LIBM})

        # Adding test for CPU
        add_test("mkldnn-compat-cpu-${example_name}" "mkldnn-compat-${example_name}" cpu)
        maybe_configure_windows_test("mkldnn-compat-cpu-${example_name}" TEST)
    endif()
endforeach()

