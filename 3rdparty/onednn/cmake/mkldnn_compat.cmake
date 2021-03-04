#===============================================================================
# Copyright 2019-2020 Intel Corporation
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

# Provides compatibility with Intel MKL-DNN build options
#===============================================================================

# Sets if DNNL_* var is unset, copy the value from corresponding MKLDNN_* var
macro(mkldnn_compat_var dnnl_var mkldnn_var props)
    if (DEFINED ${mkldnn_var} AND NOT DEFINED ${dnnl_var})
        if ("${props}" STREQUAL "CACHE STRING")
            set(${dnnl_var} "${${mkldnn_var}}" CACHE STRING "" FORCE)
        elseif ("${props}" STREQUAL "CACHE BOOL")
            set(${dnnl_var} "${${mkldnn_var}}" CACHE BOOL "" FORCE)
        else()
            set(${dnnl_var} "${${mkldnn_var}}")
        endif()
        message(STATUS "Intel MKL-DNN compat: "
            "set ${dnnl_var} to ${mkldnn_var} with value `${${dnnl_var}}`")
    endif()
endmacro()

set(COMPAT_CACHE_BOOL_VARS
    "VERBOSE"
    "ENABLE_CONCURRENT_EXEC"
    "BUILD_EXAMPLES"
    "BUILD_TESTS"
    "BUILD_FOR_CI"
    "WERROR"
    "ENABLE_JIT_PROFILING"
    )

set(COMPAT_CACHE_STRING_VARS
    "LIBRARY_TYPE"
    "INSTALL_MODE"
    "ARCH_OPT_FLAGS"
    "CPU_RUNTIME"
    "GPU_RUNTIME"
    "USE_CLANG_SANITIZER"
    )

# Map MKLDNN_ to DNNL_ options

foreach (var ${COMPAT_CACHE_BOOL_VARS})
    mkldnn_compat_var("DNNL_${var}" "MKLDNN_${var}" "CACHE BOOL")
endforeach()
mkldnn_compat_var(_DNNL_USE_MKL _MKLDNN_USE_MKL "CACHE BOOL")

foreach (var ${COMPAT_CACHE_STRING_VARS})
    mkldnn_compat_var("DNNL_${var}" "MKLDNN_${var}" "CACHE STRING")
endforeach()

# Handle legacy options: MKLDNN_THREADING and MKLDNN_GPU_BACKEND.

if(MKLDNN_THREADING)
    set(DNNL_CPU_RUNTIME "${DNNL_THREADING}" CACHE STRING "" FORCE)
    message(STATUS "Using the obsolete way to specify the CPU runtime. "
        "Use DNNL_CPU_RUNTIME=${DNNL_CPU_RUNTIME} instead.")
endif()

if(MKLDNN_GPU_BACKEND)
    if (MKLDNN_GPU_BACKEND STREQUAL "OPENCL")
        set(MKLDNN_GPU_BACKEND "OCL" CACHE STRING "" FORCE)
        message(STATUS "Using the obsolete way to specify the OpenCL runtime. "
            "Use DNNL_GPU_RUNTIME=OCL instead.")
    endif()
    set(DNNL_GPU_RUNTIME "${MKLDNN_GPU_BACKEND}" CACHE STRING "" FORCE)
    message(STATUS "Using the obsolete way to specify the GPU runtime. "
        "Use DNNL_GPU_RUNTME=${DNNL_GPU_RUNTIME} instead.")
endif()

if (MKLDNN_GPU_RUNTIME STREQUAL "OPENCL")
    set(DNNL_GPU_RUNTIME "OCL" CACHE STRING "" FORCE)
    message(STATUS "Using the obsolete way to specify the OpenCL runtime. "
        "Use DNNL_GPU_RUNTIME=OCL instead.")
endif()
