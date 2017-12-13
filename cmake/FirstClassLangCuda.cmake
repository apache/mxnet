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

#this file is CUDA help function with CMAKE first class CUDA

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11"   SUPPORT_CXX11)
if(USE_CXX14_IF_AVAILABLE)
  check_cxx_compiler_flag("-std=c++14"   SUPPORT_CXX14)
endif()

################################################################################################
# Short command for cuDNN detection. Believe it soon will be a part of CUDA toolkit distribution.
# That's why not FindcuDNN.cmake file, but just the macro
# Usage:
#   detect_cuDNN()
function(detect_cuDNN)
  set(CUDNN_ROOT "" CACHE PATH "CUDNN root folder")

  find_path(CUDNN_INCLUDE cudnn.h
            PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT}
            DOC "Path to cuDNN include directory." )


  find_library(CUDNN_LIBRARY NAMES libcudnn.so cudnn.lib # libcudnn_static.a
                             PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDNN_INCLUDE}
                             DOC "Path to cuDNN library.")

  if(CUDNN_INCLUDE AND CUDNN_LIBRARY)
    set(HAVE_CUDNN  TRUE PARENT_SCOPE)
    set(CUDNN_FOUND TRUE PARENT_SCOPE)

    mark_as_advanced(CUDNN_INCLUDE CUDNN_LIBRARY CUDNN_ROOT)
    message(STATUS "Found cuDNN (include: ${CUDNN_INCLUDE}, library: ${CUDNN_LIBRARY})")
  endif()
endfunction()



################################################################################################
# A function for automatic detection of GPUs installed  (if autodetection is enabled)
# Usage:
#   mshadow_detect_installed_gpus(out_variable)
function(mshadow_detect_installed_gpus out_variable)
  if(NOT CUDA_gpu_detect_output)
    set(__cufile ${PROJECT_BINARY_DIR}/detect_cuda_archs.cu)

    file(WRITE ${__cufile} ""
      "#include <cstdio>\n"
      "int main()\n"
      "{\n"
      "  int count = 0;\n"
      "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
      "  if (count == 0) return -1;\n"
      "  for (int device = 0; device < count; ++device)\n"
      "  {\n"
      "    cudaDeviceProp prop;\n"
      "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
      "      std::printf(\"%d.%d \", prop.major, prop.minor);\n"
      "  }\n"
      "  return 0;\n"
      "}\n")
    enable_language(CUDA)

    try_run(__nvcc_res __compile_result ${PROJECT_BINARY_DIR} ${__cufile}
      COMPILE_OUTPUT_VARIABLE __compile_out
      RUN_OUTPUT_VARIABLE __nvcc_out)

    if(__nvcc_res EQUAL 0 AND __compile_result)
      # nvcc outputs text containing line breaks when building with MSVC.
      # The line below prevents CMake from inserting a variable with line
      # breaks in the cache
      string(REGEX MATCH "([1-9].[0-9])" __nvcc_out "${__nvcc_out}")
      string(REPLACE "2.1" "2.1(2.0)" __nvcc_out "${__nvcc_out}")
      set(CUDA_gpu_detect_output ${__nvcc_out})
    else()
      message(WARNING "Running GPU detection script with nvcc failed: ${__nvcc_out} ${__compile_out}")
    endif()
  endif()

  if(NOT CUDA_gpu_detect_output)
    message(WARNING "Automatic GPU detection failed. Building for all known architectures (${mxnet_known_gpu_archs}).")
    set(${out_variable} ${mxnet_known_gpu_archs} PARENT_SCOPE)
  else()
    set(${out_variable} ${CUDA_gpu_detect_output} PARENT_SCOPE)
  endif()
endfunction()


# This list will be used for CUDA_ARCH_NAME = All option
set(CUDA_KNOWN_GPU_ARCHITECTURES "Fermi" "Kepler" "Maxwell")

# This list will be used for CUDA_ARCH_NAME = Common option (enabled by default)
set(CUDA_COMMON_GPU_ARCHITECTURES "3.0" "3.5" "5.0")

if (CUDA_TOOLSET VERSION_GREATER "6.5")
  list(APPEND CUDA_KNOWN_GPU_ARCHITECTURES "Kepler+Tegra" "Kepler+Tesla" "Maxwell+Tegra")
  list(APPEND CUDA_COMMON_GPU_ARCHITECTURES "5.2" "3.7")
endif ()

if (CUDA_TOOLSET VERSION_GREATER "7.5")
  list(APPEND CUDA_KNOWN_GPU_ARCHITECTURES "Pascal")
  list(APPEND CUDA_COMMON_GPU_ARCHITECTURES "6.0" "6.1" "6.1+PTX")
else()
  list(APPEND CUDA_COMMON_GPU_ARCHITECTURES "5.2+PTX")
endif ()

################################################################################################
# Function for selecting GPU arch flags for nvcc based on CUDA_ARCH_NAME
# Usage:
#   mshadow_select_nvcc_arch_flags(out_variable)
function(mshadow_select_nvcc_arch_flags out_variable)

  set(CUDA_ARCH_LIST "Auto" CACHE STRING "Select target NVIDIA GPU achitecture.")
  set_property( CACHE CUDA_ARCH_LIST PROPERTY STRINGS "" "All" "Common" ${CUDA_KNOWN_GPU_ARCHITECTURES} )
  mark_as_advanced(CUDA_ARCH_NAME)


  if("X${CUDA_ARCH_LIST}" STREQUAL "X" )
    set(CUDA_ARCH_LIST "All")
  endif()

  set(cuda_arch_bin)
  set(cuda_arch_ptx)

  message(STATUS " CUDA_ARCH_LIST: ${CUDA_ARCH_LIST}")
  if("${CUDA_ARCH_LIST}" STREQUAL "All")
    set(CUDA_ARCH_LIST ${CUDA_KNOWN_GPU_ARCHITECTURES})
  elseif("${CUDA_ARCH_LIST}" STREQUAL "Common")
    set(CUDA_ARCH_LIST ${CUDA_COMMON_GPU_ARCHITECTURES})
  elseif("${CUDA_ARCH_LIST}" STREQUAL "Auto" OR "${CUDA_ARCH_LIST}" STREQUAL "")
    set(mxnet_known_gpu_archs ${CUDA_COMMON_GPU_ARCHITECTURES})
    mshadow_detect_installed_gpus(CUDA_ARCH_LIST)
    message(STATUS "Autodetected CUDA architecture(s): ${CUDA_ARCH_LIST}")
  endif()

  # Now process the list and look for names
  string(REGEX REPLACE "[ \t]+" ";" CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")
  list(REMOVE_DUPLICATES CUDA_ARCH_LIST)
  foreach(arch_name ${CUDA_ARCH_LIST})
    set(arch_bin)
    set(arch_ptx)
    set(add_ptx FALSE)
    # Check to see if we are compiling PTX
    if(arch_name MATCHES "(.*)\\+PTX$")
      set(add_ptx TRUE)
      set(arch_name ${CMAKE_MATCH_1})
    endif()
    if(arch_name MATCHES "^([0-9]\\.[0-9](\\([0-9]\\.[0-9]\\))?)$")
      set(arch_bin ${CMAKE_MATCH_1})
      set(arch_ptx ${arch_bin})
    else()
      # Look for it in our list of known architectures
      if(${arch_name} STREQUAL "Fermi")
        if (CUDA_TOOLSET VERSION_LESS "8.0")
          set(arch_bin 2.0 "2.1(2.0)")
        endif()
      elseif(${arch_name} STREQUAL "Kepler+Tegra")
        set(arch_bin 3.2)
      elseif(${arch_name} STREQUAL "Kepler+Tesla")
        set(arch_bin 3.7)
      elseif(${arch_name} STREQUAL "Kepler")
        set(arch_bin 3.0 3.5)
        set(arch_ptx 3.5)
      elseif(${arch_name} STREQUAL "Maxwell+Tegra")
        set(arch_bin 5.3)
      elseif(${arch_name} STREQUAL "Maxwell")
        set(arch_bin 5.0 5.2)
        set(arch_ptx 5.2)
      elseif(${arch_name} STREQUAL "Pascal")
        set(arch_bin 6.0 6.1)
        set(arch_ptx 6.1)
      else()
        message(SEND_ERROR "Unknown CUDA Architecture Name ${arch_name} in CUDA_SELECT_NVCC_ARCH_FLAGS")
      endif()
    endif()
    list(APPEND cuda_arch_bin ${arch_bin})
    if(add_ptx)
      if (NOT arch_ptx)
        set(arch_ptx ${arch_bin})
      endif()
      list(APPEND cuda_arch_ptx ${arch_ptx})
    endif()
  endforeach()

  # remove dots and convert to lists
  string(REGEX REPLACE "\\." "" cuda_arch_bin "${cuda_arch_bin}")
  string(REGEX REPLACE "\\." "" cuda_arch_ptx "${cuda_arch_ptx}")
  string(REGEX MATCHALL "[0-9()]+" cuda_arch_bin "${cuda_arch_bin}")
  string(REGEX MATCHALL "[0-9]+"   cuda_arch_ptx "${cuda_arch_ptx}")

  if(cuda_arch_bin)
    list(REMOVE_DUPLICATES cuda_arch_bin)
  endif()
  if(cuda_arch_ptx)
    list(REMOVE_DUPLICATES cuda_arch_ptx)
  endif()

  message(STATUS "cuda arch bin: ${cuda_arch_bin}")
  message(STATUS "cuda arch ptx: ${cuda_arch_ptx}")
  set(nvcc_flags "")
  set(nvcc_archs_readable "")

  # Tell NVCC to add binaries for the specified GPUs
  foreach(arch ${cuda_arch_bin})
    if(arch MATCHES "([0-9]+)\\(([0-9]+)\\)")
      # User explicitly specified ARCH for the concrete CODE
      list(APPEND nvcc_flags -gencode arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1})
      list(APPEND nvcc_archs_readable sm_${CMAKE_MATCH_1})
    else()
      # User didn't explicitly specify ARCH for the concrete CODE, we assume ARCH=CODE
      list(APPEND nvcc_flags -gencode arch=compute_${arch},code=sm_${arch})
      list(APPEND nvcc_archs_readable sm_${arch})
    endif()
  endforeach()

  # Tell NVCC to add PTX intermediate code for the specified architectures
  foreach(arch ${cuda_arch_ptx})
    list(APPEND nvcc_flags -gencode arch=compute_${arch},code=compute_${arch})
    list(APPEND nvcc_archs_readable compute_${arch})
  endforeach()

  if(NOT MSVC)
    if(SUPPORT_CXX14)
      list(APPEND nvcc_flags "-std=c++14")
    elseif(SUPPORT_CXX11)
      list(APPEND nvcc_flags "-std=c++11")
    endif()
  endif()

  string (REPLACE " " ";" CMAKE_CXX_FLAGS_STR "${CMAKE_CXX_FLAGS}")
  foreach(_flag ${CMAKE_CXX_FLAGS_STR})
    # Remove -std=c++XX flags
    if(NOT "${_flag}" MATCHES "-std=.+")
      # Remove link flags
      if(NOT "${_flag}" MATCHES "-Wl,.+")
        list(APPEND nvcc_flags "-Xcompiler ${_flag}")
      endif()
    endif()
  endforeach()

  string(REPLACE ";" " " nvcc_archs_readable "${nvcc_archs_readable}")
  set(${out_variable}          ${nvcc_flags}          PARENT_SCOPE)
  set(${out_variable}_readable ${nvcc_archs_readable} PARENT_SCOPE)
endfunction()

