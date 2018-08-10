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

# This script will download MKLML

message(STATUS "Downloading MKLML...")

if(MSVC)
  set(MKL_NAME "mklml_win_2018.0.3.20180406")

  file(DOWNLOAD "https://github.com/intel/mkl-dnn/releases/download/v0.14/${MKL_NAME}.zip"
       "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}.zip"
       EXPECTED_MD5 "8DD73E7D3F19F004551809824C4E8970" SHOW_PROGRESS)
  file(DOWNLOAD "https://github.com/apache/incubator-mxnet/releases/download/utils/7z.exe"
       "${CMAKE_CURRENT_BINARY_DIR}/mklml/7z2.exe"
       EXPECTED_MD5 "E1CF766CF358F368EC97662D06EA5A4C" SHOW_PROGRESS)

  execute_process(COMMAND "${CMAKE_CURRENT_BINARY_DIR}/mklml/7z2.exe" "-o${CMAKE_CURRENT_BINARY_DIR}/mklml/" "-y")
  execute_process(COMMAND "${CMAKE_CURRENT_BINARY_DIR}/mklml/7z.exe"
                  "x" "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}.zip" "-o${CMAKE_CURRENT_BINARY_DIR}/mklml/" "-y")

  set(MKLROOT "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}")

  message(STATUS "Setting MKLROOT path to ${MKLROOT}")

elseif(APPLE)
  set(MKL_NAME "mklml_mac_2018.0.3.20180406")

  file(DOWNLOAD "https://github.com/intel/mkl-dnn/releases/download/v0.14/${MKL_NAME}.tgz"
       "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}.tgz"
       EXPECTED_MD5 "23a6f7fd04fb1fa6de0d52a2ec5a2a14" SHOW_PROGRESS)
  execute_process(COMMAND "tar" "-xzf" "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}.tgz"
                  "-C" "${CMAKE_CURRENT_BINARY_DIR}/mklml/")

  set(MKLROOT "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}")

  message(STATUS "Setting MKLROOT path to ${MKLROOT}")

elseif(UNIX)
  set(MKL_NAME "mklml_lnx_2018.0.3.20180406")

  file(DOWNLOAD "https://github.com/intel/mkl-dnn/releases/download/v0.14/${MKL_NAME}.tgz"
       "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}.tgz"
       EXPECTED_MD5 "DAF7EFC3C1C0036B447213004467A8AE" SHOW_PROGRESS)
  execute_process(COMMAND "tar" "-xzf" "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}.tgz"
                  "-C" "${CMAKE_CURRENT_BINARY_DIR}/mklml/")

  set(MKLROOT "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}")

  # The MXNet option USE_OPENMP is switching OpenMP for the MXNet project. MKLML will depend on it in any case.
  # Using -fopenmp with -liomp5 (linking to Intel OpenMP) leads to undefined behavior. The not needed
  # Intel OpenMP library needs to be removed that it couldn't be used accidentially. For example,
  # MKLDNN 3rdparty/mkldnn/cmake/MKL.cmake will discover it and use if found.
  # https://github.com/intel/mkl-dnn/blob/master/README.md
  set(REMOVE_INTEL_OPENMP ${USE_OPENMP})

  if(REMOVE_INTEL_OPENMP)
    message(STATUS "Intel OpenMP library will be removed")
  endif()

  if(REMOVE_INTEL_OPENMP)
    file(REMOVE ${MKLROOT}/lib/libiomp5.so)
    file(REMOVE ${MKLROOT}/lib/libmklml_intel.so)
    file(RENAME ${MKLROOT}/lib/libmklml_gnu.so ${MKLROOT}/lib/libmklml.so)
  else()
    file(REMOVE ${MKLROOT}/lib/libmklml_gnu.so)
    file(RENAME ${MKLROOT}/lib/libmklml_intel.so ${MKLROOT}/lib/libmklml.so)
  endif()

  message(STATUS "Setting MKLROOT path to ${MKLROOT}")

else()
  message(FATAL_ERROR "Wrong platform")
endif()
