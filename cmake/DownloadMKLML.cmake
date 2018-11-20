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

set(MKLDNN_RELEASE v0.17-rc)
set(MKLML_RELEASE_FILE_SUFFIX 2019.0.1.20180928)

if(MSVC)
  set(MKL_NAME "mklml_win_${MKLML_RELEASE_FILE_SUFFIX}")

  file(DOWNLOAD "https://github.com/intel/mkl-dnn/releases/download/${MKLDNN_RELEASE}/${MKL_NAME}.zip"
       "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}.zip"
       EXPECTED_MD5 "443e661bdfd32dbbc99b460b43afceee" SHOW_PROGRESS)
  file(DOWNLOAD "https://github.com/apache/incubator-mxnet/releases/download/utils/7z.exe"
       "${CMAKE_CURRENT_BINARY_DIR}/mklml/7z2.exe"
       EXPECTED_MD5 "E1CF766CF358F368EC97662D06EA5A4C" SHOW_PROGRESS)

  execute_process(COMMAND "${CMAKE_CURRENT_BINARY_DIR}/mklml/7z2.exe" "-o${CMAKE_CURRENT_BINARY_DIR}/mklml/" "-y")
  execute_process(COMMAND "${CMAKE_CURRENT_BINARY_DIR}/mklml/7z.exe"
                  "x" "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}.zip" "-o${CMAKE_CURRENT_BINARY_DIR}/mklml/" "-y")

  set(MKLROOT "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}")

  message(STATUS "Setting MKLROOT path to ${MKLROOT}")

  include_directories(${MKLROOT}/include)

elseif(APPLE)
  set(MKL_NAME "mklml_mac_${MKLML_RELEASE_FILE_SUFFIX}")

  file(DOWNLOAD "https://github.com/intel/mkl-dnn/releases/download/${MKLDNN_RELEASE}/${MKL_NAME}.tgz"
       "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}.tgz"
       EXPECTED_MD5 "95f887af332205b1d15b392260003952" SHOW_PROGRESS)
  execute_process(COMMAND "tar" "-xzf" "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}.tgz"
                  "-C" "${CMAKE_CURRENT_BINARY_DIR}/mklml/")

  set(MKLROOT "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}")

  message(STATUS "Setting MKLROOT path to ${MKLROOT}")
  include_directories(${MKLROOT}/include)

elseif(UNIX)
  set(MKL_NAME "mklml_lnx_${MKLML_RELEASE_FILE_SUFFIX}")

  file(DOWNLOAD "https://github.com/intel/mkl-dnn/releases/download/${MKLDNN_RELEASE}/${MKL_NAME}.tgz"
       "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}.tgz"
       EXPECTED_MD5 "a63abf155361322b9c03f8fc50f4f317" SHOW_PROGRESS)
  execute_process(COMMAND "tar" "-xzf" "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}.tgz"
                  "-C" "${CMAKE_CURRENT_BINARY_DIR}/mklml/")

  set(MKLROOT "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}")
  message(STATUS "Setting MKLROOT path to ${MKLROOT}")
  include_directories(${MKLROOT}/include)

else()
  message(FATAL_ERROR "Wrong platform")
endif()
