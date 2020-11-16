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

include(FindPackageHandleStandardArgs)

set(CUDNN_ROOT "/usr/local/cuda/include" CACHE PATH "cuDNN root folder")

find_path(CUDNN_INCLUDE cudnn.h
  PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT}
  DOC "Path to cuDNN include directory." )

find_library(CUDNN_LIBRARY NAMES libcudnn.so cudnn.lib # libcudnn_static.a
  PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDNN_INCLUDE}
  PATH_SUFFIXES lib lib/x64  cuda/lib cuda/lib64 lib/x64
  DOC "Path to cuDNN library.")

find_package_handle_standard_args(CUDNN DEFAULT_MSG CUDNN_LIBRARY CUDNN_INCLUDE)

mark_as_advanced(CUDNN_ROOT CUDNN_INCLUDE CUDNN_LIBRARY)
