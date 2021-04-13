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

set(CUTENSOR_ROOT "/usr/local/cuda" CACHE PATH "cuTensor root folder")

find_path(CUTENSOR_INCLUDE cutensor.h
        PATHS ${CUTENSOR_ROOT} $ENV{CUTENSOR_ROOT}
        DOC "Path to cuTensor include directory." )

find_library(CUTENSOR_LIBRARY NAMES libcutensor.so # libcutensor_static.a
        PATHS ${CUTENSOR_ROOT} $ENV{CUTENSOR_ROOT} ${CUTENSOR_INCLUDE}
        PATH_SUFFIXES lib lib/x64  cuda/lib cuda/lib64 lib/x64
        DOC "Path to cuTensor library.")

find_package_handle_standard_args(CUTENSOR DEFAULT_MSG CUTENSOR_LIBRARY CUTENSOR_INCLUDE)

mark_as_advanced(CUTENSOR_ROOT CUTENSOR_INCLUDE CUTENSOR_LIBRARY)
