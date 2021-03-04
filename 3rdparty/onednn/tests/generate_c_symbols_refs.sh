#!/bin/sh
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

dnnl_root="$1"
output="$2"
shift 2

{
    echo '#include "oneapi/dnnl/dnnl.h"'
    echo "const void *c_functions[] = {"
    # -xc++ to get rid of c++-style comments that are part of c99,
    # but -xc -std=c99 doesn't work on macOS for whatever reason...
    cpp -xc++ -w "${@/#/-I}" "${dnnl_root}/include/oneapi/dnnl/dnnl.h" \
        | grep -o 'dnnl_\w\+(' \
        | sed 's/\(.*\)(/(void*)\1,/g' \
        | sort -u
    printf 'NULL};\nint main() { return 0; }\n'
} > "$output"
