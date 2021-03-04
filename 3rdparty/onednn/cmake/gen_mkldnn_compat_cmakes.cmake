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

# Creates cmake config for MKLDNN based on oneDNN one
# (by replacing DNNL with MKLDNN)
# Parameters:
#   DIR -- path to cmake install dir

set(DNNL_DIR ${DIR}/dnnl)
set(MKLDNN_DIR ${DIR}/mkldnn)

file(MAKE_DIRECTORY ${MKLDNN_DIR})

file(GLOB_RECURSE fs "${DNNL_DIR}/*")
foreach(f ${fs})
    # set the destination
    file(RELATIVE_PATH frel ${DNNL_DIR} ${f})
    string(REGEX REPLACE "dnnl" "mkldnn" dest_rel "${frel}")
    set(dest "${MKLDNN_DIR}/${dest_rel}")
    # message(STATUS "file: ${f} --> ${frel} --> ${dest_rel} --> ${dest}")

    # read and change the content of the file
    file(STRINGS ${f} contents NEWLINE_CONSUME)
    string(REGEX REPLACE "DNNL" "MKLDNN" contents "${contents}")
    string(REGEX REPLACE "dnnl" "mkldnn" contents "${contents}")
    foreach (ext "a" "so" "dylib" "dll" "lib")
        string(REGEX REPLACE "mkldnn[.]${ext}" "dnnl.${ext}" contents "${contents}")
    endforeach()
    string(REGEX REPLACE "lmkldnn" "ldnnl" contents "${contents}")

    # store the result
    file(WRITE ${dest} ${contents})
endforeach()
