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

# Generates cpp file with GPU kernel code stored as string
# Parameters:
#   KER_FILE    -- path to the kernel source file
#   KER_INC_DIR -- include directory
#   GEN_FILE   -- path to the generated cpp file
#===============================================================================

# Read lines of kernel file and recursively substitute 'include'
# preprocessor directives.
#   ker_file  -- path to the kernel file
#   ker_lines -- list with code lines
function(read_lines ker_file ker_lines)
    file(STRINGS ${ker_file} contents NEWLINE_CONSUME)
    # Replace square brackets as they have special meaning in CMake
    string(REGEX REPLACE "\\[" "__BRACKET0__" contents "${contents}")
    string(REGEX REPLACE "\\]" "__BRACKET1__" contents "${contents}")
    # Escape backslash
    string(REGEX REPLACE "\\\\([^\n;])" "\\\\\\\\\\1" contents "${contents}")
    # Escape backslash (space is to avoid '\;' sequences after the split to a list)
    string(REGEX REPLACE "\\\\\n" "\\\\\\\\ \n" contents "${contents}")
    # Use EOL to split the contents to a list
    string(REGEX REPLACE "\n" ";" contents "${contents}")

    set(pp_lines)
    foreach(l ${contents})
        if(l MATCHES "^\\s*#include \"(.*)\"")
            set(inc_file "${KER_INC_DIR}/${CMAKE_MATCH_1}")
            set(inc_lines)
            read_lines(${inc_file} inc_lines)
            list(APPEND pp_lines "${inc_lines}")
        else()
            string(REGEX REPLACE ";" "\\\\;" esc_line "${l}")
            list(APPEND pp_lines "${esc_line}")
        endif()
    endforeach()
    set(${ker_lines} "${pp_lines}" PARENT_SCOPE)
endfunction()

read_lines(${KER_FILE} ker_lines)

# Replace unescaped semicolon by EOL
string(REGEX REPLACE "([^\\]|^);" "\\1\n" ker_lines "${ker_lines}")
# Unescape semicolon
string (REGEX REPLACE "\\\\;" ";" ker_lines "${ker_lines}")
# Escape quatation marks
string(REGEX REPLACE "\"" "\\\\\"" ker_lines "${ker_lines}")
# Add EOLs
string(REGEX REPLACE " ?\n" "\\\\n\",\n\"" ker_lines "${ker_lines}")
# Replace square brackets back
string(REGEX REPLACE "__BRACKET0__" "[" ker_lines "${ker_lines}")
string(REGEX REPLACE "__BRACKET1__" "]" ker_lines "${ker_lines}")

get_filename_component(ker_name ${KER_FILE} NAME_WE)

set(ker_lang "ocl")

set(ker_contents  "const char *${ker_name}_kernel[] ={ \"${ker_lines}\", nullptr };")
set(ker_contents "namespace ${ker_lang} {\n${ker_contents}\n}")
set(ker_contents "namespace gpu {\n${ker_contents}\n}")
set(ker_contents "namespace impl {\n${ker_contents}\n}")
set(ker_contents "namespace dnnl {\n${ker_contents}\n}")
file(WRITE ${GEN_FILE} "${ker_contents}")
