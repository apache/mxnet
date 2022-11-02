#!/bin/bash

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


set -e

# Given an array of numbers, removes any numbers of it that fall outside a given range.
#
# $1 - Array of numbers
# $2 - Min
# $3 - Max
#
# Example:
#   arr=(1 5 15)
#   remove_out_of_range arr[@] 4 10
#
#   returns (5)
function remove_out_of_range() {

    declare -a lineno_array=("${!1}")
    min=${2}
    max=${3}

    if [[ -z "${min}" || -z "${max}" ]];
    then
        echo "Error: Min and max must not be empty"
        exit 1
    fi

    if [[ "${min}" -gt "${max}" ]]
    then
        echo "Error: Min must be less than or equal to Max"
        exit 1
    fi

    return_arr=()

    for number in "${lineno_array[@]}"
    do
        if (( ${number} > ${min} && ${number} < ${max} ))
        then
            return_arr+=(${number})
        fi
    done

    echo "${return_arr[@]}"
}

# given a number and an array of numbers, retrieves the index whose value is equal to the number or the next greatest
# thing. assumes array is sorted.
#
# $1 - array of numbers
# $2 - number
#
# Example:
#   a=(1 3 5 7 9 10)
#   retrieve_closest_index a[@] 5
#
#   returns 2
#
#   retrieve_closest_index a[@] 8
#
#   returns 4
#
function retrieve_closest_index() {
    declare -a arr=("${!1}")
    number=${2}

    if [[ -z ${number}} ]]
    then
        echo "Error: number must not be empty"
        exit 1
    fi

    for (( i=0; i < ${#arr[@]}; i++ ))
    do
        cur_num=${arr[${i}]}
        if [[ ${cur_num} -eq ${number} || ${cur_num} -gt ${number} ]]
        then
            echo ${i}
            return
        fi
    done
}

# retrieves all bash commands between two given line numbers in a file
#
# $1 - Start line number
# $2 - End line number
#
# Example:
#   a=(4 1 3)
#   sort a[@]
#
function retrieve_commands() {
    section_start_index=${1}
    section_end_index=${2}

    if [[ -z ${section_start_index} || -z ${section_end_index} ]]
    then
        echo "Error: section_start & section_end must not be empty"
        exit 1
    fi

    commands=""

    for (( index=${section_start_index}; index < ${section_end_index}; index+=2 ))
    do
        open_line_number=${SOURCE_LINES[${index}]}
        close_line_number=${SOURCE_LINES[${index} + 1]}

        for (( j=${open_line_number}+1; j < ${close_line_number}; j++ ))
        do
            # 1) get the line from file given the line number
            # 2) remove everything up to the prompt character '$'
	        # 3) remove everything after a comment character"#"
            # 4) trim leading and trailing spaces
            current_line=`sed -n ${j}p ${FILE}`
            if [[ ${current_line} == *"$"* ]]
            then
                cmd=`echo ${current_line} | cut -d$ -f2- | sed 's/\(.*\)#.*$/\1/' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'`
                if [[ ! -z $cmd ]];
                then
                    commands="${commands} ${cmd};"
                fi
            fi
        done
    done
    echo ${commands}
}

# Sorts array of numbers.
#
# $1 - Start line number
# $2 - End line number
#
# Example:
#   a=(4 1 3)
#   sort a[@]
#
function sort() {
    declare -a lineno_array=("${!1}")
    size=${#lineno_array[@]}
    for (( i=1; i<=$(( $size-1)); i++ ))
    do
        j=$i
        while (( ${j} > 0 && ${lineno_array[$j-1]} > ${lineno_array[$j]} )); do
            x=${lineno_array[$j-1]}
            lineno_array[$j-1]=${lineno_array[$j]}
            lineno_array[$j]=$x
            j=$j-1
        done
    done
    printf "${lineno_array[*]}"
}

if (( $# < 1 )); then
    echo ""
    echo "Usage: $(basename $0) FILE"
    echo ""
    exit 1
fi
FILE=${1}

# get all line numbers with "```" signifying start or end of source section and put them in an array
SOURCE_REGEX="\`\`\`"
SOURCE_LINES=($(grep -n "${SOURCE_REGEX}" "${FILE}" | cut -d : -f 1))

# line numbers of the start of installation method instructions regardless of platform
VIRTUALENV_LINENO_ALL=($(grep -n "<div class=\"virtualenv\">" "${FILE}" | cut -d : -f 1))
PIP_LINENO_ALL=($(grep -n "<div class=\"pip\">" "${FILE}" | cut -d : -f 1))
DOCKER_LINENO_ALL=($(grep -n "<div class=\"docker\">" "${FILE}" | cut -d : -f 1))
BUILDFROMSOURCE_LINENO_ALL=($(grep -n "<div class=\"build-from-source\">" "${FILE}" | cut -d : -f 1))

# Given two line numbers, collects instruction sets for installing via Virtualenv, Pip, Docker, and source within the
# two lines assuming there is one of each.
#
# $1 - Start line number
# $2 - End line number
#
# Example:
#   set_instruction_set 0 64
#
function set_instruction_set() {

    if [[ -z ${1} || -z ${2} ]]
    then
        echo "Error: start line number & end line number must not be empty"
        exit 1
    fi

    if [[ ${1} -gt ${2} ]]
    then
        echo "Error: start line number must be smaller then end line number"
        exit 1
    fi

    # range of all lines inside Linux-Python-CPU instructions
    START_LINENO=${1}
    END_LINENO=${2}

    # get line numbers of the start of each installation method instructions for Linux-Python-CPU
    VIRTUALENV_LINENO=($(remove_out_of_range VIRTUALENV_LINENO_ALL[@] ${START_LINENO} ${END_LINENO}))
    PIP_LINENO=($(remove_out_of_range PIP_LINENO_ALL[@] ${START_LINENO} ${END_LINENO}))
    DOCKER_LINENO=($(remove_out_of_range DOCKER_LINENO_ALL[@] ${START_LINENO} ${END_LINENO}))
    BUILDFROMSOURCE_LINENO=($(remove_out_of_range BUILDFROMSOURCE_LINENO_ALL[@] ${START_LINENO} ${END_LINENO}))

    # get indices (or the next closest thing) of the instruction sets' starting line numbers
    start_virtualenv_command_index=$(retrieve_closest_index SOURCE_LINES[@] ${VIRTUALENV_LINENO[0]})
    start_pip_command_index=$(retrieve_closest_index SOURCE_LINES[@] ${PIP_LINENO[0]})
    start_docker_command_index=$(retrieve_closest_index SOURCE_LINES[@] ${DOCKER_LINENO[0]})
    start_buildfromsource_command_index=$(retrieve_closest_index SOURCE_LINES[@] ${BUILDFROMSOURCE_LINENO[0]})
    end_index=$(retrieve_closest_index SOURCE_LINES[@] ${END_LINENO[0]})

    # sort the indices of the instruction sets' line numbers
    unsorted_indexes=(${start_virtualenv_command_index} ${start_pip_command_index} ${start_docker_command_index} \
        ${start_buildfromsource_command_index} ${end_index})
    sorted_indexes=($(sort unsorted_indexes[@]))

    # figure out the index of the instruction sets' ending line numbers
    end_virtualenv_command_index=$(retrieve_closest_index sorted_indexes[@] \
        $(( ${start_virtualenv_command_index} + 1 )))
    end_pip_command_index=$(retrieve_closest_index sorted_indexes[@] $(( ${start_pip_command_index} + 1)))
    end_docker_command_index=$(retrieve_closest_index sorted_indexes[@] $(( ${start_docker_command_index} + 1 )))
    end_buildfromsource_command_index=$(retrieve_closest_index sorted_indexes[@] \
        $(( ${start_buildfromsource_command_index} +1 )))

    # retrieve the instruction sets' commands using the starting and ending line numbers' indices
    virtualenv_commands=$(retrieve_commands ${start_virtualenv_command_index} \
        ${sorted_indexes[$end_virtualenv_command_index]})
    pip_commands=$(retrieve_commands ${start_pip_command_index} ${sorted_indexes[$end_pip_command_index]})
    docker_commands=$(retrieve_commands ${start_docker_command_index} ${sorted_indexes[$end_docker_command_index]})
    buildfromsource_commands=$(retrieve_commands ${start_buildfromsource_command_index} \
        ${sorted_indexes[$end_buildfromsource_command_index]})
}

# given a $buildfromsource_commands string, filter out any build commands that should not be executed
# during the build from source tests. An example, the build from source instructions include the commands:
# $ git clone --recursive https://github.com/apache/mxnet 
# $ cd mxnet 
# if these commands get executed in the jenkins job, we will be testing the build from source instructions
# against the master branch and not against the version of the repository that Jenkins checks out for testing.
# This presents a particularly big problem for the version branches and their nightly builds. Because, 
# we would, in effect, be testing the build from source instructions for one version of MXNet against
# the master branch.
# in this function we target the commands cited in the example above.
# See also gh issue: https://github.com/apache/mxnet/issues/13800
function filter_build_commands() {
    filtered_build_commands="${1}"

    # Remove git commands
    filtered_build_commands=`echo "${filtered_build_commands}" | perl -pe 's/git .*?;//g'`

    # Remove 'cd mxnet'
    filtered_build_commands=`echo "${filtered_build_commands}" | perl -pe 's/cd mxnet;//'`

    echo "${filtered_build_commands}"
}

########################LINUX-PYTHON-CPU############################
echo
echo
echo "### Testing LINUX-PYTHON-CPU ###"
echo
# range of all lines inside Linux-Python-CPU instructions
LINUX_PYTHON_CPU_START_LINENO=$(grep -n "START - Linux Python CPU Installation Instructions" "${FILE}" | cut -d : -f 1)
LINUX_PYTHON_CPU_END_LINENO=$(grep -n "END - Linux Python CPU Installation Instructions" "${FILE}" | cut -d : -f 1)

set_instruction_set ${LINUX_PYTHON_CPU_START_LINENO} ${LINUX_PYTHON_CPU_END_LINENO}

ubuntu_python_cpu_virtualenv()
{
    set -e
    echo
    echo "### Testing Virtualenv ###"
    echo "${virtualenv_commands}" #> "$filewithcommands"
    echo
    eval ${virtualenv_commands}
    echo "ubuntu_python_cpu_virtualenv: MXNet Installed Successfully"
}

ubuntu_python_cpu_pip()
{
    set -e
    echo
    echo "### Testing Pip ###"
    pip_commands="sudo ${pip_commands}"
    echo "${pip_commands}"
    echo
    eval ${pip_commands}
    echo "ubuntu_python_cpu_pip: MXNet Installed Successfully"
}

ubuntu_python_cpu_docker()
{
    set -e
    echo
    echo "### Testing Docker ###"
    echo "${docker_commands}"
    echo
    eval ${docker_commands}
    echo "ubuntu_python_cpu_docker: MXNet Installed Successfully"

}

ubuntu_python_cpu_source()
{
    set -e
    echo
    echo "### Testing Build From Source ###"
    buildfromsource_commands=$(filter_build_commands "${buildfromsource_commands}")
    echo ${buildfromsource_commands}
    eval ${buildfromsource_commands}
    echo "ubuntu_python_cpu_source: MXNet Installed Successfully"

}

#########################LINUX-PYTHON-GPU###########################

echo
echo
echo "### Testing LINUX-PYTHON-GPU ###"
echo
# range of all lines inside Linux-Python-GPU instructions
LINUX_PYTHON_GPU_START_LINENO=$(grep -n "START - Linux Python GPU Installation Instructions" "${FILE}" | cut -d : -f 1)
LINUX_PYTHON_GPU_END_LINENO=$(grep -n "END - Linux Python GPU Installation Instructions" "${FILE}" | cut -d : -f 1)

set_instruction_set ${LINUX_PYTHON_GPU_START_LINENO} ${LINUX_PYTHON_GPU_END_LINENO}


ubuntu_python_gpu_virtualenv()
{
    set -e
    echo
    echo "### Testing Virtualenv ###"
    echo "${virtualenv_commands}"
    echo
    eval ${virtualenv_commands}
    echo "ubuntu_python_gpu_virtualenv: MXNet Installed Successfully"

}

ubuntu_python_gpu_pip()
{
    set -e
    echo
    echo "### Testing Pip ###"
    echo "${pip_commands}"
    echo
    eval ${pip_commands}
    echo "ubuntu_python_gpu_pip: MXNet Installed Successfully"

}

ubuntu_python_gpu_docker()
{
    set -e
    echo
    echo "### Testing Docker ###"
    echo "${docker_commands}"
    echo
    eval ${docker_commands}
    echo "ubuntu_python_gpu_docker: MXNet Installed Successfully"

}

ubuntu_python_gpu_source()
{
    set -e
    echo
    echo "### Testing Build From Source ###"
    buildfromsource_commands=$(filter_build_commands "${buildfromsource_commands}")
    echo ${buildfromsource_commands}
    eval ${buildfromsource_commands}
    echo "ubuntu_python_gpu_source: MXNet Installed Successfully"

}

func_virtual_commands()
{
  echo ${virtualenv_commands}
}
