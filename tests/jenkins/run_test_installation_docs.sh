#!/bin/bash

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

# given a number and an array of numbers, retrieves the index whose value is equal the number or the next greater thing
# assumes array is sorted
function retrieve_closest_index() {
    declare -a arr=("${!1}")
    number=${2}

    if [[ -z ${number}} ]]
    then
        echo "Error: number must not be empty"
        exit 1
    fi

    #echo ""
    #echo "retrieve_closest_index:"
    #echo ${number}
    #printf "${arr[*]}"

    for (( i=0; i < ${#arr[@]}; i++ ))
    do
        cur_num=${arr[${i}]}
        if [[ ${cur_num} -eq ${number} || ${cur_num} -gt ${number} ]]
        then
            echo ${i}
            return
        fi
    done

    #echo ""
}

# retrieves all bash commands between two given line numbers in a file
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
            cmd=`sed -n ${j}p ${FILE} | sed 's/^.*$\(.*\).*$/\1/' | sed 's/\(.*\)#.*$/\1/' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'`
            if [[ ! -z $cmd ]];
            then
                commands="${commands} ${cmd};"
            fi
        done
    done
    echo ${commands}
}

function sort() {
    declare -a lineno_array=("${!1}")
    return_arr=()
    index=0
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

FILE=install.md

# get all line numbers with "```" signifying start or end of source section and put them in an array
SOURCE_REGEX="\`\`\`"
SOURCE_LINES=($(grep -n "${SOURCE_REGEX}" "${FILE}" | cut -d : -f 1))

# line numbers of the start of installation method instructions regardless of platform
PYTHON_VIRTUALENV_LINENO_ALL=($(grep -n "<div class=\"virtualenv\">" "${FILE}" | cut -d : -f 1))
PYTHON_PIP_LINENO_ALL=($(grep -n "<div class=\"pip\">" "${FILE}" | cut -d : -f 1))
PYTHON_DOCKER_LINENO_ALL=($(grep -n "<div class=\"docker\">" "${FILE}" | cut -d : -f 1))
PYTHON_BUILDFROMSOURCE_LINENO_ALL=($(grep -n "<div class=\"build-from-source\">" "${FILE}" | cut -d : -f 1))

function get_instruction_set() {
    # pass start and end line numbers
    # set 4 arrays

    if [[ -z ${1} || -z ${2} ]]
    then
        echo "Error: start line number & end line number must not be empty"
        exit 1
    fi

    # range of all lines inside Linux-Python-CPU instructions
    LINUX_PYTHON_CPU_START_LINENO=${1}
    LINUX_PYTHON_CPU_END_LINENO=${2}

    # get line numbers of the start of each installation method instructions for Linux-Python-CPU
    LINUX_PYTHON_CPU_VIRTUALENV_LINENO=($(remove_out_of_range PYTHON_VIRTUALENV_LINENO_ALL[@] ${LINUX_PYTHON_CPU_START_LINENO} ${LINUX_PYTHON_CPU_END_LINENO}))
    LINUX_PYTHON_CPU_PIP_LINENO=($(remove_out_of_range PYTHON_PIP_LINENO_ALL[@] ${LINUX_PYTHON_CPU_START_LINENO} ${LINUX_PYTHON_CPU_END_LINENO}))
    LINUX_PYTHON_CPU_DOCKER_LINENO=($(remove_out_of_range PYTHON_DOCKER_LINENO_ALL[@] ${LINUX_PYTHON_CPU_START_LINENO} ${LINUX_PYTHON_CPU_END_LINENO}))
    LINUX_PYTHON_CPU_BUILDFROMSOURCE_LINENO=($(remove_out_of_range PYTHON_BUILDFROMSOURCE_LINENO_ALL[@] ${LINUX_PYTHON_CPU_START_LINENO} ${LINUX_PYTHON_CPU_END_LINENO}))

    # get indices (or the next closest thing) of the instruction sets' starting line numbers
    start_virtualenv_command_index=$(retrieve_closest_index SOURCE_LINES[@] ${LINUX_PYTHON_CPU_VIRTUALENV_LINENO[0]})
    start_pip_command_index=$(retrieve_closest_index SOURCE_LINES[@] ${LINUX_PYTHON_CPU_PIP_LINENO[0]})
    start_docker_command_index=$(retrieve_closest_index SOURCE_LINES[@] ${LINUX_PYTHON_CPU_DOCKER_LINENO[0]})
    start_buildfromsource_command_index=$(retrieve_closest_index SOURCE_LINES[@] ${LINUX_PYTHON_CPU_BUILDFROMSOURCE_LINENO[0]})
    end_pythoncpulinux_index=$(retrieve_closest_index SOURCE_LINES[@] ${LINUX_PYTHON_CPU_END_LINENO[0]})

    # sort the indices of the instruction sets' line numbers
    unsorted_indexes=(${start_virtualenv_command_index} ${start_pip_command_index} ${start_docker_command_index} ${start_buildfromsource_command_index} ${end_pythoncpulinux_index})
    sorted_indexes=($(sort unsorted_indexes[@]))

    # figure out the index of the instruction sets' ending line numbers
    end_virtualenv_command_index=$(retrieve_closest_index sorted_indexes[@] $(( ${start_virtualenv_command_index} + 1 )))
    end_pip_command_index=$(retrieve_closest_index sorted_indexes[@] $(( ${start_pip_command_index} + 1)))
    end_docker_command_index=$(retrieve_closest_index sorted_indexes[@] $(( ${start_docker_command_index} + 1 )))
    end_buildfromsource_command_index=$(retrieve_closest_index sorted_indexes[@] $(( ${start_buildfromsource_command_index} +1 )))

    # retrieve the instruction sets' commands using the starting and ending line numbers' indices
    virtualenv_commands=$(retrieve_commands ${start_virtualenv_command_index} ${sorted_indexes[$end_virtualenv_command_index]})
    pip_commands=$(retrieve_commands ${start_pip_command_index} ${sorted_indexes[$end_pip_command_index]})
    docker_commands=$(retrieve_commands ${start_docker_command_index} ${sorted_indexes[$end_docker_command_index]})
    buildfromsource_commands=$(retrieve_commands ${start_buildfromsource_command_index} ${sorted_indexes[$end_buildfromsource_command_index]})
}

########################LINUX-PYTHON-CPU############################

echo
echo
echo "### Testing LINUX-PYTHON-CPU ###"
echo
# range of all lines inside Linux-Python-CPU instructions
LINUX_PYTHON_CPU_START_LINENO=$(grep -n "START - Linux Python CPU Installation Instructions" "${FILE}" | cut -d : -f 1)
LINUX_PYTHON_CPU_END_LINENO=$(grep -n "END - Linux Python CPU Installation Instructions" "${FILE}" | cut -d : -f 1)

get_instruction_set ${LINUX_PYTHON_CPU_START_LINENO} ${LINUX_PYTHON_CPU_END_LINENO}
docker run --rm ubuntu:14.04 bash -c "${virtualenv_commands}"

# get line numbers of the start of each installation method instructions for Linux-Python-CPU
LINUX_PYTHON_CPU_VIRTUALENV_LINENO=($(remove_out_of_range PYTHON_VIRTUALENV_LINENO_ALL[@] ${LINUX_PYTHON_CPU_START_LINENO} ${LINUX_PYTHON_CPU_END_LINENO}))
LINUX_PYTHON_CPU_PIP_LINENO=($(remove_out_of_range PYTHON_PIP_LINENO_ALL[@] ${LINUX_PYTHON_CPU_START_LINENO} ${LINUX_PYTHON_CPU_END_LINENO}))
LINUX_PYTHON_CPU_DOCKER_LINENO=($(remove_out_of_range PYTHON_DOCKER_LINENO_ALL[@] ${LINUX_PYTHON_CPU_START_LINENO} ${LINUX_PYTHON_CPU_END_LINENO}))
LINUX_PYTHON_CPU_BUILDFROMSOURCE_LINENO=($(remove_out_of_range PYTHON_BUILDFROMSOURCE_LINENO_ALL[@] ${LINUX_PYTHON_CPU_START_LINENO} ${LINUX_PYTHON_CPU_END_LINENO}))

# get indices (or the next closest thing) of the instruction sets' starting line numbers
start_virtualenv_command_index=$(retrieve_closest_index SOURCE_LINES[@] ${LINUX_PYTHON_CPU_VIRTUALENV_LINENO[0]})
start_pip_command_index=$(retrieve_closest_index SOURCE_LINES[@] ${LINUX_PYTHON_CPU_PIP_LINENO[0]})
start_docker_command_index=$(retrieve_closest_index SOURCE_LINES[@] ${LINUX_PYTHON_CPU_DOCKER_LINENO[0]})
start_buildfromsource_command_index=$(retrieve_closest_index SOURCE_LINES[@] ${LINUX_PYTHON_CPU_BUILDFROMSOURCE_LINENO[0]})
end_pythoncpulinux_index=$(retrieve_closest_index SOURCE_LINES[@] ${LINUX_PYTHON_CPU_END_LINENO[0]})

# sort the indices of the instruction sets' line numbers
unsorted_indexes=(${start_virtualenv_command_index} ${start_pip_command_index} ${start_docker_command_index} ${start_buildfromsource_command_index} ${end_pythoncpulinux_index})
sorted_indexes=($(sort unsorted_indexes[@]))

# figure out the index of the instruction sets' ending line numbers
end_virtualenv_command_index=$(retrieve_closest_index sorted_indexes[@] $(( ${start_virtualenv_command_index} + 1 )))
end_pip_command_index=$(retrieve_closest_index sorted_indexes[@] $(( ${start_pip_command_index} + 1)))
end_docker_command_index=$(retrieve_closest_index sorted_indexes[@] $(( ${start_docker_command_index} + 1 )))
end_buildfromsource_command_index=$(retrieve_closest_index sorted_indexes[@] $(( ${start_buildfromsource_command_index} +1 )))

# retrieve the instruction sets' commands using the starting and ending line numbers' indices
virtualenv_commands=$(retrieve_commands ${start_virtualenv_command_index} ${sorted_indexes[$end_virtualenv_command_index]})
pip_commands=$(retrieve_commands ${start_pip_command_index} ${sorted_indexes[$end_pip_command_index]})
docker_commands=$(retrieve_commands ${start_docker_command_index} ${sorted_indexes[$end_docker_command_index]})
buildfromsource_commands=$(retrieve_commands ${start_buildfromsource_command_index} ${sorted_indexes[$end_buildfromsource_command_index]})

echo "virtualenv commands: " ${virtualenv_commands}
echo "pip commands: " ${pip_commands}
echo "docker commands: " ${docker_commands}
echo "build from source commands: " ${buildfromsource_commands}

#docker run --rm ubuntu:14.04 bash -c "${virtualenv_commands}"
#docker run --rm ubuntu:14.04 bash -c "${pip_commands}"
#eval ${docker_commands}
#docker run --rm ubuntu:14.04 bash -c "${buildfromsource_commands}"

#########################LINUX-PYTHON-GPU###########################

echo
echo
echo "### Testing LINUX-PYTHON-GPU ###"
echo
# range of all lines inside Linux-Python-GPU instructions
LINUX_PYTHON_GPU_START_LINENO=$(grep -n "START - Linux Python GPU Installation Instructions" "${FILE}" | cut -d : -f 1)
LINUX_PYTHON_GPU_END_LINENO=$(grep -n "END - Linux Python GPU Installation Instructions" "${FILE}" | cut -d : -f 1)
#VERIFY ONLY ONE ELEMENT FOR EACH OF ABOVE

get_instruction_set ${LINUX_PYTHON_GPU_START_LINENO} ${LINUX_PYTHON_GPU_END_LINENO}
nvidia-docker run --rm nvidia/cuda:7.5-cudnn5-devel bash -c "${virtualenv_commands}"
exit

# get line numbers of the start of each installation instruction sets for Linux-Python-GPU
LINUX_PYTHON_GPU_VIRTUALENV_LINENO=($(remove_out_of_range PYTHON_VIRTUALENV_LINENO_ALL[@] ${LINUX_PYTHON_GPU_START_LINENO} ${LINUX_PYTHON_GPU_END_LINENO}))
LINUX_PYTHON_GPU_PIP_LINENO=($(remove_out_of_range PYTHON_PIP_LINENO_ALL[@] ${LINUX_PYTHON_GPU_START_LINENO} ${LINUX_PYTHON_GPU_END_LINENO}))
LINUX_PYTHON_GPU_DOCKER_LINENO=($(remove_out_of_range PYTHON_DOCKER_LINENO_ALL[@] ${LINUX_PYTHON_GPU_START_LINENO} ${LINUX_PYTHON_GPU_END_LINENO}))
LINUX_PYTHON_GPU_BUILDFROMSOURCE_LINENO=($(remove_out_of_range PYTHON_BUILDFROMSOURCE_LINENO_ALL[@] ${LINUX_PYTHON_GPU_START_LINENO} ${LINUX_PYTHON_GPU_END_LINENO}))

# get indices (or the next closest thing) of the instruction sets' starting line numbers
start_virtualenv_command_index=$(retrieve_closest_index SOURCE_LINES[@] ${LINUX_PYTHON_GPU_VIRTUALENV_LINENO[0]})
start_pip_command_index=$(retrieve_closest_index SOURCE_LINES[@] ${LINUX_PYTHON_GPU_PIP_LINENO[0]})
start_docker_command_index=$(retrieve_closest_index SOURCE_LINES[@] ${LINUX_PYTHON_GPU_DOCKER_LINENO[0]})
start_buildfromsource_command_index=$(retrieve_closest_index SOURCE_LINES[@] ${LINUX_PYTHON_GPU_BUILDFROMSOURCE_LINENO[0]})
end_pythongpulinux_index=$(retrieve_closest_index SOURCE_LINES[@] ${LINUX_PYTHON_GPU_END_LINENO[0]})

# sort the indices of the instruction sets' line numbers
unsorted_indexes=(${start_virtualenv_command_index} ${start_pip_command_index} ${start_docker_command_index} ${start_buildfromsource_command_index} ${end_pythongpulinux_index})
sorted_indexes=($(sort unsorted_indexes[@]))

# figure out the index of the instruction sets' ending line numbers
end_virtualenv_command_index=$(retrieve_closest_index sorted_indexes[@] $(( ${start_virtualenv_command_index} + 1 )))
end_pip_command_index=$(retrieve_closest_index sorted_indexes[@] $(( ${start_pip_command_index} + 1)))
end_docker_command_index=$(retrieve_closest_index sorted_indexes[@] $(( ${start_docker_command_index} + 1 )))
end_buildfromsource_command_index=$(retrieve_closest_index sorted_indexes[@] $(( ${start_buildfromsource_command_index} +1 )))

# retrieve the instruction sets' commands using the starting and ending line numbers' indices
virtualenv_commands=$(retrieve_commands ${start_virtualenv_command_index} ${sorted_indexes[$end_virtualenv_command_index]})
pip_commands=$(retrieve_commands ${start_pip_command_index} ${sorted_indexes[$end_pip_command_index]})
docker_commands=$(retrieve_commands ${start_docker_command_index} ${sorted_indexes[$end_docker_command_index]})
buildfromsource_commands=$(retrieve_commands ${start_buildfromsource_command_index} ${sorted_indexes[$end_buildfromsource_command_index]})

echo "virtualenv commands: " ${virtualenv_commands}
echo "pip commands: " ${pip_commands}
echo "docker commands: " ${docker_commands}
echo "build from source commands: " ${buildfromsource_commands}

nvidia-docker run --rm nvidia/cuda:7.5-cudnn5-devel bash -c "${virtualenv_commands}"
nvidia-docker run --rm nvidia/cuda:7.5-cudnn5-devel bash -c "${pip_commands}"
eval ${docker_commands}
nvidia-docker run --rm nvidia/cuda:7.5-cudnn5-devel bash -c "${buildfromsource_commands}"
