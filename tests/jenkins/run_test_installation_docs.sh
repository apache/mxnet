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
#    echo ""
#    echo ""

    declare -a lineno_array=("${!1}")
    min=${2}
    max=${3}

#    echo "Min ${2} Max ${3}"
#    echo ""
#    echo "Input:"
#    printf "${lineno_array[*]}"

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


#    echo "Output:"
#    printf "${return_arr[*]}"    
#    echo ""
#    echo ""
    echo "${return_arr[@]}"
}

FILE=install.md

# range of lines inside Linux-Python-CPU instructions
LINUX_PYTHON_CPU_START_LINENO=$(grep -n "START - Linux Python CPU Installation Instructions" "${FILE}" | cut -d : -f 1)
LINUX_PYTHON_CPU_END_LINENO=$(grep -n "END - Linux Python CPU Installation Instructions" "${FILE}" | cut -d : -f 1)

LINUX_PYTHON_GPU_START_LINENO=$(grep -n "START - Linux Python GPU Installation Instructions" "${FILE}" | cut -d : -f 1)
LINUX_PYTHON_GPU_END_LINENO=$(grep -n "END - Linux Python GPU Installation Instructions" "${FILE}" | cut -d : -f 1)
#VERIFY ONLY ONE ELEMENT FOR EACH OF ABOVE


# line numbers of the start of installation type instructions
# ARE THESE WITHIN THE RANGE ABOVE?
LINUX_PYTHON_CPU_VIRTUALENV_LINENO_ALL=($(grep -n "<div class=\"virtualenv\">" "${FILE}" | cut -d : -f 1))
LINUX_PYTHON_CPU_PIP_LINENO_ALL=($(grep -n "<div class=\"pip\">" "${FILE}" | cut -d : -f 1))
LINUX_PYTHON_CPU_DOCKER_LINENO_ALL=($(grep -n "<div class=\"docker\">" "${FILE}" | cut -d : -f 1))
LINUX_PYTHON_CPU_BUILDFROMSOURCE_LINENO_ALL=($(grep -n "<div class=\"build-from-source\">" "${FILE}" | cut -d : -f 1))

printf "\nStart: ${LINUX_PYTHON_CPU_START_LINENO}\n"
printf "\nEnd: ${LINUX_PYTHON_CPU_END_LINENO}\n"
printf "\nStart: ${LINUX_PYTHON_GPU_START_LINENO}\n"
printf "\nEnd: ${LINUX_PYTHON_GPU_END_LINENO}\n"
#printf "\nVirtualEnv: ${LINUX_PYTHON_CPU_VIRTUALENV_LINENO[*]}\n"
#printf "\nPip: ${LINUX_PYTHON_CPU_PIP_LINENO[*]}\n"
#printf "\nDocker: ${LINUX_PYTHON_CPU_DOCKER_LINENO[*]}\n"
#printf "\nBuildFromSource: ${LINUX_PYTHON_CPU_BUILDFROMSOURCE_LINENO[*]}\n"

LINUX_PYTHON_CPU_VIRTUALENV_LINENO=($(remove_out_of_range LINUX_PYTHON_CPU_VIRTUALENV_LINENO_ALL[@] ${LINUX_PYTHON_CPU_START_LINENO} ${LINUX_PYTHON_CPU_END_LINENO})) 
LINUX_PYTHON_CPU_PIP_LINENO=($(remove_out_of_range LINUX_PYTHON_CPU_PIP_LINENO_ALL[@] ${LINUX_PYTHON_CPU_START_LINENO} ${LINUX_PYTHON_CPU_END_LINENO}))
LINUX_PYTHON_CPU_DOCKER_LINENO=($(remove_out_of_range LINUX_PYTHON_CPU_DOCKER_LINENO_ALL[@] ${LINUX_PYTHON_CPU_START_LINENO} ${LINUX_PYTHON_CPU_END_LINENO}))
LINUX_PYTHON_CPU_BUILDFROMSOURCE_LINENO=($(remove_out_of_range LINUX_PYTHON_CPU_BUILDFROMSOURCE_LINENO_ALL[@] ${LINUX_PYTHON_CPU_START_LINENO} ${LINUX_PYTHON_CPU_END_LINENO}))

remove_out_of_range LINUX_PYTHON_CPU_VIRTUALENV_LINENO_ALL[@] ${LINUX_PYTHON_GPU_START_LINENO} ${LINUX_PYTHON_GPU_END_LINENO}
LINUX_PYTHON_GPU_VIRTUALENV_LINENO=($(remove_out_of_range LINUX_PYTHON_CPU_VIRTUALENV_LINENO_ALL[@] ${LINUX_PYTHON_GPU_START_LINENO} ${LINUX_PYTHON_GPU_END_LINENO}))
LINUX_PYTHON_GPU_PIP_LINENO=($(remove_out_of_range LINUX_PYTHON_CPU_PIP_LINENO_ALL[@] ${LINUX_PYTHON_GPU_START_LINENO} ${LINUX_PYTHON_GPU_END_LINENO}))
LINUX_PYTHON_GPU_DOCKER_LINENO=($(remove_out_of_range LINUX_PYTHON_CPU_DOCKER_LINENO_ALL[@] ${LINUX_PYTHON_GPU_START_LINENO} ${LINUX_PYTHON_GPU_END_LINENO}))
LINUX_PYTHON_GPU_BUILDFROMSOURCE_LINENO=($(remove_out_of_range LINUX_PYTHON_CPU_BUILDFROMSOURCE_LINENO_ALL[@] ${LINUX_PYTHON_GPU_START_LINENO} ${LINUX_PYTHON_GPU_END_LINENO}))

printf "\nVirtualEnv: ${LINUX_PYTHON_CPU_VIRTUALENV_LINENO[*]}\n"
printf "\nPip: ${LINUX_PYTHON_CPU_PIP_LINENO[*]}\n"
printf "\nDocker: ${LINUX_PYTHON_CPU_DOCKER_LINENO[*]}\n"
printf "\nBuildFromSource: ${LINUX_PYTHON_CPU_BUILDFROMSOURCE_LINENO[*]}\n"

printf "\nVirtualEnv: ${LINUX_PYTHON_GPU_VIRTUALENV_LINENO[*]}\n"
printf "\nPip: ${LINUX_PYTHON_GPU_PIP_LINENO[*]}\n"
printf "\nDocker: ${LINUX_PYTHON_GPU_DOCKER_LINENO[*]}\n"
printf "\nBuildFromSource: ${LINUX_PYTHON_GPU_BUILDFROMSOURCE_LINENO[*]}\n"
# MAKE SURE ONLY ONE ITEM IN EACH

SOURCE_REGEX="\`\`\`"

SOURCE_LINES=($(grep -n "${SOURCE_REGEX}" "${FILE}" | cut -d : -f 1))
printf "${SOURCE_LINES[*]}"


SOURCE_COUNT=$((${#SOURCE_LINES[@]}/2))

# given a number and an array of numbers, retrieves the index whose value is equal the number or the next greater thing
# assumes array is sorted
function retrieve_closest_index() {
    declare -a arr=("${!1}")
    number=${2}
    find_next_greatest=true

    if [[ -z ${number} || -z ${find_next_greatest} ]]
    then
        echo "Error: number and find_next_greatest must not be empty"
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
    echo "inside retrieve_Command *${section_start_index}* *${section_end_index}*"

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
    #printf "unsorted *${lineno_array[*]}* *${lineno_array[0]} ${lineno_array[1]} ${lineno_array[2]}* *${#lineno_array[@]}*"
    index=0
    size=${#lineno_array[@]}
    for (( i=1; i<=$(( $size-1)); i++ ))
    do
        j=$i
    #    echo "hey" ${#lineno_array[@]} $i $j
     #   printf "sorting... *${lineno_array[0]} ${lineno_array[1]} ${lineno_array[2]}* *${#lineno_array[@]}*"
        while (( ${j} > 0 && ${lineno_array[$j-1]} > ${lineno_array[$j]} )); do
            x=${lineno_array[$j-1]}
            lineno_array[$j-1]=${lineno_array[$j]}
            lineno_array[$j]=$x
            j=$j-1
        done
    done
    printf "${lineno_array[*]}"
#    return_arr+=(${number})
}

#function search() {

#}

# CPU
start_virtualenv_command_index=$(retrieve_closest_index SOURCE_LINES[@] ${LINUX_PYTHON_CPU_VIRTUALENV_LINENO[0]} true)
start_pip_command_index=$(retrieve_closest_index SOURCE_LINES[@] ${LINUX_PYTHON_CPU_PIP_LINENO[0]} true)
start_docker_command_index=$(retrieve_closest_index SOURCE_LINES[@] ${LINUX_PYTHON_CPU_DOCKER_LINENO[0]} true)
start_buildfromsource_command_index=$(retrieve_closest_index SOURCE_LINES[@] ${LINUX_PYTHON_CPU_BUILDFROMSOURCE_LINENO[0]} true)
end_pythoncpulinux_index=$(retrieve_closest_index SOURCE_LINES[@] ${LINUX_PYTHON_CPU_END_LINENO[0]} true)

#virtualenv_commands=$(retrieve_commands ${start_virtualenv_command_index} ${start_pip_command_index})
#pip_commands=$(retrieve_commands ${start_pip_command_index} ${start_docker_command_index})
#docker_commands=$(retrieve_commands ${start_docker_command_index} ${start_buildfromsource_command_index})
#buildfromsource_commands=$(retrieve_commands ${start_buildfromsource_command_index} ${end_pythoncpulinux_index})

echo "CPU"
echo "virtualenv start index: " ${start_virtualenv_command_index} "\n" ${virtualenv_commands}
echo "pip start index: " ${start_pip_command_index} "\n" ${pip_commands}
echo "docker start index: " ${start_docker_command_index} "\n" ${docker_commands}
echo "buildfromsource start index: " ${start_buildfromsource_command_index} "\n" ${buildfromsource_commands}

#################################

# GPU
#echo "retrieve commands for gpu"
#retrieve_closest_index SOURCE_LINES[@] ${LINUX_PYTHON_GPU_VIRTUALENV_LINENO[0]} true
#exit
start_virtualenv_command_index=$(retrieve_closest_index SOURCE_LINES[@] ${LINUX_PYTHON_GPU_VIRTUALENV_LINENO[0]} true)
start_pip_command_index=$(retrieve_closest_index SOURCE_LINES[@] ${LINUX_PYTHON_GPU_PIP_LINENO[0]} true)
start_docker_command_index=$(retrieve_closest_index SOURCE_LINES[@] ${LINUX_PYTHON_GPU_DOCKER_LINENO[0]} true)
start_buildfromsource_command_index=$(retrieve_closest_index SOURCE_LINES[@] ${LINUX_PYTHON_GPU_BUILDFROMSOURCE_LINENO[0]} true)
end_pythoncpulinux_index=$(retrieve_closest_index SOURCE_LINES[@] ${LINUX_PYTHON_GPU_END_LINENO[0]} true)
echo "sort"
a=(${start_virtualenv_command_index} ${start_pip_command_index} ${start_docker_command_index} ${start_buildfromsource_command_index} ${end_pythoncpulinux_index})
#a=(10 5 2)
printf "unsorted ${a[*]}"
sorted_indexes=($(sort a[@]))
printf "sorted ${sorted_indexes[*]}"

echo "virtualenv start index: " ${start_virtualenv_command_index}
echo "pip start index: " ${start_pip_command_index}
echo "docker start index: " ${start_docker_command_index}
echo "buildfromsource start index: " ${start_buildfromsource_command_index}
echo "pythoncpulinux end index: " ${end_pythoncpulinux_index}


end_virtualenv_command_index=$(retrieve_closest_index sorted_indexes[@] $(( ${start_virtualenv_command_index} + 1 )) true)
echo "virtualenv start: ${start_virtualenv_command_index}  end: ${sorted_indexes[$end_virtualenv_command_index]}"
end_pip_command_index=$(retrieve_closest_index sorted_indexes[@] $(( ${start_pip_command_index} + 1)) true)
echo "pip start: ${start_pip_command_index}  end: ${sorted_indexes[$end_pip_command_index]}"
end_docker_command_index=$(retrieve_closest_index sorted_indexes[@] $(( ${start_docker_command_index} + 1 )) true)
echo "docker start: ${start_docker_command_index}  end: ${sorted_indexes[$end_docker_command_index]}"
end_buildfromsource_command_index=$(retrieve_closest_index sorted_indexes[@] $(( ${start_buildfromsource_command_index} +1 )) true)
echo "buildfromsource start: ${start_buildfromsource_command_index}  end: ${sorted_indexes[$end_buildfromsource_command_index]}"

echo "retrieve commands"
retrieve_commands ${start_virtualenv_command_index} ${sorted_indexes[$end_virtualenv_command_index]}
virtualenv_commands=$(retrieve_commands ${start_virtualenv_command_index} ${sorted_indexes[$end_virtualenv_command_index]})
pip_commands=$(retrieve_commands ${start_pip_command_index} ${sorted_indexes[$end_pip_command_index]})
docker_commands=$(retrieve_commands ${start_docker_command_index} ${sorted_indexes[$end_docker_command_index]})
buildfromsource_commands=$(retrieve_commands ${start_buildfromsource_command_index} ${sorted_indexes[$end_buildfromsource_command_index]})

#echo "GPU"
echo "virtualenv start index: " ${start_virtualenv_command_index} "\n" ${virtualenv_commands}
echo "pip start index: " ${start_pip_command_index} "\n" ${pip_commands}
echo "docker start index: " ${start_docker_command_index} "\n" ${docker_commands}
echo "buildfromsource start index: " ${start_buildfromsource_command_index} "\n" ${buildfromsource_commands}




#################################

#docker run --rm ubuntu:14.04 bash -c "${virtualenv_commands}"
#docker run --rm ubuntu:14.04 bash -c "${pip_commands}"
#eval ${docker_commands}
#docker run --rm ubuntu:14.04 bash -c "${buildfromsource_commands}"
