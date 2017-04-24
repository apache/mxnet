#!/bin/bash

set -e

FILE=~/Downloads/virtualenvlinuxcpuinstall.md

#NOTHING BEFORE IT?
SOURCE_REGEX="\`\`\`"

SOURCE_LINES=($(grep -n "${SOURCE_REGEX}" "${FILE}" | cut -d : -f 1))
printf 'hey%s\n' "${SOURCE_LINES[@]}"

COMMANDS=""
SOURCE_COUNT=$((${#SOURCE_LINES[@]}/2))

echo ${#SOURCE_LINES[@]} "number of tick lines"

for (( i=0; i <= ${#SOURCE_LINES[@]}; i+=2 ))
do
    open_line_number=${SOURCE_LINES[${i}]}
    close_line_number=${SOURCE_LINES[${i} + 1]}

    for (( j=${open_line_number}+1; j < ${close_line_number}; j++ ))
    do
	# 1) get the line from file given the line number
	# 2) remove everything up to the prompt character '$'
	# 3) trim leading and trailing spaces
        cmd=`sed -n ${j}p ${FILE} | sed 's/^.*$\(.*\)$/\1/' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'` 
        if [[ ! -z $cmd ]];
        then
            COMMANDS="${COMMANDS} ${cmd};"
        fi           
    done
done

echo $COMMANDS

docker run --rm ubuntu:14.04 bash -c "${COMMANDS}"
