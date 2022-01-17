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

# Script for running python benchmark against number of used OMP threads


help_and_exit() {
	echo "Usage:"
	echo "    `basename "$0"`  [start_num_threads step_num_threads end_num_threads] python_script [python script parameters]"
	echo "Number of threads range parameters and python script are optional."
	exit
}

if [ "$#" -eq 0 ] ; then
	help_and_exit
fi

NUM_SOCKET=`lscpu | grep 'Socket(s)' | awk '{print $NF}'`
CORES_PER_SOCKET=`lscpu | grep 'Core(s) per socket' | awk '{print $NF}'`
NUM_CORES=$((CORES_PER_SOCKET * NUM_SOCKET))

NT_START=1
NT_STEP=1
NT_END=$NUM_CORES

integer_reg='^[0-9]+$'
signed_integer_reg='^[+-]*[0-9]+$'
if [[ $1 =~ $integer_reg ]] ; then
	if [[ $2 =~ $signed_integer_reg ]] && [[ $3 =~ $integer_reg ]]; then
		NT_START=$1
		NT_STEP=$2
		NT_END=$3
		shift 3
		if [ "$#" -eq 0 ] ; then
			help_and_exit
		fi
	else
		echo "Provide 3 numbers for threads range: start, step and the end."
		help_and_exit
	fi
fi

NT_SEQUENCE=`seq $NT_START $NT_STEP $NT_END`
if [ -z "$NT_SEQUENCE" ]; then
	echo "Given threads range produce empy sequence."
	help_and_exit
else
	echo "Start python script $1 for following number of threads:"  >&2
	echo $NT_SEQUENCE  >&2
fi

RUN_SCRIPT=`dirname "$0"`/run.sh
for NT in $NT_SEQUENCE;
do
	TMP_FILE=/tmp/_result_${NT}.txt
	echo  1>${TMP_FILE}
	if [[ $NT -eq $NT_START ]]; then
		echo "NUM_THREADS = $NT" 1>>${TMP_FILE}
		$RUN_SCRIPT $NT $@ 1>>${TMP_FILE}
	else
		echo " $NT" 1>>${TMP_FILE}
		$RUN_SCRIPT $NT $@ --no_size_column --no_test_header 1>>${TMP_FILE}
	fi
	TMP_FILES+=" ${TMP_FILE}"
done
paste -d "" ${TMP_FILES}
