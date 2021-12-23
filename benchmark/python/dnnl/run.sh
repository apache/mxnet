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

# Script for running python benchmark with properly setting OMP prarameters for it

check_parametrs() {
 	if [ "$#" -eq 0 ] ; then
		echo "Please give python script to run as parameter."
		echo "Optionally you can give number of threads to use and python scripts parameters:"
		echo "    `basename "$0"`  [num_threads] python_script [python script parameters]"
		exit
	fi
}

check_parametrs $@

NUM_SOCKET=`lscpu | grep 'Socket(s)' | awk '{print $NF}'`
CORES_PER_SOCKET=`lscpu | grep 'Core(s) per socket' | awk '{print $NF}'`
NUM_CORES=$((CORES_PER_SOCKET * NUM_SOCKET))

integer_reg='^[0-9]+$'
if [[ $1 =~ $integer_reg ]] ; then
	if (($1 > $NUM_CORES)); then
		echo >&2
		echo "WARNING: given number of threads = $1" \
			" is greater than number of physical cores = $NUM_CORES." >&2
		echo >&2
	fi
	NUM_CORES=$1
	shift
	check_parametrs $@
fi

CORES={0}:${NUM_CORES}:1

INSTRUCTION="OMP_NUM_THREADS=${NUM_CORES} OMP_PROC_BIND=TRUE OMP_PLACES=${CORES} python3 -u $@"
echo $INSTRUCTION >&2
eval $INSTRUCTION
