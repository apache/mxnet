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


# setup


print_help() {
	echo "The performance test runs train_imagenet script with following default values."
	echo "FLOATTYPE = float16"
	echo "BATCH_SIZE = 128"
	echo "STORE = nccl"
	echo "EPOCHS = 1"
	echo "LAYERS = 50"
	echo "NETWORK=resnet-v1"
	echo "By default, the script will run the train_imagenet script for incremental batch sizes and gpus."
	echo "For example, 128 batch size for 1 GPU, 256 for 2 GPUS, etc"
	echo
	echo "Following tunnable environment variables are supported"
	echo "MXNET_HOME: Path to mxnet source is mandatory."
	echo "LAYERS : Number of layers"
	echo "NETWORK : Network to be used."
	echo "STORE: kv-store parameter"
	echo "FIXED_GPUS: Run the test only for given number of gpus"
	echo "BATCH_SIZE_FIXED: Keep the batch size fixed."
}

while [ "$1" != "" ]; do
    case $1 in
        -h | --help )
		print_help
		shift
        exit
        ;;
    esac
done

if [ -z ${MXNET_HOME+x} ] ; then
    echo "MXNET_HOME is unset. Can not proceed without finding mxnet source"
    print_help
    exit
fi

export RESULT_FILE="Result-"$(date +%F-%H-%M-%S)".log"
exec &>> $RESULT_FILE


if [ -z ${TEST_FILE+x} ] ; then
    echo "TEST_FILE is unset. Using TEST_FILE = $MXNET_HOME/example/image-classification/train_imagenet.py"
    export TEST_FILE=${MXNET_HOME}/example/image-classification/train_imagenet.py
fi

if [ -z ${FLOATTYPE+x} ] ; then
    echo "FLOATTYPE is unset. Using FLOATTYPE=float16"
    export FLOATTYPE=float16
fi

if [ -z ${EPOCHS+x} ] ; then
    echo "EPOCHS is unset. Using EPOCHS=1"
    export EPOCHS=5
fi

if [ -z ${LAYERS+x} ] ; then
    echo "LAYERS is unset. Using LAYERS=50"
    export LAYERS=50
fi

if [ -z ${BATCH_SIZE+x} ] ; then
    echo "BATCH_SIZE is unset. Using BATCH_SIZE=128 for a single GPU"
    export BATCH_SIZE=128
fi

if [ -z ${BATCH_SIZE_FIXED+x} ]; then
    echo "BATCH_SIZE_FIXED is unset, BATCH_SIZE will increment by 2 with the number of GPUS"
    export BATCH_SIZE_FIXED=0
fi

if [ -z ${GPUS+x} ] ; then
    echo "GPUS is unset. Using GPUS=\"0\""
    export GPUS="0"
fi

if [ -z ${LOG_FOLDER+x} ] ; then
    echo "LOG_FOLDER is unset. Using LOG_FOLDER=temp"
    export LOG_FOLDER=temp
fi


if [ -z ${NUM_GPUS+x} ] ; then
    echo "NUM_GPUS is unset. Using NUM_GPUS=`nvidia-smi -L | grep -c ^GPU`"
    export NUM_GPUS=`nvidia-smi -L | grep -c ^GPU`
fi

if [ -z ${STORE+x} ] ; then
    echo "STORE is unset. Using STORE=nccl"
    export STORE=nccl
fi

if [ -z ${RESULT_FILE+x} ]; then
    echo "RESULT_FILE is unset. Using RESULT_FILE=`date +%F-%H-%M-%S`";
    export RESULT_FILE="Result-"$(date +%F-%H-%M-%S)".log"
fi

if [ -z ${NETWORK+x} ]; then
    export NETWORK=resnet-v1
fi

if [ -z ${FIXED_GPUS+x} ]; then
	export FIXED_GPUS=0
fi

echo
echo
echo

echo "Test parameters used....."
echo "MXNET_HOME=$MXNET_HOME"
echo "TEST_FILE=$TEST_FILE"
echo "FLOATTYPE=$FLOATTYPE"
echo "EPOCHS=$EPOCHS"
echo "LAYERS=$LAYERS"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "NUM_GPUS=$NUM_GPUS"
echo "STORE=$STORE"
echo "NETWORK=$NETWORK"
echo "BATCH_SIZE_FIXED=$BATCH_SIZE_FIXED"
echo "FIXED_GPUS=$FIXED_GPUS"

export OUTPUT_LOG="output_"${BATCH_SIZE}".log"

generateGpus() {
    i=1
    GPUS=0
    while [ ${i}  -lt $1 ]
    do
        GPUS=${GPUS}","${i}
        (( i++ ))
    done
}

runPerfCommand() {
    echo "************ $1 $2 $STORE ****************"
    finalCommand="python ${TEST_FILE} --data-nthreads 40 --disp-batches 20 --network ${NETWORK}  --max-random-shear-ratio 0 --min-random-scale 0.533 --max-random-rotate-angle 0 --max-random-h 0 --max-random-l 0 --max-random-s 0 --max-random-aspect-ratio 0 --benchmark 1 --gpu "$1" --num-epochs "${EPOCHS}" --num-layers "${LAYERS}" --dtype "${FLOATTYPE}" --batch-size "$2" --kv-store "${STORE}
    OUTPUT_LOG=`mktemp /tmp/output_${2}_final_${STORE}.XXX`
    echo $finalCommand
    $finalCommand &> ${OUTPUT_LOG}
    echo "GPUS $1 Batch Size $2 Epoch Size ${EPOCHS} KV-Store ${STORE}" >> ${RESULT_FILE}
    awk 'BEGIN{sum=0;count=0} { if (NF == 7) {sum=sum+$5;count=count+1}} END{ avg=sum/count;printf "Average Samples/Sec %d\n",avg}' ${OUTPUT_LOG} >> ${RESULT_FILE}
}

if [ $FIXED_GPUS -gt 0 ]; then
	echo "The performance test will be run for a given BATCH_SIZE=$BATCH_SIZE and number of GPUS=$FIXED_GPUS"
	if [ $FIXED_GPUS -gt $NUM_GPUS ]; then
		echo "The system has only $NUM_GPUS available"
	fi
	generateGpus $FIXED_GPUS
	runPerfCommand ${GPUS} ${BATCH_SIZE}
	exit 0
fi

CURRENT_GPUS=1
while [ $CURRENT_GPUS -le $NUM_GPUS ]
do
    generateGpus $CURRENT_GPUS
    if [ ${BATCH_SIZE_FIXED} -eq 0 ]; then
        NEW_BATCH_SIZE=$((BATCH_SIZE*$CURRENT_GPUS))
    else
        NEW_BATCH_SIZE=${BATCH_SIZE}
    fi
    runPerfCommand ${GPUS} ${NEW_BATCH_SIZE}

    CURRENT_GPUS=$((CURRENT_GPUS*2))
done

