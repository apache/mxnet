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

usage()
{
    echo "usage: bash ./benchmark.sh [[[-p prefix ] [-e epoch] [-d dataset] [-b batch_size] [-i instance] [-c cores/instance]] | [-h]]"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --prefix | -p)
      shift
      PREFIX=$1
      ;;
    --epoch | -e)
      shift
      EPOCH=$1
      ;;
    --dataset | -d)
      shift
      DATASET=$1
      ;;
    --batch-size | -b)
      shift
      BS=$1
      ;;
    --instance | -i)
      shift
      INS=$1
      ;;
    --core | -c)
      shift
      CORES=$1
      ;;
    --help | -h)
      usage
      exit 1
      ;;
    *)
      usage
      exit 1
  esac
  shift
done

NUM_SOCKET=`lscpu | grep 'Socket(s)' | awk '{print $NF}'`
NUM_NUMA_NODE=`lscpu | grep 'NUMA node(s)' | awk '{print $NF}'`
CORES_PER_SOCKET=`lscpu | grep 'Core(s) per socket' | awk '{print $NF}'`
NUM_CORES=$((CORES_PER_SOCKET * NUM_SOCKET))
CORES_PER_NUMA=$((NUM_CORES / NUM_NUMA_NODE))
echo "target machine has $NUM_CORES physical core(s) on $NUM_NUMA_NODE numa nodes of $NUM_SOCKET socket(s)."

if [ -z $PREFIX ]; then
  echo "Error: Need a model prefix."
  exit
fi
if [ -z $EPOCH ]; then
  echo "Default: set epoch of model parameters to 7."
  EPOCH=7
fi
if [ -z $DATASET ]; then
  echo "Default: set dataset to ml-20m."
  DATASET='ml-20m'
fi
if [ -z $INS ]; then
  echo "Default: launch one instance per physical core."
  INS=$NUM_CORES
fi
if [ -z $CORES ]; then
  echo "Default: divide full physical cores."
  CORES=$((NUM_CORES / $INS))
fi
if [ -z $BS ]; then
  echo "Default: set batch size to 700."
  BS=700
fi

echo "  cores/instance: $CORES"
echo "  total instances: $INS"
echo "  batch size: $BS"
echo ""

rm NCF_*.log

for((i=0;i<$INS;i++));
do
  ((a=$i*$CORES))
  ((b=$a+$CORES-1))
  memid=$((b/CORES_PER_NUMA % NUM_NUMA_NODE))
  LOG=NCF_$i.log
  echo "  Instance $i use $a-$b cores with $LOG"
  KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0 \
  OMP_NUM_THREADS=$CORES \
  numactl --physcpubind=$a-$b --membind=$memid python ncf.py --batch-size=$BS --dataset=$DATASET --epoch=$EPOCH --benchmark --prefix=$PREFIX 2>&1 | tee $LOG &
done
wait

sps=`grep speed NCF_*.log | awk '{ sum += $(NF-1) }; END { print sum }'`
latency=$(awk "BEGIN {printf \"%.2f\", 1000*${BS}*${INS}/${sps}}")
echo "overall throughput (samples/sec): $sps"
echo "latency per batch per instance (ms): $latency"
echo "benchmark finish:)"
