#!/bin/sh

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
    echo "usage: bash ./launch_inference_mkldnn.sh [[[-s symbol_file ] [-b batch_size] [-iter iteraton] [-ins instance] [-c cores/instance]] | [-h]]"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --symbol | -s)
      shift
      SYMBOL=$1
      ;;
    --batch-size | -b)
      shift
      BS=$1
      ;;
    --iteration | -iter)
      shift
      ITERATIONS=$1
      ;;
    --instance | -ins)
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

if [ -z $SYMBOL ]; then
  echo "Error: Need a symbol file as input."
fi
if [ -z $INS ]; then
  echo "Default: launch one instance per socket."
  INS=$NUM_SOCKET
fi
if [ -z $CORES ]; then
  echo "Default: divide full physical cores."
  CORES=$((NUM_CORES / $INS))
fi
if [ -z $BS ]; then
  echo "Default: set batch size to 64."
  BS=64
fi
if [ -z $ITERATIONS ]; then
  echo "Default: set iterations to 500."
  ITERATIONS=500
fi

echo "  benchmark configs"
echo "  cores per instance: $CORES"
echo "  total instances: $INS"
echo "  batch size: $BS"
echo "  iterations: $ITERATIONS"
echo ""

rm BENCHMARK_*.log  || echo "benchmarking..."

for((i=0;i<$INS;i++));
do
  ((a=$i*$CORES))
  ((b=$a+$CORES-1))
  memid=$((b/CORES_PER_NUMA % NUM_NUMA_NODE))
  LOG=BENCHMARK_$i.log
  echo "  Instance $i use $a-$b cores and mem $memid with $LOG"
  KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0 \
  OMP_NUM_THREADS=$CORES \
  nohup numactl --physcpubind=$a-$b --membind=$memid python imagenet_inference.py --symbol-file=$SYMBOL --batch-size=$BS --num-inference-batches=$ITERATIONS --ctx=cpu --benchmark=True > $LOG 2>&1 &
done
wait

fps=`grep image/sec BENCHMARK_*.log | awk '{ sum += $(NF) }; END { print sum }'`
latency=$(awk "BEGIN {printf \"%.2f\", 1000*${BS}*${INS}/${fps}}")
echo "overall throughput (image/sec): $fps"
echo "latency per batch per instance (ms): $latency"
echo "benchmark finish:)"
