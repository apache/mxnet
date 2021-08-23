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

SCRIPTDIR=$(realpath $(dirname $0))
TOPDIR=$(realpath ${SCRIPTDIR}/../../)
export LD_LIBRARY_PATH=${TOPDIR}/cmake-build-relwithdebinfo:${TOPDIR}/../cmake-build-relwithdebinfo/mxnet:$LD_LIBRARY_PATH
export PYTHONPATH=${TOPDIR}/python:$PYTHONPATH
cd ${TOPDIR}
/opt/intel/vtune_amplifier_xe/bin64/amplxe-cl -collect hotspots -run-pass-thru=-timestamp=sys -knob analyze-openmp=true -knob sampling-interval=1 -knob enable-user-tasks=true -- /usr/bin/python $@
