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


CFLAGS=-std=c++17 -g -Wno-unknown-pragmas -Wall -DMXNET_USE_CUDA=1 -DMXNET_USE_CUDNN=1 -DMXNET_USE_ONEDNN=1

export MXNET_ROOT = `pwd`/../..

CFLAGS += `pkg-config --cflags opencv`
LDFLAGS += `pkg-config --libs opencv`

ifndef USE_CUDA_PATH
    export USE_CUDA_PATH = /usr/local/cuda
endif

ifndef ONEDNN_BUILD_DIR
    export ONEDNN_BUILD_DIR = $(MXNET_ROOT)/3rdparty/onednn/build
    # Cmake build path by default
    # Uncomment below line for CMake build
    #export ONEDNN_BUILD_DIR = $(MXNET_ROOT)/build/3rdparty/onednn
endif

ifndef ONEDNN_INCLUDE_DIR
    export ONEDNN_INCLUDE_DIR = $(MXNET_ROOT)/3rdparty/onednn/include
    # Cmake build path by default
    # Uncomment below line for CMake build
    #export ONEDNN_INCLUDE_DIR = $(MXNET_ROOT)/3rdparty/onednn/include
endif

CFLAGS += -I$(MXNET_ROOT)/include -I$(USE_CUDA_PATH)/include -I$(ONEDNN_INCLUDE_DIR) -I$(ONEDNN_BUILD_DIR)/include

# If MXNET_LIB_DIR env variable set use that, otherwise defaults to MXNET_ROOT/build
ifndef MXNET_LIB_DIR
    MXNET_LIB_DIR=$(MXNET_ROOT)/lib
    # Uncomment below line for CMake build
    #MXNET_LIB_DIR=$(MXNET_ROOT)/build
endif
LDFLAGS += $(MXNET_LIB_DIR)/libmxnet.so -lpthread -L$(ONEDNN_BUILD_DIR)/src -lmkldnn -Wl,-rpath,'$${ORIGIN}'

multi_threaded_inference: multi_threaded_inference.o
	g++ -O3 -o multi_threaded_inference multi_threaded_inference.o $(LDFLAGS)

multi_threaded_inference.o: multi_threaded_inference.cc
	g++ -O3 -c multi_threaded_inference.cc $(CFLAGS)

clean:
	rm multi_threaded_inference
	rm -rf *.d *.o

lint:
	python ../../../3rdparty/dmlc-core/scripts/lint.py mxnet "cpp" ./
