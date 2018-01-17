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

CFLAGS += -I$(CAFFE_PATH)/include -I$(CAFFE_PATH)/build/src -I$(CAFFE_PATH)/build/include
LDFLAGS += -lprotobuf -lboost_system -lboost_thread -lboost_filesystem -lgflags -lglog -L$(CAFFE_PATH)/build/lib -lcaffe

ifeq ($(USE_CUDNN), 1)
	CFLAGS += -DUSE_CUDNN=1
endif

ifeq ($(USE_CUDA), 0)
	CFLAGS += -DCPU_ONLY=1
endif

CAFFE_SRC = $(wildcard plugin/caffe/*.cc)
PLUGIN_OBJ += $(patsubst %.cc, build/%.o, $(CAFFE_SRC))
CAFFE_CUSRC = $(wildcard plugin/caffe/*.cu)
PLUGIN_CUOBJ += $(patsubst %.cu, build/%_gpu.o, $(CAFFE_CUSRC))
