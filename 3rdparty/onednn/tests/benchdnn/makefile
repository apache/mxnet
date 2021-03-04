#===============================================================================
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

CXX = g++
LD = g++

CXXFLAGS = -std=c++11 -g -O0 -fopenmp
LDFLAGS = -liomp5 -lrt -lpthread -ldl -lm

DNNL_ROOT ?= ../../.install
INC = -I$(DNNL_ROOT)/include -I.

build_dir = .build

srcs = $(wildcard *.cpp */*.cpp */*/*.cpp)
hdrs = $(wildcard *.hpp */*.hpp */*/*.hpp)
objs = $(addprefix $(build_dir)/,$(subst .cpp,.o,$(srcs)))
dirs = $(dir $(objs))

all: benchdnn

benchdnn: $(objs)
	$(LD) $^ -o $@ \
		-L$(DNNL_ROOT)/lib -Wl,-rpath=$(DNNL_ROOT)/lib -ldnnl \
		$(LDFLAGS)

.build/%.o: %.cpp $(hdrs) directories
	$(CXX) $(CXXFLAGS) $(DEFS) $(INC) $< -c -o $@

.PHONY: clean directories
directories: ; @mkdir -p $(dirs)
clean: ; -rm -rf $(build_dir) benchdnn
