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

# set LD_LIBRARY_PATH
export CC  = gcc
export CXX = g++
export CFLAGS = -Wall -O3 
# specify tensor path
BIN = exp_lazy exp_template exp_template_op

.PHONY: clean all

all: $(BIN) 

exp_lazy: exp_lazy.cpp
exp_template: exp_template.cpp
exp_template_op: exp_template_op.cpp

$(BIN) :
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

clean:
	rm -rf $(BIN) *~
