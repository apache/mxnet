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

CFLAGS += -I$(TORCH_PATH)/install/include -I$(TORCH_PATH)/install/include/TH -I$(TORCH_PATH)/install/include/THC/ -DMXNET_USE_TORCH=1
LDFLAGS += -L$(TORCH_PATH)/install/lib -lluajit -lluaT -lTH -lTHC

TORCH_SRC = $(wildcard plugin/torch/*.cc)
PLUGIN_OBJ += $(patsubst %.cc, build/%.o, $(TORCH_SRC))
TORCH_CUSRC = $(wildcard plugin/torch/*.cu)
PLUGIN_CUOBJ += $(patsubst %.cu, build/%_gpu.o, $(TORCH_CUSRC))
