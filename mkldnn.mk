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

ifeq ($(USE_MKLDNN), 1)
	MKLDNN_SUBMODDIR = $(ROOTDIR)/3rdparty/mkldnn
	MKLDNN_BUILDDIR = $(MKLDNN_SUBMODDIR)/build
	MXNET_LIBDIR = $(ROOTDIR)/lib
	MXNET_INCLDIR = $(ROOTDIR)/include
	MKLDNN_LIBFILE = $(MKLDNNROOT)/lib/libdnnl.a
endif

mkldnn_FLAGS  = -DCMAKE_INSTALL_PREFIX=$(MKLDNNROOT)
mkldnn_FLAGS += -DCMAKE_INSTALL_LIBDIR=lib
mkldnn_FLAGS += -B$(MKLDNN_BUILDDIR)
mkldnn_FLAGS += -DMKLDNN_ARCH_OPT_FLAGS=""
mkldnn_FLAGS += -DMKLDNN_BUILD_TESTS=OFF
mkldnn_FLAGS += -DMKLDNN_BUILD_EXAMPLES=OFF
mkldnn_FLAGS += -DMKLDNN_ENABLE_JIT_PROFILING=OFF
mkldnn_FLAGS += -DMKLDNN_LIBRARY_TYPE=STATIC
mkldnn_FLAGS += -DDNNL_ENABLE_CONCURRENT_EXEC=ON

ifneq ($(USE_OPENMP), 1)
	mkldnn_FLAGS += -DMKLDNN_CPU_RUNTIME=SEQ
endif

ifeq ($(DEBUG), 1)
	mkldnn_FLAGS += -DCMAKE_BUILD_TYPE=Debug
endif

.PHONY: mkldnn mkldnn_clean

mkldnn_build: $(MKLDNN_LIBFILE)

$(MKLDNN_LIBFILE):
	mkdir -p $(MKLDNNROOT)/lib
	cmake $(MKLDNN_SUBMODDIR) $(mkldnn_FLAGS)
	$(MAKE) -C $(MKLDNN_BUILDDIR) VERBOSE=1
	$(MAKE) -C $(MKLDNN_BUILDDIR) install
	cp $(MKLDNN_BUILDDIR)/include/dnnl_version.h $(MXNET_INCLDIR)/mkldnn/.
	cp $(MKLDNN_BUILDDIR)/include/dnnl_config.h $(MXNET_INCLDIR)/mkldnn/.

mkldnn_clean:
	$(RM) -r 3rdparty/mkldnn/build
	$(RM) -r include/mkldnn/dnnl_version.h
	$(RM) -r include/mkldnn/dnnl_config.h

ifeq ($(USE_MKLDNN), 1)
mkldnn: mkldnn_build
else
mkldnn:
endif
