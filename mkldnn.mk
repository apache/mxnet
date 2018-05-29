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
	MKLDNN_ROOTDIR = $(ROOTDIR)/3rdparty/mkldnn
	MKLDNN_BUILDDIR = $(MKLDNN_ROOTDIR)/build
	MKLDNN_INSTALLDIR = $(MKLDNN_ROOTDIR)/install
	MKLDNN_LIBDIR = $(ROOTDIR)/lib
ifeq ($(UNAME_S), Darwin)
	OMP_LIBFILE = $(MKLDNN_INSTALLDIR)/lib/libiomp5.dylib
	MKLML_LIBFILE = $(MKLDNN_INSTALLDIR)/lib/libmklml.dylib
	MKLDNN_LIBFILE = $(MKLDNN_INSTALLDIR)/lib/libmkldnn.0.dylib
else
	OMP_LIBFILE = $(MKLDNN_INSTALLDIR)/lib/libiomp5.so
	MKLML_LIBFILE = $(MKLDNN_INSTALLDIR)/lib/libmklml_intel.so
	MKLDNN_LIBFILE = $(MKLDNN_INSTALLDIR)/lib/libmkldnn.so.0
endif
endif

.PHONY: mkldnn mkldnn_clean mkldnn_lib_sync

mkldnn_build: $(MKLDNN_INSTALLDIR)/lib/libmkldnn.so 

$(MKLDNN_INSTALLDIR)/lib/libmkldnn.so:
	cd $(MKLDNN_ROOTDIR) && rm -rf external && cd scripts && ./prepare_mkl.sh >&2 && cd .. && cp -a external/*/* $(MKLDNN_INSTALLDIR)/.
	cmake $(MKLDNN_ROOTDIR) -DCMAKE_INSTALL_PREFIX=$(MKLDNN_INSTALLDIR) -B$(MKLDNN_BUILDDIR) -DARCH_OPT_FLAGS="-mtune=generic" -DWITH_TEST=OFF -DWITH_EXAMPLE=OFF >&2
	$(MAKE) -C $(MKLDNN_BUILDDIR) VERBOSE=1 >&2
	$(MAKE) -C $(MKLDNN_BUILDDIR) install >&2
	mkdir -p $(MKLDNN_LIBDIR)
	rsync -a $(OMP_LIBFILE) $(MKLDNN_LIBDIR)
	rsync -a $(MKLML_LIBFILE) $(MKLDNN_LIBDIR)
	rsync -a $(MKLDNN_LIBFILE) $(MKLDNN_LIBDIR)

mkldnn_lib_sync:
	mkdir -p $(MKLDNNROOT)
	rsync -a $(MKLDNN_INSTALLDIR)/include $(MKLDNN_INSTALLDIR)/lib $(MKLDNNROOT)/.

mkldnn_clean:
	$(RM) -r 3rdparty/mkldnn/build
	$(RM) -r 3rdparty/mkldnn/install/*

ifeq ($(USE_MKLDNN), 1)
ifeq ($(MKLDNNROOT), $(ROOTDIR)/3rdparty/mkldnn/install)
mkldnn: mkldnn_build
else
mkldnn: mkldnn_lib_sync
mkldnn_lib_sync: mkldnn_build
endif
else
mkldnn:
endif
