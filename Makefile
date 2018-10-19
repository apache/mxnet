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

ROOTDIR = $(CURDIR)
TPARTYDIR = $(ROOTDIR)/3rdparty

SCALA_VERSION_PROFILE := scala-2.11

ifeq ($(OS),Windows_NT)
	UNAME_S := Windows
else
	UNAME_S := $(shell uname -s)
endif

ifndef config
ifdef CXXNET_CONFIG
	config = $(CXXNET_CONFIG)
else ifneq ("$(wildcard ./config.mk)","")
	config = config.mk
else
	config = make/config.mk
endif
endif

ifndef DMLC_CORE
	DMLC_CORE = $(TPARTYDIR)/dmlc-core
endif
CORE_INC = $(wildcard $(DMLC_CORE)/include/*/*.h)

ifndef NNVM_PATH
	NNVM_PATH = $(TPARTYDIR)/tvm/nnvm
endif

ifndef DLPACK_PATH
	DLPACK_PATH = $(ROOTDIR)/3rdparty/dlpack
endif

ifndef AMALGAMATION_PATH
	AMALGAMATION_PATH = $(ROOTDIR)/amalgamation
endif

ifneq ($(USE_OPENMP), 1)
	export NO_OPENMP = 1
endif

# use customized config file
include $(config)

ifeq ($(USE_MKL2017), 1)
$(warning "USE_MKL2017 is deprecated. We will switch to USE_MKLDNN.")
	USE_MKLDNN=1
endif

ifeq ($(USE_MKLDNN), 1)
	MKLDNNROOT = $(ROOTDIR)/3rdparty/mkldnn/build/install
	MKLROOT = $(ROOTDIR)/3rdparty/mkldnn/build/install
	export USE_MKLML = 1
endif

include $(TPARTYDIR)/mshadow/make/mshadow.mk
include $(DMLC_CORE)/make/dmlc.mk

# all tge possible warning tread
WARNFLAGS= -Wall -Wsign-compare
CFLAGS = -DMSHADOW_FORCE_STREAM $(WARNFLAGS)

ifeq ($(DEV), 1)
	CFLAGS += -g -Werror
	NVCCFLAGS += -Werror cross-execution-space-call
endif

# CFLAGS for debug
ifeq ($(DEBUG), 1)
	CFLAGS += -g -O0
else
	CFLAGS += -O3 -DNDEBUG=1
endif
CFLAGS += -I$(TPARTYDIR)/mshadow/ -I$(TPARTYDIR)/dmlc-core/include -fPIC -I$(NNVM_PATH)/include -I$(DLPACK_PATH)/include -I$(TPARTYDIR)/tvm/include -Iinclude $(MSHADOW_CFLAGS)
LDFLAGS = -pthread $(MSHADOW_LDFLAGS) $(DMLC_LDFLAGS)

ifeq ($(ENABLE_TESTCOVERAGE), 1)
        CFLAGS += --coverage
        LDFLAGS += --coverage
endif

ifeq ($(USE_TENSORRT), 1)
	CFLAGS +=  -I$(ROOTDIR) -I$(TPARTYDIR) -DONNX_NAMESPACE=$(ONNX_NAMESPACE) -DMXNET_USE_TENSORRT=1
	LDFLAGS += -lprotobuf -pthread -lonnx -lonnx_proto -lnvonnxparser -lnvonnxparser_runtime -lnvinfer -lnvinfer_plugin
endif
# -L/usr/local/lib

ifeq ($(DEBUG), 1)
	NVCCFLAGS += -std=c++11 -Xcompiler -D_FORCE_INLINES -g -G -O0 -ccbin $(CXX) $(MSHADOW_NVCCFLAGS)
else
	NVCCFLAGS += -std=c++11 -Xcompiler -D_FORCE_INLINES -O3 -ccbin $(CXX) $(MSHADOW_NVCCFLAGS)
endif

# CFLAGS for segfault logger
ifeq ($(USE_SIGNAL_HANDLER), 1)
	CFLAGS += -DMXNET_USE_SIGNAL_HANDLER=1
endif

# Caffe Plugin
ifdef CAFFE_PATH
	CFLAGS += -DMXNET_USE_CAFFE=1
endif

ifndef LINT_LANG
	LINT_LANG="all"
endif

ifeq ($(USE_MKLDNN), 1)
	CFLAGS += -DMXNET_USE_MKLDNN=1
	CFLAGS += -DUSE_MKL=1
	CFLAGS += -I$(ROOTDIR)/src/operator/nn/mkldnn/
	ifneq ($(MKLDNNROOT), $(MKLROOT))
		CFLAGS += -I$(MKLROOT)/include
		LDFLAGS += -L$(MKLROOT)/lib
	endif
	CFLAGS += -I$(MKLDNNROOT)/include
	LDFLAGS += -L$(MKLDNNROOT)/lib -lmkldnn -Wl,-rpath,'$${ORIGIN}'
endif

# setup opencv
ifeq ($(USE_OPENCV), 1)
	CFLAGS += -DMXNET_USE_OPENCV=1 $(shell pkg-config --cflags opencv)
	LDFLAGS += $(filter-out -lopencv_ts, $(shell pkg-config --libs opencv))
	BIN += bin/im2rec
else
	CFLAGS+= -DMXNET_USE_OPENCV=0
endif

ifeq ($(USE_OPENMP), 1)
	CFLAGS += -fopenmp
endif

ifeq ($(USE_NNPACK), 1)
	CFLAGS += -DMXNET_USE_NNPACK=1
	LDFLAGS += -lnnpack
endif

ifeq ($(USE_OPERATOR_TUNING), 1)
	CFLAGS += -DMXNET_USE_OPERATOR_TUNING=1
endif

# verify existence of separate lapack library when using blas/openblas/atlas
# switch off lapack support in case it can't be found
# issue covered with this
#   -  for Ubuntu 14.04 or lower, lapack is not automatically installed with openblas
#   -  for Ubuntu, installing atlas will not automatically install the atlas provided lapack library
#   -  for rhel7.2, try installing the package `lapack-static` via yum will dismiss this warning.
# silently switching lapack off instead of letting the build fail because of backward compatibility
ifeq ($(USE_LAPACK), 1)
ifeq ($(USE_BLAS),$(filter $(USE_BLAS),blas openblas atlas mkl))
ifeq (,$(wildcard $(USE_LAPACK_PATH)/liblapack.a))
ifeq (,$(wildcard $(USE_LAPACK_PATH)/liblapack.so))
ifeq (,$(wildcard /lib/liblapack.a))
ifeq (,$(wildcard /lib/liblapack.so))
ifeq (,$(wildcard /usr/lib/liblapack.a))
ifeq (,$(wildcard /usr/lib/liblapack.so))
ifeq (,$(wildcard /usr/lib64/liblapack.a))
ifeq (,$(wildcard /usr/lib64/liblapack.so))
	USE_LAPACK = 0
        $(warning "USE_LAPACK disabled because libraries were not found")
endif
endif
endif
endif
endif
endif
endif
endif
endif
endif

# lapack settings.
ifeq ($(USE_LAPACK), 1)
	ifneq ($(USE_LAPACK_PATH), )
		LDFLAGS += -L$(USE_LAPACK_PATH)
	endif
	ifeq ($(USE_BLAS),$(filter $(USE_BLAS),blas openblas atlas mkl))
		LDFLAGS += -llapack
	endif
	CFLAGS += -DMXNET_USE_LAPACK
endif

ifeq ($(USE_CUDNN), 1)
	CFLAGS += -DMSHADOW_USE_CUDNN=1
	LDFLAGS += -lcudnn
endif

# whether to use F16C instruction set extension for fast fp16 compute on CPU
# if cross compiling you may want to explicitly turn it off if target system does not support it
ifndef USE_F16C
    ifneq ($(OS),Windows_NT)
        detected_OS := $(shell uname -s)
        ifeq ($(detected_OS),Darwin)
            F16C_SUPP = $(shell sysctl -a | grep machdep.cpu.features | grep F16C)
        endif
        ifeq ($(detected_OS),Linux)
            F16C_SUPP = $(shell cat /proc/cpuinfo | grep flags | grep f16c)
        endif
	ifneq ($(strip $(F16C_SUPP)),)
                USE_F16C=1
        else
                USE_F16C=0
        endif
    endif
    # if OS is Windows, check if your processor and compiler support F16C architecture.
    # One way to check if processor supports it is to download the tool
    # https://docs.microsoft.com/en-us/sysinternals/downloads/coreinfo.
    # If coreinfo -c shows F16C and compiler supports it,
    # then you can set USE_F16C=1 explicitly to leverage that capability"
endif

# gperftools malloc library (tcmalloc)
ifeq ($(USE_GPERFTOOLS), 1)
FIND_LIBFILEEXT=so
ifeq ($(USE_GPERFTOOLS_STATIC), 1)
FIND_LIBFILEEXT=a
endif
FIND_LIBFILE=$(wildcard $(USE_GPERFTOOLS_PATH)/libtcmalloc.$(FIND_LIBFILEEXT))
ifeq (,$(FIND_LIBFILE))
FIND_LIBFILE=$(wildcard /lib/libtcmalloc.$(FIND_LIBFILEEXT))
ifeq (,$(FIND_LIBFILE))
FIND_LIBFILE=$(wildcard /usr/lib/libtcmalloc.$(FIND_LIBFILEEXT))
ifeq (,$(FIND_LIBFILE))
FIND_LIBFILE=$(wildcard /usr/local/lib/libtcmalloc.$(FIND_LIBFILEEXT))
ifeq (,$(FIND_LIBFILE))
FIND_LIBFILE=$(wildcard /usr/lib64/libtcmalloc.$(FIND_LIBFILEEXT))
ifeq (,$(FIND_LIBFILE))
	USE_GPERFTOOLS=0
endif
endif
endif
endif
endif
ifeq ($(USE_GPERFTOOLS), 1)
	CFLAGS += -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free
	LDFLAGS += $(FIND_LIBFILE)
endif

# jemalloc malloc library (if not using gperftools)
else
ifeq ($(USE_JEMALLOC), 1)
FIND_LIBFILEEXT=so
ifeq ($(USE_JEMALLOC_STATIC), 1)
FIND_LIBFILEEXT=a
endif
FIND_LIBFILE=$(wildcard $(USE_JEMALLOC_PATH)/libjemalloc.$(FIND_LIBFILEEXT))
ifeq (,$(FIND_LIBFILE))
FIND_LIBFILE=$(wildcard /lib/libjemalloc.$(FIND_LIBFILEEXT))
ifeq (,$(FIND_LIBFILE))
FIND_LIBFILE=$(wildcard /usr/lib/libjemalloc.$(FIND_LIBFILEEXT))
ifeq (,$(FIND_LIBFILE))
FIND_LIBFILE=$(wildcard /usr/local/lib/libjemalloc.$(FIND_LIBFILEEXT))
ifeq (,$(FIND_LIBFILE))
FIND_LIBFILE=$(wildcard /usr/lib/x86_64-linux-gnu/libjemalloc.$(FIND_LIBFILEEXT))
ifeq (,$(FIND_LIBFILE))
FIND_LIBFILE=$(wildcard /usr/lib64/libjemalloc.$(FIND_LIBFILEEXT))
ifeq (,$(FIND_LIBFILE))
	USE_JEMALLOC=0
endif
endif
endif
endif
endif
endif
ifeq ($(USE_JEMALLOC), 1)
	CFLAGS += -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc \
	-fno-builtin-free -DUSE_JEMALLOC
	LDFLAGS += $(FIND_LIBFILE)
endif
endif
endif

# If not using tcmalloc or jemalloc, print a warning (user should consider installing)
ifneq ($(USE_GPERFTOOLS), 1)
ifneq ($(USE_JEMALLOC), 1)
$(warning WARNING: Significant performance increases can be achieved by installing and \
enabling gperftools or jemalloc development packages)
endif
endif

ifeq ($(USE_THREADED_ENGINE), 1)
	CFLAGS += -DMXNET_USE_THREADED_ENGINE
endif

ifneq ($(ADD_CFLAGS), NONE)
	CFLAGS += $(ADD_CFLAGS)
endif

ifneq ($(ADD_LDFLAGS), NONE)
	LDFLAGS += $(ADD_LDFLAGS)
endif

ifeq ($(NVCC), NONE)
	# If NVCC has not been manually defined, use the CUDA_PATH bin dir.
	ifneq ($(USE_CUDA_PATH), NONE)
		NVCC=$(USE_CUDA_PATH)/bin/nvcc
	endif
endif

# Guard against displaying nvcc info messages to users not using CUDA.
ifeq ($(USE_CUDA), 1)
	# If NVCC is not at the location specified, use CUDA_PATH instead.
	ifeq ("$(wildcard $(NVCC))","")
		ifneq ($(USE_CUDA_PATH), NONE)
			NVCC=$(USE_CUDA_PATH)/bin/nvcc
$(info INFO: nvcc was not found on your path)
$(info INFO: Using $(NVCC) as nvcc path)
		else
$(warning WARNING: could not find nvcc compiler, the specified path was: $(NVCC))
		endif
	endif
endif

# Sets 'CUDA_ARCH', which determines the GPU architectures supported
# by the compiled kernels.  Users can edit the KNOWN_CUDA_ARCHS list below
# to remove archs they don't wish to support to speed compilation, or they can
# pre-set the CUDA_ARCH args in config.mk to a non-null value for full control.
#
# For archs in this list, nvcc will create a fat-binary that will include
# the binaries (SASS) for all architectures supported by the installed version
# of the cuda toolkit, plus the assembly (PTX) for the most recent such architecture.
# If these kernels are then run on a newer-architecture GPU, the binary will
# be JIT-compiled by the updated driver from the included PTX.
ifeq ($(USE_CUDA), 1)
ifeq ($(CUDA_ARCH),)
	KNOWN_CUDA_ARCHS := 30 35 50 52 60 61 70
	# Run nvcc on a zero-length file to check architecture-level support.
	# Create args to include SASS in the fat binary for supported levels.
	CUDA_ARCH := $(foreach arch,$(KNOWN_CUDA_ARCHS), \
				$(shell $(NVCC) -arch=sm_$(arch) -E --x cu /dev/null >/dev/null 2>&1 && \
						echo -gencode arch=compute_$(arch),code=sm_$(arch)))
	# Convert a trailing "code=sm_NN" to "code=[sm_NN,compute_NN]" to also
	# include the PTX of the most recent arch in the fat-binaries for
	# forward compatibility with newer GPUs.
	CUDA_ARCH := $(shell echo $(CUDA_ARCH) | sed 's/sm_\([0-9]*\)$$/[sm_\1,compute_\1]/')
	# Add fat binary compression if supported by nvcc.
	COMPRESS := --fatbin-options -compress-all
	CUDA_ARCH += $(shell $(NVCC) -cuda $(COMPRESS) --x cu /dev/null -o /dev/null >/dev/null 2>&1 && \
						 echo $(COMPRESS))
endif
$(info Running CUDA_ARCH: $(CUDA_ARCH))
endif

# ps-lite
PS_PATH=$(ROOTDIR)/3rdparty/ps-lite
DEPS_PATH=$(shell pwd)/deps
include $(PS_PATH)/make/ps.mk
ifeq ($(USE_DIST_KVSTORE), 1)
	CFLAGS += -DMXNET_USE_DIST_KVSTORE -I$(PS_PATH)/include -I$(DEPS_PATH)/include
	LIB_DEP += $(PS_PATH)/build/libps.a
	LDFLAGS += $(PS_LDFLAGS_A)
endif

.PHONY: clean all extra-packages test lint docs clean_all rcpplint rcppexport roxygen\
	cython2 cython3 cython cyclean

all: lib/libmxnet.a lib/libmxnet.so $(BIN) extra-packages

SRC = $(wildcard src/*/*/*/*.cc src/*/*/*.cc src/*/*.cc src/*.cc)
OBJ = $(patsubst %.cc, build/%.o, $(SRC))
CUSRC = $(wildcard src/*/*/*/*.cu src/*/*/*.cu src/*/*.cu src/*.cu)
CUOBJ = $(patsubst %.cu, build/%_gpu.o, $(CUSRC))

# extra operators
ifneq ($(EXTRA_OPERATORS),)
	EXTRA_SRC = $(wildcard $(patsubst %, %/*.cc, $(EXTRA_OPERATORS)) $(patsubst %, %/*/*.cc, $(EXTRA_OPERATORS)))
	EXTRA_OBJ = $(patsubst %.cc, %.o, $(EXTRA_SRC))
	EXTRA_CUSRC = $(wildcard $(patsubst %, %/*.cu, $(EXTRA_OPERATORS)) $(patsubst %, %/*/*.cu, $(EXTRA_OPERATORS)))
	EXTRA_CUOBJ = $(patsubst %.cu, %_gpu.o, $(EXTRA_CUSRC))
else
	EXTRA_SRC =
	EXTRA_OBJ =
	EXTRA_CUSRC =
	EXTRA_CUOBJ =
endif

# plugin
PLUGIN_OBJ =
PLUGIN_CUOBJ =
include $(MXNET_PLUGINS)

ifeq ($(UNAME_S), Windows)
	# TODO(yizhi) currently scala package does not support windows
	SCALA_PKG_PROFILE := windows
else
	ifeq ($(UNAME_S), Darwin)
		WHOLE_ARCH= -all_load
		NO_WHOLE_ARCH= -noall_load
		SCALA_PKG_PROFILE := osx-x86_64
	else
		WHOLE_ARCH= --whole-archive
		NO_WHOLE_ARCH= --no-whole-archive
		SCALA_PKG_PROFILE := linux-x86_64
	endif
endif

# all dep
LIB_DEP += $(DMLC_CORE)/libdmlc.a $(NNVM_PATH)/lib/libnnvm.a
ALL_DEP = $(OBJ) $(EXTRA_OBJ) $(PLUGIN_OBJ) $(LIB_DEP)

ifeq ($(USE_CUDA), 1)
	CFLAGS += -I$(ROOTDIR)/3rdparty/cub
	ALL_DEP += $(CUOBJ) $(EXTRA_CUOBJ) $(PLUGIN_CUOBJ)
	LDFLAGS += -lcufft
	ifeq ($(ENABLE_CUDA_RTC), 1)
		LDFLAGS += -lcuda -lnvrtc
		CFLAGS += -DMXNET_ENABLE_CUDA_RTC=1
	endif
	# Make sure to add stubs as fallback in order to be able to build
	# without full CUDA install (especially if run without nvidia-docker)
	LDFLAGS += -L/usr/local/cuda/lib64/stubs
	SCALA_PKG_PROFILE := $(SCALA_PKG_PROFILE)-gpu
	ifeq ($(USE_NCCL), 1)
		ifneq ($(USE_NCCL_PATH), NONE)
			CFLAGS += -I$(USE_NCCL_PATH)/include
			LDFLAGS += -L$(USE_NCCL_PATH)/lib
		endif
		LDFLAGS += -lnccl
		CFLAGS += -DMXNET_USE_NCCL=1
	else
		CFLAGS += -DMXNET_USE_NCCL=0
	endif
else
	SCALA_PKG_PROFILE := $(SCALA_PKG_PROFILE)-cpu
	CFLAGS += -DMXNET_USE_NCCL=0
endif

ifeq ($(USE_LIBJPEG_TURBO), 1)
	ifneq ($(USE_LIBJPEG_TURBO_PATH), NONE)
		CFLAGS += -I$(USE_LIBJPEG_TURBO_PATH)/include
		LDFLAGS += -L$(USE_LIBJPEG_TURBO_PATH)/lib
	endif
	LDFLAGS += -lturbojpeg
	CFLAGS += -DMXNET_USE_LIBJPEG_TURBO=1
else
	CFLAGS += -DMXNET_USE_LIBJPEG_TURBO=0
endif

# For quick compile test, used smaller subset
ALLX_DEP= $(ALL_DEP)

build/src/%.o: src/%.cc | mkldnn
	@mkdir -p $(@D)
	$(CXX) -std=c++11 -c $(CFLAGS) -MMD -c $< -o $@

build/src/%_gpu.o: src/%.cu | mkldnn
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -Xcompiler "$(CFLAGS)" -M -MT build/src/$*_gpu.o $< >build/src/$*_gpu.d
	$(NVCC) -c -o $@ $(NVCCFLAGS) $(CUDA_ARCH) -Xcompiler "$(CFLAGS)" $<

# A nvcc bug cause it to generate "generic/xxx.h" dependencies from torch headers.
# Use CXX to generate dependency instead.
build/plugin/%_gpu.o: plugin/%.cu
	@mkdir -p $(@D)
	$(CXX) -std=c++11 $(CFLAGS) -MM -MT build/plugin/$*_gpu.o $< >build/plugin/$*_gpu.d
	$(NVCC) -c -o $@ $(NVCCFLAGS) $(CUDA_ARCH) -Xcompiler "$(CFLAGS)" $<

build/plugin/%.o: plugin/%.cc
	@mkdir -p $(@D)
	$(CXX) -std=c++11 -c $(CFLAGS) -MMD -c $< -o $@

%_gpu.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -Xcompiler "$(CFLAGS) -Isrc/operator" -M -MT $*_gpu.o $< >$*_gpu.d
	$(NVCC) -c -o $@ $(NVCCFLAGS) $(CUDA_ARCH) -Xcompiler "$(CFLAGS) -Isrc/operator" $<

%.o: %.cc $(CORE_INC)
	@mkdir -p $(@D)
	$(CXX) -std=c++11 -c $(CFLAGS) -MMD -Isrc/operator -c $< -o $@

# NOTE: to statically link libmxnet.a we need the option
# --Wl,--whole-archive -lmxnet --Wl,--no-whole-archive
lib/libmxnet.a: $(ALLX_DEP)
	@mkdir -p $(@D)
	ar crv $@ $(filter %.o, $?)

lib/libmxnet.so: $(ALLX_DEP)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -shared -o $@ $(filter-out %libnnvm.a, $(filter %.o %.a, $^)) $(LDFLAGS) \
	-Wl,${WHOLE_ARCH} $(filter %libnnvm.a, $^) -Wl,${NO_WHOLE_ARCH}
ifeq ($(USE_MKLDNN), 1)
ifeq ($(UNAME_S), Darwin)
	install_name_tool -change '@rpath/libmklml.dylib' '@loader_path/libmklml.dylib' $@
	install_name_tool -change '@rpath/libiomp5.dylib' '@loader_path/libiomp5.dylib' $@
	install_name_tool -change '@rpath/libmkldnn.0.dylib' '@loader_path/libmkldnn.0.dylib' $@
endif
endif

$(PS_PATH)/build/libps.a: PSLITE

PSLITE:
	$(MAKE) CXX="$(CXX)" DEPS_PATH="$(DEPS_PATH)" -C $(PS_PATH) ps

$(DMLC_CORE)/libdmlc.a: DMLCCORE

DMLCCORE:
	+ cd $(DMLC_CORE); $(MAKE) libdmlc.a USE_SSE=$(USE_SSE) config=$(ROOTDIR)/$(config); cd $(ROOTDIR)

NNVM_INC = $(wildcard $(NNVM_PATH)/include/*/*.h)
NNVM_SRC = $(wildcard $(NNVM_PATH)/src/*/*/*.cc $(NNVM_PATH)/src/*/*.cc $(NNVM_PATH)/src/*.cc)
$(NNVM_PATH)/lib/libnnvm.a: $(NNVM_INC) $(NNVM_SRC)
	+ cd $(NNVM_PATH); $(MAKE) lib/libnnvm.a DMLC_CORE_PATH=$(DMLC_CORE); cd $(ROOTDIR)

bin/im2rec: tools/im2rec.cc $(ALLX_DEP)

$(BIN) :
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -std=c++11  -o $@ $(filter %.cpp %.o %.c %.a %.cc, $^) $(LDFLAGS)

# CPP Package
ifeq ($(USE_CPP_PACKAGE), 1)
include cpp-package/cpp-package.mk
endif

include mkldnn.mk
include tests/cpp/unittest.mk

extra-packages: $(EXTRA_PACKAGES)

test: $(TEST)

lint: cpplint rcpplint jnilint pylint

cpplint:
	3rdparty/dmlc-core/scripts/lint.py mxnet cpp include src plugin cpp-package tests \
	--exclude_path src/operator/contrib/ctc_include

pylint:
	pylint --rcfile=$(ROOTDIR)/ci/other/pylintrc --ignore-patterns=".*\.so$$,.*\.dll$$,.*\.dylib$$" python/mxnet tools/caffe_converter/*.py

doc: docs

docs:
	make -C docs html

clean_docs:
	make -C docs clean

doxygen:
	doxygen docs/Doxyfile

# Cython build
cython:
	cd python; python setup.py build_ext --inplace --with-cython

cython2:
	cd python; python2 setup.py build_ext --inplace --with-cython

cython3:
	cd python; python3 setup.py build_ext --inplace --with-cython

cyclean:
	rm -rf python/mxnet/*/*.so python/mxnet/*/*.cpp

# R related shortcuts
rcpplint:
	3rdparty/dmlc-core/scripts/lint.py mxnet-rcpp ${LINT_LANG} R-package/src

rpkg:
	mkdir -p R-package/inst/libs
	cp src/io/image_recordio.h R-package/src
	cp -rf lib/libmxnet.so R-package/inst/libs
	mkdir -p R-package/inst/include
	cp -rf include/* R-package/inst/include
	cp -rf 3rdparty/dmlc-core/include/* R-package/inst/include/
	cp -rf 3rdparty/tvm/nnvm/include/* R-package/inst/include
	Rscript -e "if(!require(devtools)){install.packages('devtools', repo = 'https://cloud.r-project.org/')}"
	Rscript -e "library(devtools); library(methods); options(repos=c(CRAN='https://cloud.r-project.org/')); install_deps(pkg='R-package', dependencies = TRUE)"
	echo "import(Rcpp)" > R-package/NAMESPACE
	echo "import(methods)" >> R-package/NAMESPACE
	R CMD INSTALL R-package
	Rscript -e "require(mxnet); mxnet:::mxnet.export('R-package')"
	Rscript -e "if (!require('roxygen2')||packageVersion('roxygen2')!= '5.0.1'){\
	devtools::install_version('roxygen2',version='5.0.1',\
	repo='https://cloud.r-project.org/',quiet=TRUE)}"
	Rscript -e "require(roxygen2); roxygen2::roxygenise('R-package')"
	R CMD INSTALL R-package

rpkgtest:
	Rscript -e 'require(testthat);res<-test_dir("R-package/tests/testthat");if(!testthat:::all_passed(res)){stop("Test failures", call. = FALSE)}'
	Rscript -e 'res<-covr:::package_coverage("R-package");fileConn<-file("r-package_coverage.json");writeLines(covr:::to_codecov(res), fileConn);close(fileConn)'

scalaclean:
	(cd $(ROOTDIR)/scala-package; \
		mvn clean -P$(SCALA_PKG_PROFILE),$(SCALA_VERSION_PROFILE))

scalapkg:
	(cd $(ROOTDIR)/scala-package; \
		mvn package -P$(SCALA_PKG_PROFILE),$(SCALA_VERSION_PROFILE) -Dcxx="$(CXX)" \
		    -Dbuild.platform="$(SCALA_PKG_PROFILE)" \
			-Dcflags="$(CFLAGS)" -Dldflags="$(LDFLAGS)" \
			-Dcurrent_libdir="$(ROOTDIR)/lib" \
			-Dlddeps="$(LIB_DEP) $(ROOTDIR)/lib/libmxnet.a")

scalaunittest:
	(cd $(ROOTDIR)/scala-package; \
		mvn integration-test -P$(SCALA_PKG_PROFILE),$(SCALA_VERSION_PROFILE),unittest -Dcxx="$(CXX)" \
			-Dcflags="$(CFLAGS)" -Dldflags="$(LDFLAGS)" \
			-Dlddeps="$(LIB_DEP) $(ROOTDIR)/lib/libmxnet.a" $(SCALA_TEST_ARGS))

scalaintegrationtest:
	(cd $(ROOTDIR)/scala-package; \
		mvn integration-test -P$(SCALA_PKG_PROFILE),$(SCALA_VERSION_PROFILE),integrationtest -Dcxx="$(CXX)" \
			-Dcflags="$(CFLAGS)" -Dldflags="$(LDFLAGS)" \
			-Dlddeps="$(LIB_DEP) $(ROOTDIR)/lib/libmxnet.a" $(SCALA_TEST_ARGS))

scalainstall:
	(cd $(ROOTDIR)/scala-package; \
		mvn install -P$(SCALA_PKG_PROFILE),$(SCALA_VERSION_PROFILE) -DskipTests=true -Dcxx="$(CXX)" \
		    -Dbuild.platform="$(SCALA_PKG_PROFILE)" \
			-Dcflags="$(CFLAGS)" -Dldflags="$(LDFLAGS)" \
			-Dlddeps="$(LIB_DEP) $(ROOTDIR)/lib/libmxnet.a")

scalarelease-dryrun:
	(cd $(ROOTDIR)/scala-package; \
		mvn release:clean release:prepare -DdryRun=true -DautoVersionSubmodules=true \
		-Papache-release,$(SCALA_PKG_PROFILE),$(SCALA_VERSION_PROFILE) \
		-Darguments=""-Dbuild\.platform=\""$(SCALA_PKG_PROFILE)\""\ -DskipTests=true\ -Dcflags=\""$(CFLAGS)\""\ -Dcxx=\""$(CXX)\""\ -Dldflags=\""$(LDFLAGS)\""\ -Dlddeps=\""$(LIB_DEP) $(ROOTDIR)/lib/libmxnet.a\"""")

scalarelease-prepare:
	(cd $(ROOTDIR)/scala-package; \
		mvn release:clean release:prepare -DautoVersionSubmodules=true \
		-Papache-release,$(SCALA_PKG_PROFILE),$(SCALA_VERSION_PROFILE) \
		-Darguments=""-Dbuild\.platform=\""$(SCALA_PKG_PROFILE)\""\ -DskipTests=true\ -Dcflags=\""$(CFLAGS)\""\ -Dcxx=\""$(CXX)\""\ -Dldflags=\""$(LDFLAGS)\""\ -Dlddeps=\""$(LIB_DEP) $(ROOTDIR)/lib/libmxnet.a\"""")

scalarelease-perform:
	(cd $(ROOTDIR)/scala-package; \
		mvn release:perform -DautoVersionSubmodules=true \
		-Papache-release,$(SCALA_PKG_PROFILE),$(SCALA_VERSION_PROFILE) \
		-Darguments=""-Dbuild\.platform=\""$(SCALA_PKG_PROFILE)\""\ -DskipTests=true\ -Dcflags=\""$(CFLAGS)\""\ -Dcxx=\""$(CXX)\""\ -Dldflags=\""$(LDFLAGS)\""\ -Dlddeps=\""$(LIB_DEP) $(ROOTDIR)/lib/libmxnet.a\"""")

scaladeploy:
	(cd $(ROOTDIR)/scala-package; \
		mvn deploy -Papache-release,$(SCALA_PKG_PROFILE),$(SCALA_VERSION_PROFILE) \-DskipTests=true -Dcxx="$(CXX)" \
		    -Dbuild.platform="$(SCALA_PKG_PROFILE)" \
			-Dcflags="$(CFLAGS)" -Dldflags="$(LDFLAGS)" \
			-Dlddeps="$(LIB_DEP) $(ROOTDIR)/lib/libmxnet.a")

jnilint:
	3rdparty/dmlc-core/scripts/lint.py mxnet-jnicpp cpp scala-package/native/src

ifneq ($(EXTRA_OPERATORS),)
clean: cyclean $(EXTRA_PACKAGES_CLEAN)
	$(RM) -r build lib bin *~ */*~ */*/*~ */*/*/*~ R-package/NAMESPACE R-package/man R-package/R/mxnet_generated.R \
		R-package/inst R-package/src/*.o R-package/src/*.so mxnet_*.tar.gz
	cd $(DMLC_CORE); $(MAKE) clean; cd -
	cd $(PS_PATH); $(MAKE) clean; cd -
	cd $(NNVM_PATH); $(MAKE) clean; cd -
	cd $(AMALGAMATION_PATH); $(MAKE) clean; cd -
	$(RM) -r  $(patsubst %, %/*.d, $(EXTRA_OPERATORS)) $(patsubst %, %/*/*.d, $(EXTRA_OPERATORS))
	$(RM) -r  $(patsubst %, %/*.o, $(EXTRA_OPERATORS)) $(patsubst %, %/*/*.o, $(EXTRA_OPERATORS))
else
clean: mkldnn_clean cyclean testclean $(EXTRA_PACKAGES_CLEAN)
	$(RM) -r build lib bin *~ */*~ */*/*~ */*/*/*~ R-package/NAMESPACE R-package/man R-package/R/mxnet_generated.R \
		R-package/inst R-package/src/image_recordio.h R-package/src/*.o R-package/src/*.so mxnet_*.tar.gz
	cd $(DMLC_CORE); $(MAKE) clean; cd -
	cd $(PS_PATH); $(MAKE) clean; cd -
	cd $(NNVM_PATH); $(MAKE) clean; cd -
	cd $(AMALGAMATION_PATH); $(MAKE) clean; cd -
endif

clean_all: clean

-include build/*.d
-include build/*/*.d
-include build/*/*/*.d
-include build/*/*/*/*.d
ifneq ($(EXTRA_OPERATORS),)
	-include $(patsubst %, %/*.d, $(EXTRA_OPERATORS)) $(patsubst %, %/*/*.d, $(EXTRA_OPERATORS))
endif
