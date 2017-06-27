ROOTDIR = $(CURDIR)

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
	DMLC_CORE = $(ROOTDIR)/dmlc-core
endif

ifndef NNVM_PATH
	NNVM_PATH = $(ROOTDIR)/nnvm
endif

ifndef DLPACK_PATH
	DLPACK_PATH = $(ROOTDIR)/dlpack
endif

ifneq ($(USE_OPENMP), 1)
	export NO_OPENMP = 1
endif

# use customized config file
include $(config)

ifeq ($(USE_MKL2017), 1)
# must run ./prepare_mkl before including mshadow.mk
	RETURN_STRING = $(shell ./prepare_mkl.sh $(MKLML_ROOT))
	MKLROOT = $(firstword $(RETURN_STRING))
	export USE_MKLML = $(lastword $(RETURN_STRING))
endif

include mshadow/make/mshadow.mk
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
CFLAGS += -I$(ROOTDIR)/mshadow/ -I$(ROOTDIR)/dmlc-core/include -fPIC -I$(NNVM_PATH)/include -I$(DLPACK_PATH)/include -Iinclude $(MSHADOW_CFLAGS)
LDFLAGS = -pthread $(MSHADOW_LDFLAGS) $(DMLC_LDFLAGS)
ifeq ($(DEBUG), 1)
	NVCCFLAGS += -std=c++11 -Xcompiler -D_FORCE_INLINES -g -G -O0 -ccbin $(CXX) $(MSHADOW_NVCCFLAGS)
else
	NVCCFLAGS += -std=c++11 -Xcompiler -D_FORCE_INLINES -O3 -ccbin $(CXX) $(MSHADOW_NVCCFLAGS)
endif

# CFLAGS for profiler
ifeq ($(USE_PROFILER), 1)
	CFLAGS += -DMXNET_USE_PROFILER=1
endif

# Caffe Plugin
ifdef CAFFE_PATH
  CFLAGS += -DMXNET_USE_CAFFE=1
endif

ifndef LINT_LANG
	LINT_LANG="all"
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

ifeq ($(USE_MKL2017), 1)
	CFLAGS += -DMXNET_USE_MKL2017=1
	CFLAGS += -DUSE_MKL=1
	CFLAGS += -I$(ROOTDIR)/src/operator/mkl/
	CFLAGS += -I$(MKLML_ROOT)/include
	LDFLAGS += -L$(MKLML_ROOT)/lib
ifeq ($(USE_MKL2017_EXPERIMENTAL), 1)
	CFLAGS += -DMKL_EXPERIMENTAL=1
else
	CFLAGS += -DMKL_EXPERIMENTAL=0
endif
endif

# verify existence of separate lapack library when using blas/openblas/atlas
# switch off lapack support in case it can't be found
# issue covered with this
#   -  for Ubuntu 14.04 or lower, lapack is not automatically installed with openblas
#   -  for Ubuntu, installing atlas will not automatically install the atlas provided lapack library
# silently switching lapack off instead of letting the build fail because of backward compatibility
ifeq ($(USE_LAPACK), 1)
ifeq ($(USE_BLAS),$(filter $(USE_BLAS),blas openblas atlas))
ifeq (,$(wildcard /lib/liblapack.a))
ifeq (,$(wildcard /usr/lib/liblapack.a))
ifeq (,$(wildcard $(USE_LAPACK_PATH)/liblapack.a))
	USE_LAPACK = 0
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
	ifeq ($(USE_BLAS),$(filter $(USE_BLAS),blas openblas atlas))
		LDFLAGS += -llapack
	endif
	CFLAGS += -DMXNET_USE_LAPACK
endif

ifeq ($(USE_CUDNN), 1)
	CFLAGS += -DMSHADOW_USE_CUDNN=1
	LDFLAGS += -lcudnn
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

ifneq ($(USE_CUDA_PATH), NONE)
	NVCC=$(USE_CUDA_PATH)/bin/nvcc
endif

# Sets 'CUDA_ARCH', which determines the GPU architectures supported
# by the compiled kernels.  Users can edit the KNOWN_CUDA_ARCHS list below
# to remove archs they don't wish to support to speed compilation, or they
# can pre-set the CUDA_ARCH args in config.mk for full control.
#
# For archs in this list, nvcc will create a fat-binary that will include
# the binaries (SASS) for all architectures supported by the installed version
# of the cuda toolkit, plus the assembly (PTX) for the most recent such architecture.
# If these kernels are then run on a newer-architecture GPU, the binary will
# be JIT-compiled by the updated driver from the included PTX.
ifeq ($(USE_CUDA), 1)
ifeq ($(origin CUDA_ARCH), undefined)
	KNOWN_CUDA_ARCHS := 30 35 50 52 60 61
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
endif

# ps-lite
PS_PATH=$(ROOTDIR)/ps-lite
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

SRC = $(wildcard src/*/*/*.cc src/*/*.cc src/*.cc)
OBJ = $(patsubst %.cc, build/%.o, $(SRC))
CUSRC = $(wildcard src/*/*/*.cu src/*/*.cu src/*.cu)
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

# scala package profile
ifeq ($(OS),Windows_NT)
	# TODO(yizhi) currently scala package does not support windows
	SCALA_PKG_PROFILE := windows
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S), Darwin)
		WHOLE_ARCH= -all_load
		NO_WHOLE_ARCH= -noall_load
		SCALA_PKG_PROFILE := osx-x86_64
	else
		SCALA_PKG_PROFILE := linux-x86_64
		WHOLE_ARCH= --whole-archive
		NO_WHOLE_ARCH= --no-whole-archive
	endif
endif

# all dep
LIB_DEP += $(DMLC_CORE)/libdmlc.a $(NNVM_PATH)/lib/libnnvm.a
ALL_DEP = $(OBJ) $(EXTRA_OBJ) $(PLUGIN_OBJ) $(LIB_DEP)

ifeq ($(USE_CUDA), 1)
	CFLAGS += -I$(ROOTDIR)/cub
	ALL_DEP += $(CUOBJ) $(EXTRA_CUOBJ) $(PLUGIN_CUOBJ)
	LDFLAGS += -lcuda -lcufft
	SCALA_PKG_PROFILE := $(SCALA_PKG_PROFILE)-gpu
else
	SCALA_PKG_PROFILE := $(SCALA_PKG_PROFILE)-cpu
endif

# For quick compile test, used smaller subset
ALLX_DEP= $(ALL_DEP)

ifeq ($(USE_NVRTC), 1)
	LDFLAGS += -lnvrtc
	CFLAGS += -DMXNET_USE_NVRTC=1
else
	CFLAGS += -DMXNET_USE_NVRTC=0
endif

build/src/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) -std=c++11 -c $(CFLAGS) -MMD -c $< -o $@

build/src/%_gpu.o: src/%.cu
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

%.o: %.cc
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

$(PS_PATH)/build/libps.a: PSLITE

PSLITE:
	$(MAKE) CXX=$(CXX) DEPS_PATH=$(DEPS_PATH) -C $(PS_PATH) ps

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

include tests/cpp/unittest.mk

extra-packages: $(EXTRA_PACKAGES)

test: $(TEST)

lint: cpplint rcpplint jnilint pylint

cpplint:
	python2 dmlc-core/scripts/lint.py mxnet cpp include src plugin cpp-package tests \
	--exclude_path src/operator/contrib/ctc_include

pylint:
	pylint python/mxnet tools/caffe_converter/*.py --rcfile=$(ROOTDIR)/tests/ci_build/pylintrc

doc: docs

docs:
	tests/ci_build/ci_build.sh doc make -C docs html

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
	python2 dmlc-core/scripts/lint.py mxnet-rcpp ${LINT_LANG} R-package/src

rpkg:
	mkdir -p R-package/inst
	mkdir -p R-package/inst/libs
	cp -rf lib/libmxnet.so R-package/inst/libs
	mkdir -p R-package/inst/include
	cp -rf include/* R-package/inst/include
	cp -rf dmlc-core/include/* R-package/inst/include/
	cp -rf nnvm/include/* R-package/inst/include
	echo "import(Rcpp)" > R-package/NAMESPACE
	echo "import(methods)" >> R-package/NAMESPACE
	R CMD INSTALL R-package
	Rscript -e "require(mxnet); mxnet:::mxnet.export(\"R-package\")"
	rm -rf R-package/NAMESPACE
	Rscript -e "require(devtools); install_version(\"roxygen2\", version = \"5.0.1\", repos = \"https://cloud.r-project.org/\", quiet = TRUE)"
	Rscript -e "require(roxygen2); roxygen2::roxygenise(\"R-package\")"
	R CMD build --no-build-vignettes R-package
	rm -rf mxnet_current_r.tar.gz
	mv mxnet_*.tar.gz mxnet_current_r.tar.gz

scalapkg:
	(cd $(ROOTDIR)/scala-package; \
		mvn clean package -P$(SCALA_PKG_PROFILE) -Dcxx="$(CXX)" \
			-Dcflags="$(CFLAGS)" -Dldflags="$(LDFLAGS)" \
			-Dcurrent_libdir="$(ROOTDIR)/lib" \
			-Dlddeps="$(LIB_DEP) $(ROOTDIR)/lib/libmxnet.a")

scalatest:
	(cd $(ROOTDIR)/scala-package; \
		mvn verify -P$(SCALA_PKG_PROFILE) -Dcxx="$(CXX)" \
			-Dcflags="$(CFLAGS)" -Dldflags="$(LDFLAGS)" \
			-Dlddeps="$(LIB_DEP) $(ROOTDIR)/lib/libmxnet.a" $(SCALA_TEST_ARGS))

scalainstall:
	(cd $(ROOTDIR)/scala-package; \
		mvn install -P$(SCALA_PKG_PROFILE) -DskipTests -Dcxx="$(CXX)" \
			-Dcflags="$(CFLAGS)" -Dldflags="$(LDFLAGS)" \
			-Dlddeps="$(LIB_DEP) $(ROOTDIR)/lib/libmxnet.a")

scaladeploy:
	(cd $(ROOTDIR)/scala-package; \
		mvn deploy -Prelease,$(SCALA_PKG_PROFILE) -DskipTests -Dcxx="$(CXX)" \
			-Dcflags="$(CFLAGS)" -Dldflags="$(LDFLAGS)" \
			-Dlddeps="$(LIB_DEP) $(ROOTDIR)/lib/libmxnet.a")

jnilint:
	python2 dmlc-core/scripts/lint.py mxnet-jnicpp cpp scala-package/native/src

ifneq ($(EXTRA_OPERATORS),)
clean: cyclean $(EXTRA_PACKAGES_CLEAN)
	$(RM) -r build lib bin *~ */*~ */*/*~ */*/*/*~ R-package/NAMESPACE R-package/man R-package/R/mxnet_generated.R \
		R-package/inst R-package/src/*.o R-package/src/*.so mxnet_*.tar.gz
	cd $(DMLC_CORE); $(MAKE) clean; cd -
	cd $(PS_PATH); $(MAKE) clean; cd -
	cd $(NNVM_PATH); $(MAKE) clean; cd -
	$(RM) -r  $(patsubst %, %/*.d, $(EXTRA_OPERATORS)) $(patsubst %, %/*/*.d, $(EXTRA_OPERATORS))
	$(RM) -r  $(patsubst %, %/*.o, $(EXTRA_OPERATORS)) $(patsubst %, %/*/*.o, $(EXTRA_OPERATORS))
else
clean: cyclean testclean $(EXTRA_PACKAGES_CLEAN)
	$(RM) -r build lib bin *~ */*~ */*/*~ */*/*/*~ R-package/NAMESPACE R-package/man R-package/R/mxnet_generated.R \
		R-package/inst R-package/src/*.o R-package/src/*.so mxnet_*.tar.gz
	cd $(DMLC_CORE); $(MAKE) clean; cd -
	cd $(PS_PATH); $(MAKE) clean; cd -
	cd $(NNVM_PATH); $(MAKE) clean; cd -
endif

clean_all: clean

-include build/*.d
-include build/*/*.d
-include build/*/*/*.d
-include build/*/*/*/*.d
ifneq ($(EXTRA_OPERATORS),)
	-include $(patsubst %, %/*.d, $(EXTRA_OPERATORS)) $(patsubst %, %/*/*.d, $(EXTRA_OPERATORS))
endif
