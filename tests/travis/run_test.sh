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


if ! tests/travis/is_core_changed.sh
then
  exit 0
fi

if [ ${TASK} == "lint" ]; then
    make lint || exit -1
    echo "Check documentations of c++ code..."
    make doc 2>log.txt
    (cat log.txt| grep -v ENABLE_PREPROCESSING |grep -v "unsupported tag") > logclean.txt
    echo "---------Error Log----------"
    cat logclean.txt
    echo "----------------------------"
    (cat logclean.txt|grep warning) && exit -1
    (cat logclean.txt|grep error) && exit -1
    exit 0
fi

cp make/config.mk config.mk

if [[ ${TRAVIS_OS_NAME} == "osx" ]]; then
    echo "USE_BLAS=apple" >> config.mk
    echo "USE_OPENMP=0" >> config.mk
else
    # use g++-4.8 for linux
    if [[ ${CXX} == "g++" ]]; then
        export CXX=g++-4.8
    fi
    echo "USE_BLAS=blas" >> config.mk
fi
echo "CXX=${CXX}" >>config.mk
echo "USE_PROFILER=1" >> config.mk

if [ ${TASK} == "build" ]; then
    if [ ${TRAVIS_OS_NAME} == "linux" ]; then
        echo "USE_CUDA=1" >> config.mk
        ./dmlc-core/scripts/setup_nvcc.sh $NVCC_PREFIX
    fi
    make all
    exit $?
fi

if [ ${TASK} == "cpp_test" ]; then
    make -f dmlc-core/scripts/packages.mk gtest
    echo "GTEST_PATH="${CACHE_PREFIX} >> config.mk
    make test || exit -1
    export MXNET_ENGINE_INFO=true
    ./build/tests/cpp/mxnet_test
    exit 0
fi

if [ ${TASK} == "r_test" ]; then
    make all || exit -1
    # use cached dir for storing data
    rm -rf ${PWD}/data
    mkdir -p ${CACHE_PREFIX}/data
    ln -s ${CACHE_PREFIX}/data ${PWD}/data

    set -e
    export _R_CHECK_TIMINGS_=0

    if [[ ${TRAVIS_OS_NAME} == "osx" ]]; then
        wget https://cran.rstudio.com/bin/macosx/R-latest.pkg  -O /tmp/R-latest.pkg
        sudo installer -pkg "/tmp/R-latest.pkg" -target /
        Rscript -e "install.packages('devtools', repo = 'https://cran.rstudio.com')"
    fi

    cd R-package
    Rscript -e "library(devtools); library(methods); options(repos=c(CRAN='https://cran.rstudio.com')); install_deps(dependencies = TRUE)"
    cd ..

    make rpkg
#    R CMD check --no-examples --no-manual --no-vignettes --no-build-vignettes mxnet_*.tar.gz
    R CMD INSTALL mxnet_*.tar.gz

    Rscript tests/travis/r_vignettes.R

    wget http://data.mxnet.io/mxnet/data/Inception.zip
    unzip Inception.zip && rm -rf Inception.zip
    wget http://data.mxnet.io/mxnet/data/mnist.zip
    unzip mnist.zip && rm -rf mnist.zip

    cat CallbackFunctionTutorial.R \
    fiveMinutesNeuralNetwork.R \
    mnistCompetition.R \
    ndarrayAndSymbolTutorial.R > r_test.R

    Rscript r_test.R || exit -1

    exit 0
fi

if [ ${TASK} == "python_test" ]; then
    make all || exit -1
    # use cached dir for storing data
    rm -rf ${PWD}/data
    mkdir -p ${PWD}/data

    if [ ${TRAVIS_OS_NAME} == "osx" ]; then
        python -m nose -v tests/python/unittest || exit -1
        python3 -m nose -v tests/python/unittest || exit -1
        # make cython3
        # cython tests
        # export MXNET_ENFORCE_CYTHON=1
        # python3 -m nose tests/python/unittest || exit -1
        python3 -m nose -v tests/python/train || exit -1
        python -m nose -v tests/python/doctest || exit -1
        python3 -m nose -v tests/python/doctest || exit -1
    else
        nosetests -v tests/python/unittest || exit -1
        nosetests3 -v tests/python/unittest || exit -1
        nosetests3 -v tests/python/train || exit -1
        nosetests -v tests/python/doctest || exit -1
        nosetests3 -v tests/python/doctest || exit -1
    fi
    exit 0
fi

if [ ${TASK} == "julia" ]; then
    make all || exit -1
    # use cached dir for storing data
    rm -rf ${PWD}/data
    mkdir -p ${PWD}/data

    export MXNET_HOME="${PWD}"
    julia -e 'Pkg.clone("MXNet"); Pkg.checkout("MXNet"); Pkg.build("MXNet"); Pkg.test("MXNet")' || exit -1
    exit 0
fi

if [ ${TASK} == "scala_test" ]; then
    if [ ${TRAVIS_OS_NAME} == "osx" ]; then
        LIB_GOMP_PATH=`find /usr/local/lib -name libgomp.dylib | grep -v i386 | head -n1`
        ln -sf $LIB_GOMP_PATH /usr/local/lib/libgomp.dylib
    fi
    make all || exit -1
    # use cached dir for storing data
    rm -rf ${PWD}/data
    mkdir -p ${PWD}/data

    export JAVA_HOME=$(/usr/libexec/java_home)

    make scalapkg || exit -1
    make scalatest || exit -1

    exit 0
fi

if [ ${TASK} == "perl_test" ]; then
    make all || exit -1

    # use cached dir for storing data
    MXNET_HOME=${PWD}
    rm -rf ${MXNET_HOME}/perl-package/AI-MXNet/data
    mkdir -p ${CACHE_PREFIX}/data
    ln -s ${CACHE_PREFIX}/data ${MXNET_HOME}/perl-package/AI-MXNet/data

    export LD_LIBRARY_PATH=${MXNET_HOME}/lib
    export PERL5LIB=${HOME}/perl5/lib/perl5

    cd ${MXNET_HOME}/perl-package/AI-MXNetCAPI/
    perl Makefile.PL INSTALL_BASE=${HOME}/perl5
    make || exit -1
    if [ ${TRAVIS_OS_NAME} == "osx" ]; then
        install_name_tool -change lib/libmxnet.so \
            ${MXNET_HOME}/lib/libmxnet.so \
            blib/arch/auto/AI/MXNetCAPI/MXNetCAPI.bundle
    fi
    make install || exit -1

    cd ${MXNET_HOME}/perl-package/AI-NNVMCAPI/
    perl Makefile.PL INSTALL_BASE=${HOME}/perl5
    make || exit -1
    if [ ${TRAVIS_OS_NAME} == "osx" ]; then
        install_name_tool -change lib/libmxnet.so \
            ${MXNET_HOME}/lib/libmxnet.so \
            blib/arch/auto/AI/NNVMCAPI/NNVMCAPI.bundle
    fi
    make install || exit -1

    cd ${MXNET_HOME}/perl-package/AI-MXNet/
    perl Makefile.PL
    make test || exit -1
    exit 0
fi

if [ ${TASK} == "cpp_package_test" ]; then
    MXNET_HOME=${PWD}
    make travis -C ${MXNET_HOME}/cpp-package/example
    exit 0
fi
