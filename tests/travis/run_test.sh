#!/bin/bash

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

if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    echo "USE_BLAS=apple" >> config.mk
    echo "USE_OPENMP=0" >> config.mk
else
    # use g++-4.8 for linux
    if [ ${CXX} == "g++" ]; then
        export CXX=g++-4.8
    fi
    echo "USE_BLAS=blas" >> config.mk
fi
echo "CXX=${CXX}" >>config.mk

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
    for test in tests/cpp/*_test; do
        ./$test || exit -1
    done
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

    wget https://cran.rstudio.com/bin/macosx/R-latest.pkg  -O /tmp/R-latest.pkg
    wget https://xquartz-dl.macosforge.org/SL/XQuartz-2.7.8.dmg -O /tmp/XQuartz-2.7.8.dmg
    sudo installer -pkg "/tmp/R-latest.pkg" -target /
    sudo hdiutil attach /tmp/XQuartz-2.7.8.dmg
    cd /Volumes/XQuartz-2.7.8
    sudo installer -pkg "XQuartz.pkg" -target /
    cd -
    Rscript -e "install.packages('devtools', repo = 'https://cran.rstudio.com')"
    Rscript -e "install.packages('imager', repo = 'https://cran.rstudio.com', type = 'source')"
    cd R-package
    Rscript -e "library(devtools); library(methods); options(repos=c(CRAN='https://cran.rstudio.com')); install_deps(dependencies = TRUE)"
    cd ..
    make rpkg
    R CMD check --no-examples --no-manual --no-vignettes --no-build-vignettes mxnet_*.tar.gz
    R CMD INSTALL mxnet_*.tar.gz

    Rscript tests/travis/r_vignettes.R

    wget http://webdocs.cs.ualberta.ca/~bx3/data/Inception.zip
    unzip Inception.zip && rm -rf Inception.zip
    wget https://s3-us-west-2.amazonaws.com/mxnet/train.csv -O train.csv
    wget https://s3-us-west-2.amazonaws.com/mxnet/test.csv -O test.csv

    cat *.R > r_test.R

    Rscript r_test.R || exit -1

    exit 0
fi

if [ ${TASK} == "python_test" ]; then
    make all || exit -1
    # use cached dir for storing data
    rm -rf ${PWD}/data
    mkdir -p ${CACHE_PREFIX}/data
    ln -s ${CACHE_PREFIX}/data ${PWD}/data

    if [ ${TRAVIS_OS_NAME} == "osx" ]; then
        python -m nose tests/python/unittest || exit -1
        # python -m nose tests/python/train || exit -1
        python3 -m nose tests/python/unittest || exit -1
        python3 -m nose tests/python/train || exit -1
    else
        nosetests tests/python/unittest || exit -1
        # nosetests tests/python/train || exit -1
        nosetests3 tests/python/unittest || exit -1
        nosetests3 tests/python/train || exit -1
    fi
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
    mkdir -p ${CACHE_PREFIX}/data
    ln -s ${CACHE_PREFIX}/data ${PWD}/data

    export JAVA_HOME=$(/usr/libexec/java_home)

    make scalapkg || exit -1
    make scalatest || exit -1

    exit 0
fi
