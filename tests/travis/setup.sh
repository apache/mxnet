#!/bin/bash

if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    brew update
    brew tap homebrew/science
    brew info opencv
    brew install opencv
    brew install python3
    brew install fftw
    brew install ImageMagick
    LIB_GOMP_PATH=`find /usr/local -name libgomp.dylib | grep -v i386 | head -n1`
    echo "libgomp: $LIB_GOMP"
    ln -sf $LIB_GOMP_PATH /usr/local/lib/libgomp.dylib
    if [ ${TASK} == "python_test" ]; then
        python -m pip install nose numpy --user `whoami`
        python3 -m pip install nose numpy --user `whoami`
    fi
fi

if [ ${TASK} == "lint" ]; then
    pip install cpplint 'pylint==1.4.4' 'astroid==1.3.6' --user `whoami`
fi
