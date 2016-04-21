#!/bin/bash

if ! tests/travis/is_core_changed.sh
then
  exit 0
fi

if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    brew update
    brew tap homebrew/science
    brew info opencv
    brew install opencv
    brew install python3
    brew install fftw
    brew install ImageMagick
    if [ ${TASK} == "python_test" ]; then
        python -m pip install nose numpy --user `whoami`
        python3 -m pip install nose numpy --user `whoami`
    fi
fi

if [ ${TASK} == "lint" ]; then
    pip install cpplint 'pylint==1.4.4' 'astroid==1.3.6' --user `whoami`
fi

if [ ${TASK} == "julia" ]; then
  mkdir -p ~/julia
  curl -s -L --retry 7 "https://s3.amazonaws.com/julialang/bin/linux/x64/${JULIA_VER}/julia-${JULIA_VER}-latest-linux-x86_64.tar.gz" | tar -C ~/julia -x -z --strip-components=1 -f -
  export PATH="${PATH}:${HOME}/julia/bin"
  julia -e 'versioninfo()'
fi
