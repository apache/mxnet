#!/bin/bash

if ! tests/travis/is_core_changed.sh
then
  exit 0
fi

if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    brew update
    brew tap homebrew/science
    brew install opencv
    brew install python3
    brew install fftw
    brew install libpng
    brew install ImageMagick
    if [ ${TASK} == "python_test" ]; then
        python -m pip install --user nose numpy cython
        python3 -m pip install --user nose numpy cython
    fi
fi

if [ ${TASK} == "lint" ]; then
    pip install --user cpplint 'pylint==1.4.4' 'astroid==1.3.6'
fi

if [ ${TASK} == "julia" ]; then
  mkdir -p ~/julia
  curl -s -L --retry 7 "https://s3.amazonaws.com/julialang/bin/linux/x64/${JULIA_VER}/julia-${JULIA_VER}-latest-linux-x86_64.tar.gz" | tar -C ~/julia -x -z --strip-components=1 -f -
  export PATH="${PATH}:${HOME}/julia/bin"
  julia -e 'versioninfo()'
fi

if [ ${TASK} == "perl_test" ]; then
  cpanm -L "${HOME}/perl5" Function::Parameters
  export PERL5LIB="${HOME}/perl5/lib/perl5"
fi
