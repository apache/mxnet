#!/bin/bash

if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    brew update
    brew tap homebrew/science
    brew info opencv
    brew install opencv
    brew install python3
fi

if [ ${TASK} == "lint" ]; then
    pip install cpplint pylint graphviz --user `whoami`
fi

if [ ${TASK} == "python_test"]; then
    python -m pip install nose numpy --user `whoami`
    python3 -m pip install nose numpy --user `whoami`
fi
