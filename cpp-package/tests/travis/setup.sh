#!/bin/bash

if [ ${TASK} == "lint" ]; then
    pip install cpplint 'pylint==1.4.4' 'astroid==1.3.6' --user
fi
