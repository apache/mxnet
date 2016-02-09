#!/bin/bash

if [ ${TASK} == "r_test" ]; then
    cat mxnet/mxnet.Rcheck/*.log
fi
