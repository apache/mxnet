#!/bin/bash

if [ ${TASK} == "r_test" ]; then
    echo "Print the install log..."
    cat mxnet.Rcheck/*.out
    echo "Print the check log..."
    cat mxnet.Rcheck/*.log
fi
