#!/bin/bash
# check Jenkinsfile for a sample usage
set -ex

if [ $# -ne 1 ]; then
    echo "Usage: $0 TARGET_DIR"
    exit -1
fi

DIR=$1

conda activate mxnet-docs

cd Rsite

# TODO(mli) remove EVAL=0
export EVAL=0; make html

rm -rf ../$DIR
mv build/ ../$DIR
