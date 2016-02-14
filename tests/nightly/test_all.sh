#!/bin/bash

# setup
cd `pwd`/`dirname $0`
. sh2ju.sh
## clean last build log
juLogClean
if [ $# -eq 1 ]; then
    export NUM_GPUS=$1
else
    export NUM_GPUS=4
fi

# build
cp ../../make/config.mk ../..
cat >>../../config.mk <<EOF
USE_CUDA=1
USE_CUDA_PATH=/usr/local/cuda
USE_CUDNN=1
EOF
# make -C ../.. clean
juLog -name=Build -error=Error make -C ../.. -j8

# download data
juLog -name=Download bash ./download.sh

# run tests
bash ./test_python.sh

exit $errors
