#!/bin/bash

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


if [ -f $(which nvidia-smi) ]; then
    if [ $# -eq 1 ]; then
        num_gpus=$1
    else
        num_gpus=$(nvidia-smi -L | grep "GPU" | wc -l)
    fi
    gpus=`seq 0 $((num_gpus-1)) | paste -sd ","`
    device_arg="--gpus $gpus"
else
    device_arg=""
fi

# check if the final evaluation accuracy exceed the threshold
check_val() {
    expected=$1
    pass="Final validation >= $expected, PASS"
    fail="Final validation < $expected, FAIL"
    python tools/parse_log.py log --format none | tail -n1 | \
        awk "{ if (\$3~/^[.0-9]+$/ && \$3 > $expected) print \"$pass\"; else print \"$fail\"}"
    rm -f log
}

example_dir=example/image-classification
# python: lenet + mnist
test_lenet() {
    optimizers="adam sgd adagrad"
    for optimizer in ${optimizers}; do
        echo "OPTIMIZER: $optimizer"
        if [ "$optimizer" == "adam" ]; then
            learning_rate=0.0005
        else
            learning_rate=0.01
        fi
        desired_accuracy=0.98
        python $example_dir/train_mnist.py --lr $learning_rate \
            --network lenet --optimizer $optimizer --gpus $gpus \
            --num-epochs 10 2>&1 | tee log
       if [ $? -ne 0 ]; then
           return $?
       fi
       check_val $desired_accuracy
    done
}

test_lenet

exit $errors
