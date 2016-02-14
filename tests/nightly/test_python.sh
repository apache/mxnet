#!/bin/bash
. sh2ju.sh
gpus=`seq 0 $((NUM_GPUS-1)) | paste -sd ","`

example_dir=../../example/image-classification

# check if the final evaluation accuracy exceed the threshold
check_val() {
    expected=$1
    pass="Final validation >= $expected, Pass"
    fail="Final validation < $expected, Fail"
    python ../../tools/parse_log.py log --format none | tail -n1 | \
        awk "{ if (\$3~/^[.0-9]+$/ && \$3 > $expected) print \"$pass\"; else print \"$fail\"}"
    rm -f log
}

test_lenet() {
    python $example_dir/train_mnist.py \
        --data-dir `pwd`/data/mnist/ --network lenet --gpus $gpus --num-epochs 5 \
        2>&1 | tee log
    check_val 0.99
}

juLog -name=Python.Lenet -error=Fail test_lenet
