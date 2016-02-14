#!/bin/bash

# setup
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
cd `pwd`/`dirname $0`
. sh2ju.sh
## clean last build log
juLogClean
if [ $# -eq 1 ]; then
    num_gpus=$1
else
    num_gpus=4
fi
gpus=`seq 0 $((num_gpus-1)) | paste -sd ","`

# build
cp ../../make/config.mk ../..
cat >>../../config.mk <<EOF
USE_CUDA=1
USE_CUDA_PATH=/usr/local/cuda
USE_CUDNN=1
EOF
make -C ../.. clean
juLog -name=Build -error=Error make -C ../.. -j8

# download data
juLog -name=DownloadData bash ./download.sh


# check if the final evaluation accuracy exceed the threshold
check_val() {
    expected=$1
    pass="Final validation >= $expected, Pass"
    fail="Final validation < $expected, Fail"
    python ../../tools/parse_log.py log --format none | tail -n1 | \
        awk "{ if (\$3~/^[.0-9]+$/ && \$3 > $expected) print \"$pass\"; else print \"$fail\"}"
    rm -f log
}

example_dir=../../example/image-classification
# python: lenet + mnist
test_lenet() {
    python $example_dir/train_mnist.py \
        --data-dir `pwd`/data/mnist/ --network lenet --gpus $gpus --num-epochs 10 \
        2>&1 | tee log
    check_val 0.99
}
juLog -name=Python.Lenet.Mnist -error=Fail test_lenet

# python: inception + cifar10
test_inception_cifar10() {
    python $example_dir/train_cifar10.py \
        --data-dir `pwd`/data/cifar10/ --gpus $gpus --num-epochs 10 --batch-size 512 \
        2>&1 | tee log
    check_val 0.78
}
juLog -name=Python.Inception.Cifar10 -error=Fail test_inception_cifar10

exit $errors
