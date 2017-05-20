set -e # exit on the first error
cd $(dirname $(readlink -f $0))/../example
echo $PWD
ls -l ../..
ln -sf ../../lib/libmxnet.so .
ls -l libmxnet.so

cp ../../build/cpp-package/example/test_score .
./get_mnist.sh
./test_score 0.93
