set -e # exit on the first error
cd $(dirname $(readlink -f $0))/../example
echo $PWD
export LD_LIBRARY_PATH=$(readlink -f ../../lib):$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
ls -l ../../lib/

cp ../../build/cpp-package/example/test_score .
./get_mnist.sh
./test_score 0.93
