export MXNET_ROOT=`pwd`/..
export OPENBLAS_ROOT=/usr/local/Cellar/openblas/0.2.14_1
rm -f ./mxnet
echo "Linking $MXNET_ROOT to ./mxnet"
ln -s $MXNET_ROOT ./mxnet
echo "Generating deps from $MXNET_ROOT to mxnet0.d with mxnet0.cc"
g++ -MD -MF mxnet0.d -std=c++11 -Wall -I ./mxnet/ -I ./mxnet/mshadow/ -I ./mxnet/dmlc-core/include -I ./mxnet/include -I$OPENBLAS_ROOT/include -c  mxnet0.cc

echo "Generating amalgamation to mxnet.cc"
python ./expand.py

echo "Done"


