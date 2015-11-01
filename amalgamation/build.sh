export CXX=g++
echo "Building mxnet.a"
$CXX -O3 -std=c++11 -I/usr/local/Cellar/openblas/0.2.14_1/include -c  -o mxnet.o mxnet.cc
ar rcs mxnet.a mxnet.o


