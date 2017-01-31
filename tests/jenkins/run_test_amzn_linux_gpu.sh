#!/bin/bash

echo "BUILD make"
cp make/config.mk .
echo "USE_CUDA=0" >> config.mk
echo "USE_CUDNN=0" >> config.mk
echo "USE_BLAS=openblas" >> config.mk
echo "ADD_CFLAGS += -I/usr/include/openblas" >>config.mk
echo "GTEST_PATH=/usr/local/gtest" >> config.mk
echo 'export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH' >> ~/.profile
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.profile
echo 'export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.111-1.b15.25.amzn1.x86_64' >> ~/.profile
echo 'export JRE_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.111-1.b15.25.amzn1.x86_64/jre' >> ~/.profile
echo 'export PATH=$PATH:/apache-maven-3.3.9/bin/:/usr/bin:/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.111-1.b15.25.amzn1.x86_64/bin:/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.111-1.b15.25.amzn1.x86_64/jre/bin' >> ~/.profile
source ~/.profile
user=`id -u -n`
make -j 4 || exit -1

echo "BUILD python2 mxnet"
cd python
if [ $user == 'root' ]
then
    python setup.py install || exit 1
else
    python setup.py install --prefix ~/.local || exit 1
fi
cd ..

echo "BUILD python3 mxnet"
cd python
if [ $user == 'root' ]
then
    python3 setup.py install || exit 1
else
    python3 setup.py install --prefix ~/.local || exit 1
fi
cd ..

echo "BUILD lint"
make lint || exit -1

echo "BUILD cpp_test"
make -j 4 test || exit -1
export MXNET_ENGINE_INFO=true
for test in tests/cpp/*_test; do
    ./$test || exit -1
done
export MXNET_ENGINE_INFO=false

echo "BUILD python_test"
nosetests --verbose tests/python/unittest || exit -1
nosetests --verbose tests/python/train || exit -1

echo "BUILD python3_test"
nosetests3 --verbose tests/python/unittest || exit -1
nosetests3 --verbose tests/python/train || exit -1

echo "BUILD julia_test"
export MXNET_HOME="${PWD}"
julia -e 'try Pkg.clone("MXNet"); catch end; Pkg.checkout("MXNet"); Pkg.build("MXNet"); Pkg.test("MXNet")' || exit -1

echo "BUILD scala_test"
make scalapkg || exit -1
make scalatest || exit -1
