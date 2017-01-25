#!/bin/sh
echo "BUILD make"
cp ./make/config.mk .
echo "USE_CUDA=0" >> ./config.mk
echo "USE_CUDNN=0" >> ./config.mk
echo "USE_BLAS=openblas" >> ./config.mk
echo "ADD_CFLAGS += -I/usr/include/openblas" >> ./config.mk
echo "GTEST_PATH=/usr/local/gtest" >> ./config.mk
echo 'export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH' >> ~/.profile
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.profile
echo 'export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.111-1.b15.25.amzn1.x86_64' >> ~/.profile
echo 'export JRE_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.111-1.b15.25.amzn1.x86_64/jre' >> ~/.profile
echo 'export PATH=$PATH:/apache-maven-3.3.9/bin/:/usr/bin:/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.111-1.b15.25.amzn1.x86_64/bin:/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.111-1.b15.25.amzn1.x86_64/jre/bin' >> ~/.profile
source ~/.profile
make clean
make -j 4 || exit -1

echo "BUILD python2 mxnet"
cd ./python
python setup.py install || exit 1

echo "BUILD python3 mxnet"
python3 setup.py install || exit 1
echo "~/.local"
cd ../tests/nightly

echo "Pull mxnet-notebook"
git clone https://github.com/dmlc/mxnet-notebooks.git

echo "Test Jupyter notebook"
python test_ipynb.py

echo "Test Summary Start"
cat test_summary.txt
echo "Test Summary End"
