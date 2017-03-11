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

echo "Add python path"
export PYTHONPATH=$PYTHONPATH:python


echo "BUILD mxnet document"
cd docs
make html

echo "Check spell and grammar for documentation"
cd ../tests/nightly/TestDoc
rm -rf web-data
git clone https://github.com/dmlc/web-data.git
sudo cp web-data/mxnet/doc/en_US-large.aff web-data/mxnet/doc/en_US-large.dic web-data/mxnet/doc/en_US.aff web-data/mxnet/doc/en_US.dic /usr/share/myspell
python doc_spell_checker.py || exit 1

echo "Check spell and grammar End"
