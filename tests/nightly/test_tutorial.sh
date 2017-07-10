#!/bin/bash

sudo apt-get update && apt-get install -y \
    maven default-jdk
wget http://downloads.lightbend.com/scala/2.11.8/scala-2.11.8.deb && \
    sudo dpkg -i scala-2.11.8.deb && rm scala-2.11.8.deb

sudo apt-get -y install git
sudo apt-get -y install ipython ipython-notebook
sudo apt-get -y install graphviz
sudo apt-get -y install doxygen
sudo apt-get -y install pandoc
sudo apt-get -y install python-tk
sudo apt-get -y install python-opencv

sudo python -m pip install -U pip
sudo pip install virtualenv
sudo pip install sphinx==1.5.1 CommonMark==0.5.4 breathe mock==1.0.1 recommonmark pypandoc
sudo pip install --upgrade requests

#Build mxnet and docs
git clone https://github.com/dmlc/mxnet.git --recursive
cd mxnet
cp make/config.mk .
make -j8 USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 || exit 1
cd docs
make html

#Setup virtualenv and install packages
cd ../python
virtualenv ENV
source /home/ec2-user/workspace/NightlyTutorialUbuntu/mxnet/python/ENV/bin/activate
cp /usr/lib/python2.7/dist-packages/cv2.so /home/ec2-user/workspace/NightlyTutorialUbuntu/mxnet/python/ENV/lib/python2.7/site-packages
cp /usr/lib/python2.7/dist-packages/cv.py /home/ec2-user/workspace/NightlyTutorialUbuntu/mxnet/python/ENV/lib/python2.7/site-packages
pip install six
/home/ec2-user/workspace/NightlyTutorialUbuntu/mxnet/python/ENV/bin/python setup.py install

pip install requests
pip install jupyter
pip install graphviz
pip install matplotlib

#Test tutorials
cd ../tests/nightly
python test_tutorial.py || exit 1
