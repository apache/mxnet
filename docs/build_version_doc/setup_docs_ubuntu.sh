#!/bin/bash

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

######################################################################
# This script installs MXNet with required dependencies on a Ubuntu Machine.
# Tested on Ubuntu 16.04+ distro.
# Important Maintenance Instructions:
#    Align changes with CI in /ci/docker/install/ubuntu_core.sh
#    and ubuntu_python.sh
######################################################################

set -ex
cd "$(dirname "$0")"

sudo apt-get update
echo "Installing MXNet core dependencies..."
sudo apt-get install -y \
    apt-transport-https \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libatlas-base-dev \
    libcurl4-openssl-dev \
    libjemalloc-dev \
    liblapack-dev \
    libopenblas-dev \
    libopencv-dev \
    libzmq3-dev \
    ninja-build \
    python-dev \
    python3-dev \
    software-properties-common \
    sudo \
    unzip \
    virtualenv \
    wget


echo "Installing MXNet docs dependencies..."
sudo apt-get install -y \
    doxygen \
    pandoc


echo "Installing Clojure dependencies..."
wget https://raw.githubusercontent.com/technomancy/leiningen/stable/bin/lein
chmod 775 lein
sudo cp lein /usr/local/bin
echo "Y" | sudo lein downgrade 2.8.3


echo "Installing R dependencies..."
# install libraries for mxnet's r package on ubuntu
sudo sh -c 'echo "deb http://cran.rstudio.com/bin/linux/ubuntu trusty/" >> /etc/apt/sources.list'

key=E084DAB9

sudo sh -c 'gpg --keyserver keyserver.ubuntu.com --recv-key $key || \
    gpg --keyserver keyserver.pgp.com --recv-keys $key || \
    gpg --keyserver ha.pool.sks-keyservers.net --recv-keys $key ;'

# Installing the latest version (3.3+) that is compatible with MXNet
sudo add-apt-repository 'deb [arch=amd64,i386] https://cran.rstudio.com/bin/linux/ubuntu xenial/'
sudo apt-key add ../../ci/docker/install/sbt.gpg

sudo apt-get update
sudo apt-get install -y --allow-unauthenticated \
    libcairo2-dev \
    libssl-dev \
    libxml2-dev \
    libxt-dev \
    r-base \
    r-base-dev


echo "Installing Scala dependencies..."
sudo apt-get install -y software-properties-common
sudo apt-get update
sudo apt-get install -y openjdk-8-jdk
sudo apt-get install -y openjdk-8-jre

sudo sh -c 'echo "deb https://dl.bintray.com/sbt/debian /" | tee -a /etc/apt/sources.list.d/sbt.list'
# ubuntu keyserver is very flaky
#apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823
#apt-key adv --keyserver keys.gnupg.net --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823
sudo apt-key add ../../ci/docker/install/sbt.gpg
sudo apt-get update && sudo apt-get install -y \
    maven \
    sbt \
    scala


wget -nv https://bootstrap.pypa.io/get-pip.py
echo "Installing for Python 3..."
sudo python3 get-pip.py
pip3 install --user -r ../../ci/docker/install/docs_requirements
echo "Installing for Python 2..."
sudo python2 get-pip.py
pip2 install --user -r ../../ci/docker/install/docs_requirements


cd ../../

echo "Checking for GPUs..."
gpu_install=$(which nvidia-smi | wc -l)
if [ "$gpu_install" = "0" ]; then
    make_params="USE_OPENCV=1 USE_BLAS=openblas"
    echo "nvidia-smi not found. Installing in CPU-only mode with these build flags: $make_params"
else
    make_params="USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1"
    echo "nvidia-smi found! Installing with CUDA and cuDNN support with these build flags: $make_params"
fi

echo "Building MXNet core. This can take few minutes..."
make -j $(nproc) $make_params


# Setup Apache2
echo "Setting up Apache web server..."
sudo apt-get install -y apache2
sudo ufw allow 'Apache Full'
# Turn on mod_rewrite
sudo a2enmod rewrite

echo 'To enable redirects you need to edit /etc/apache2/apache2.conf '
echo '--> Change directives for Directory for /var/www/html using the following: '
echo '       AllowOverride all '
echo '--> Then restart apache with: '
echo '       sudo systemctl restart apache2'


# Cleanup
sudo apt autoremove -y
