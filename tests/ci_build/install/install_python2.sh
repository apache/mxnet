#!/usr/bin/env bash

set -e

yum groupinstall -y "Development Tools"
yum install -y mlocate python27 python27-setuptools python27-tools python27-numpy python27-scipy python27-nose python27-matplotlib unzip
ln -s -f /usr/bin/python2.7 /usr/bin/python2
wget https://bootstrap.pypa.io/get-pip.py
python2 get-pip.py
$(which easy_install-2.7) --upgrade pip
if [ -f /usr/local/bin/pip ] && [ -f /usr/bin/pip ]; then
  mv /usr/bin/pip /usr/bin/pip.bak
  ln /usr/local/bin/pip /usr/bin/pip
fi

ln -s -f /usr/local/bin/pip /usr/bin/pip
for i in ipython[all] jupyter pandas scikit-image h5py pandas sklearn sympy; do echo "${i}..."; pip install -U $i >/dev/null; done

