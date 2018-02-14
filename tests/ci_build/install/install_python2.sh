#!/usr/bin/env bash

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
set -e
set -x

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

