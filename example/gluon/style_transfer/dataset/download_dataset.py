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

import os, zipfile
import mxnet
from mxnet.test_utils import download

def unzip_file(filename, outpath):
    fh = open(filename, 'rb')
    z = zipfile.ZipFile(fh)
    for name in z.namelist():
        z.extract(name, outpath)
    fh.close()

# Dataset from COCO 2014: http://cocodataset.org/#download
# for full text of the license, see https://creativecommons.org/licenses/by/4.0/legalcode
download('http://msvocds.blob.core.windows.net/coco2014/train2014.zip', 'dataset/train2014.zip')
download('http://msvocds.blob.core.windows.net/coco2014/val2014.zip', 'dataset/val2014.zip')

unzip_file('dataset/train2014.zip', 'dataset')
unzip_file('dataset/val2014.zip', 'dataset')
