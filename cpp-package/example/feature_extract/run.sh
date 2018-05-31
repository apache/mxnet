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

### To run the this example,
###
### 1.
### Get Inseption-BN model first, from here
###     https://github.com/dmlc/mxnet-model-gallery
###
### 2.
### Then Prepare 2 pictures, 1.jpg 2.jpg to extract

# Getting the data
mkdir -p model
wget -nc http://data.dmlc.ml/mxnet/models/imagenet/inception-bn.tar.gz
wget -nc -O 1.jpg https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/python/predict_image/cat.jpg?raw=true 
wget -nc -O 2.jpg https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/python/predict_image/dog.jpg?raw=true 
wget -nc -O model/mean_224.nd https://github.com/h2oai/deepwater/raw/master/mxnet/src/main/resources/deepwater/backends/mxnet/models/Inception/mean_224.nd
tar -xvzf inception-bn.tar.gz -C model --skip-old-files

# Building
make

# Preparing the data
./prepare_data_with_opencv

# Running the featurization
LD_LIBRARY_PATH=../../../lib ./feature_extract
