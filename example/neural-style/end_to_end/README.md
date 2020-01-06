<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# End to End Neural Art

Please refer to this [blog](http://dmlc.ml/mxnet/2016/06/20/end-to-end-neural-style.html) for details of how it is implemented.

## How to use


1. First use `../download.sh` to download pre-trained model and sample inputs.

2. Prepare training dataset. Put image samples to `../data/` (one file for each image sample). The pretrained model here was trained by 26k images sampled from [MIT Place dataset](http://places.csail.mit.edu).

3. Use `boost_train.py` for training.

## Pretrained Model

- Model: [https://github.com/dmlc/web-data/raw/master/mxnet/art/model.zip](https://github.com/dmlc/web-data/raw/master/mxnet/art/model.zip)
- Inference script: `boost_inference.py`
