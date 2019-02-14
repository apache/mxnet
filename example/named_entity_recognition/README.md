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

## Goal

- This repo contains an MXNet implementation of this state of the art [entity recognition model](https://www.aclweb.org/anthology/Q16-1026).
- You can find my blog post on the model [here](https://opringle.github.io/2018/02/06/CNNLSTM_entity_recognition.html).

![](https://github.com/dmlc/web-data/blob/master/mxnet/example/ner/arch1.png?raw=true)

## Running the code

To reproduce the preprocessed training data:

1. Download and unzip the data: https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/downloads/ner_dataset.csv
2. Move ner_dataset.csv into `./data`
3. `$ cd src && python preprocess.py`

To train the model:

- `$ cd src && python ner.py`

To run inference using trained model:

1. Recreate the bucketing module using `sym_gen` defined in `ner.py`
2. Loading saved parameters using `module.set_params()`

Refer to the `test` function in the [Bucketing Module example](https://github.com/apache/incubator-mxnet/blob/master/example/rnn/bucketing/cudnn_rnn_bucketing.py)
and this [issue](https://github.com/apache/incubator-mxnet/issues/5008) on Bucketing Module Prediction