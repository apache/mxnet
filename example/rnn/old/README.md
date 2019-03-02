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

RNN Example
===========
This folder contains RNN examples using low level symbol interface.

## Data
Run `get_sherlockholmes_data.sh` to download Sherlock Holmes data.

## Python

- [lstm.py](lstm.py) Functions for building a LSTM Network
- [gru.py](gru.py) Functions for building a GRU Network
- [lstm_bucketing.py](lstm_bucketing.py) Sherlock Holmes language model by using LSTM
- [gru_bucketing.py](gru_bucketing.py) Sherlock Holmes language model by using GRU
- [char-rnn.ipynb](char-rnn.ipynb) Notebook to demo how to train a character LSTM by using ```lstm.py```


Performance Note:
More ```MXNET_GPU_WORKER_NTHREADS``` may lead to better performance. For setting ```MXNET_GPU_WORKER_NTHREADS```, please refer to [Environment Variables](https://mxnet.readthedocs.org/en/latest/faq/env_var.html).
