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

Recurrent Neural Network Examples
===========

For more current implementations of NLP and RNN models with MXNet, please visit [gluon-nlp](http://gluon-nlp.mxnet.io/index.html)

------


This directory contains functions for creating recurrent neural networks
models using high level mxnet.rnn interface.

Here is a short overview of what is in this directory.

Directory | What's in it?
--- | ---
`word_lm/` | Language model trained on the Sherlock Holmes dataset achieving state of the art performance
`bucketing/` | Language model with bucketing API with python
`bucket_R/` | Language model with bucketing API with R
`old/` | Language model trained with low level symbol interface (deprecated)
