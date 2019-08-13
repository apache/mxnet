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

## Wide and Deep Learning

The example demonstrates how to train [wide and deep model](https://arxiv.org/abs/1606.07792). The [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/Census+Income) that this example uses for training is hosted by the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/). Tricks of feature engineering are adapted from tensorflow's [wide and deep tutorial](https://github.com/tensorflow/models/tree/master/official/wide_deep).

The final accuracy should be around 85%.
For training:
- `python train.py`

For inference:
- `python inference.py`
