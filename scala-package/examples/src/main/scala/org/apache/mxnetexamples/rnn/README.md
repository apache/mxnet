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

# RNN Example for MXNet Scala
This folder contains the following examples writing in new Scala type-safe API:
- [x] LSTM Bucketing
- [x] CharRNN Inference : Generate similar text based on the model
- [x] CharRNN Training: Training the language model using RNN

These example is only for Illustration and not modeled to achieve the best accuracy.

## Setup
### Download the Network Definition, Weights and Training Data
`obama.zip` contains the training inputs (Obama's speech) for CharCNN examples and `sherlockholmes` contains the data for LSTM Bucketing
```bash
https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/RNN/obama.zip
https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/RNN/sherlockholmes.train.txt
https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/RNN/sherlockholmes.valid.txt
```
### Unzip the file
```bash
unzip obama.zip
```
### Arguement Configuration
Then you need to define the arguments that you would like to pass in the model:

#### LSTM Bucketing
```bash
--data-train
<path>/sherlockholmes.train.txt
--data-val
<path>/sherlockholmes.valid.txt
--cpus
<num_cpus>
--gpus
<num_gpu>
```
#### TrainCharRnn
```bash
--data-path
<path>/obama.txt
--save-model-path
<path>/
```
#### TestCharRnn
```bash
--data-path
<path>/obama.txt
--model-prefix
<path>/obama
```