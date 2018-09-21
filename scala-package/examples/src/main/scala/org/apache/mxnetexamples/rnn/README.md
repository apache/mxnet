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