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

# MNIST classification example

This script shows a simple example how to do image classification with Gluon. 
The model is trained on MNIST digits image dataset and the goal is to classify the digits ```0-9```.  The model has the following layout:
```
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(128, activation='relu'))
    net.add(nn.Dense(64, activation='relu'))
    net.add(nn.Dense(10))
```

The script provides the following commandline arguments: 


```
MXNet Gluon MNIST Example

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        batch size for training and testing (default: 100)
  --epochs EPOCHS       number of epochs to train (default: 10)
  --lr LR               learning rate (default: 0.1)
  --momentum MOMENTUM   SGD momentum (default: 0.9)
  --cuda                Train on GPU with CUDA
  --log-interval N      how many batches to wait before logging training
                        status
```

After one epoch we get the following output vector for the given test image:

<img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/example/gluon/mnist/test_image.png" width="250" height="250">

[-5.461655  -4.745     -1.8203478 -0.5705207  8.923972  -2.2358544 -3.3020825 -2.409004   4.0074944 10.362008] 

As we can see the highest activation is 10.362 which corresponds to label `9`.

