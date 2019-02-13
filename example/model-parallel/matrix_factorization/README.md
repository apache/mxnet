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

Model Parallel Matrix Factorization
===================================

This example walks you through a matrix factorization algorithm for recommendations and also
demonstrates the basic usage of `group2ctxs` in `Module`, which allows one part of the model to be
trained on cpu and the other on gpu. So, it is necessary to have GPUs available on the machine
to run this example.

To run this example, first make sure you download a dataset of 10 million movie ratings available
from [the MovieLens project](http://files.grouplens.org/datasets/movielens/) by running following command:

`python get_data.py`

This will download MovieLens 10M dataset under ml-10M100K folder. Now, you can run the training as follows:

`python train.py --num-gpus 1`

You can also specify other attributes such as num-epoch, batch-size,
factor-size(output dim of the embedding operation) to train.py.

While training you will be able to see the usage of ctx_group attribute to divide the operators
into different groups corresponding to different CPU/GPU devices.
