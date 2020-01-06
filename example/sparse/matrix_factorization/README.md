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

Matrix Factorization w/ Sparse Embedding
===========
The example demonstrates the basic usage of the sparse.Embedding operator in MXNet, adapted based on @leopd's recommender examples.
This is for demonstration purpose only.

```
usage: train.py [-h] [--num-epoch NUM_EPOCH] [--seed SEED]
                [--batch-size BATCH_SIZE] [--log-interval LOG_INTERVAL]
                [--factor-size FACTOR_SIZE] [--gpus GPUS] [--dense]

Run matrix factorization with sparse embedding

optional arguments:
  -h, --help            show this help message and exit
  --num-epoch NUM_EPOCH
                        number of epochs to train (default: 3)
  --seed SEED           random seed (default: 1)
  --batch-size BATCH_SIZE
                        number of examples per batch (default: 128)
  --log-interval LOG_INTERVAL
                        logging interval (default: 100)
  --factor-size FACTOR_SIZE
                        the factor size of the embedding operation (default: 128)
  --gpus GPUS           list of gpus to run, e.g. 0 or 0,2. empty means using
                        cpu(). (default: None)
  --dense               whether to use dense embedding (default: False)
```
