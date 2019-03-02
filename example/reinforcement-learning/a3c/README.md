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

# A3C Implementation
This is an attempt to implement the A3C algorithm in paper Asynchronous Methods for Deep Reinforcement Learning.

Author: Junyuan Xie (@piiswrong)

The algorithm should be mostly correct. However I cannot reproduce the result in the paper, possibly due to hyperparameter settings. If you can find a better set of parameters please propose a pull request.

Note this is a generalization of the original algorithm since we use `batch_size` threads for each worker instead of the original 1 thread.

## Prerequisites
  - Install OpenAI Gym: `pip install gym`
  - Install the Atari Env: `pip install gym[atari]`
  - You may need to install flask: `pip install flask`
  - You may have to install cv2: `pip install opencv-python`

## Usage
run `python a3c.py --batch-size=32 --gpus=0` to run training on gpu 0 with batch-size=32.

run `python launcher.py --gpus=0,1 -n 2 python a3c.py` to launch training on 2 gpus (0 and 1), each gpu has two workers.
Note: You might have to update the path to dmlc-core in launcher.py.
