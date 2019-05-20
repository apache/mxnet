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

# Superresolution

This example trains a convolutional neural network to enhance the resolution of images (also known as superresolution). 
The script takes the following commandline arguments:

```
Super-resolution using an efficient sub-pixel convolution neural network.

optional arguments:
  -h, --help            show this help message and exit
  --upscale_factor UPSCALE_FACTOR
                        super resolution upscale factor. default is 3.
  --batch_size BATCH_SIZE
                        training batch size, per device. default is 4.
  --test_batch_size TEST_BATCH_SIZE
                        test batch size
  --epochs EPOCHS       number of training epochs
  --lr LR               learning Rate. default is 0.001.
  --use-gpu             whether to use GPU.
  --seed SEED           random seed to use. Default=123
  --resolve_img RESOLVE_IMG
                        input image to use
```

Once the network is trained you can use the following command to increase the resolution of your image:
```
python  super_resolution.py --resolve_img myimage.jpg
```
