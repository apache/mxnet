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

# Mulit-task learning example
 
This is a simple example to show how to use mxnet for multi-task learning. It uses MNIST as an example, trying to predict jointly the digit and whether this digit is odd or even.

For example:

![](https://camo.githubusercontent.com/ed3cf256f47713335dc288f32f9b0b60bf1028b7/68747470733a2f2f7777772e636c61737365732e63732e756368696361676f2e6564752f617263686976652f323031332f737072696e672f31323330302d312f70612f7061312f64696769742e706e67)

Should be jointly classified as 4, and Even.

In this example we don't expect the tasks to contribute to each other much, but for example multi-task learning has been successfully applied to the domain of image captioning. In [A Multi-task Learning Approach for Image Captioning](https://www.ijcai.org/proceedings/2018/0168.pdf) by Wei Zhao, Benyou Wang, Jianbo Ye, Min Yang, Zhou Zhao, Ruotian Luo, Yu Qiao, they train a network to jointly classify images and generate text captions

Please refer to the notebook for a fully worked example.
