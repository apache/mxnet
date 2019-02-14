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

This is the R version of [captcha recognition](http://blog.xlvector.net/2016-05/mxnet-ocr-cnn/) example by xlvector and it can be used as an example of multi-label training. For a captcha below, we consider it as an image with 4 labels and train a CNN over the data set.

![](captcha_example.png)

You can download the images and `.rec` files from [here](https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/data/captcha_example.zip). Since each image has 4 labels, please remember to use `label_width=4` when generating the `.rec` files.
