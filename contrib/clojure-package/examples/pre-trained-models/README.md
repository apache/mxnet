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

# pre-trained-models

This shows examples of how to use the pretrained models. MXNet comes with a number of pretrained models
https://mxnet.incubator.apache.org/model_zoo/index.html


## Predict Image from pretrained models

From the example on https://mxnet.incubator.apache.org/tutorials/python/predict_image.html


The `predict-image.clj` file loads up the pre-trained resnet-152 model and uses it to predict the classifications from images on the internet

*To use run download-reset-152.sh to get the model params and json *


## Fine Tune from pretrained models

From the finetune example https://mxnet.incubator.apache.org/faq/finetune.html

The `fine-tune.clj` file loads up the samller resnet-50 model and adds a fine tune layer to reclassify the caltech iamge set

*To use run download-resnet-50.sh to get the model params and json and download-caltech.sh to get the pregenerated rec files*

You can run the fine tune example by doing `lein run` (cpu)

You can control the devices you run on by doing:

`lein run :cpu 2` - This will run on 2 cpu devices
`lein run :gpu 1` - This will run on 1 gpu device
`lein run :gpu 2` - This will run on 2 gpu devices



