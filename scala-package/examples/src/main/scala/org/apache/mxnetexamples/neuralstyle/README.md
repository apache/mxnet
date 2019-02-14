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

# Neural Style Example for Scala

## Introduction
This model contains three important components:
- Boost Inference
- Boost Training
- Neural Style conversion

You can use the prebuilt VGG model to do the conversion.
By adding a style image, you can create several interesting images.

Original Image            |  Style Image
:-------------------------:|:-------------------------:
![](https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/NeuralStyle/IMG_4343.jpg)  |  ![](https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/NeuralStyle/starry_night.jpg)

Boost Inference Image (pretrained)           |  Epoch 150 Image
:-------------------------:|:-------------------------:
![](https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/NeuralStyle/out_3.jpg)  |  ![](https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/NeuralStyle/tmp_150.jpg)

## Setup
Please download the input image and style image following the links below:

Input image
```bash
https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/NeuralStyle/IMG_4343.jpg
```
Style image
```bash
https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/NeuralStyle/starry_night.jpg
```

VGG model --Boost inference
```bash
https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/NeuralStyle/model.zip
```

VGG model --Boost Training
```bash
https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/NeuralStyle/vgg19.params
```

Please unzip the model before you use it.

## Boost Inference Example

Please provide the corresponding arguments before you execute the program
```bash
--input-image
<path>/IMG_4343.jpg
--model-path
<path>/model
--output-path
<outputPath>
```

## Boost Training Example
Please download your own training data for boost training.
You can use 26k images sampled from [MIT Place dataset](http://places.csail.mit.edu/).
```bash
--style-image
<path>/starry_night.jpg
--data-path
<path>/images
--vgg-model-path
<path>/vgg19.params
--save-model-path
<path>
```

## NeuralStyle Example
Please provide the corresponding arguments before you execute the program
```bash
--model-path
<path>/vgg19.params
--content-image
<path>/IMG_4343.jpg
--style-image
<path>/starry_night.jpg
--gpu
<num_of_gpus>
--output-dir
<path>
```