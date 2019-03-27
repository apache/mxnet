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

# Image Classification using Java Predictor

In this example, you will learn how to use Java Inference API to 
build and run pre-trained Resnet 18 model.

## Contents

1. [Prerequisites](#prerequisites)
2. [Download artifacts](#download-artifacts)
3. [Setup datapath and parameters](#setup-datapath-and-parameters)
4. [Run the image classifier example](#run-the-image-inference-example)

## Prerequisites

1. Build from source with [MXNet](https://mxnet.incubator.apache.org/install/index.html)
2. [IntelliJ IDE (or alternative IDE) project setup](https://github.com/apache/incubator-mxnet/blob/master/docs/tutorials/java/mxnet_java_on_intellij.md) with the MXNet Java Package
3. wget

## Download Artifacts

For this tutorial, you can get the model and sample input image by running following bash file. This script will use `wget` to download these artifacts from AWS S3.

From the `scala-package/examples/scripts/infer/imageclassifier/` folder run:

```bash
./get_resnet_18_data.sh
```

**Note**: You may need to run `chmod +x get_resnet_18_data.sh` before running this script.

### Setup Datapath and Parameters

The available arguments are as follows:

| Argument                      | Comments                                 |
| ----------------------------- | ---------------------------------------- |
| `model-dir`                   | Folder path with prefix to the model (including json, params, and any synset file). |
| `input-image`                 | The image to run inference on. |

## Run the image classifier example

After the previous steps, you should be able to run the code using the following script that will pass all of the required parameters to the Predictor API.

From the `scala-package/examples/scripts/infer/predictor/` folder run:

```bash
bash run_predictor_java_example.sh ../models/resnet-18/resnet-18 ../images/kitten.jpg
```

**Notes**:
* These are relative paths to this script.
* You may need to run `chmod +x run_predictor_java_example.sh` before running this script.

The example should give an output similar to the one shown below:
```
Predict with Float input
Probability : 0.30337515 Class : n02123159 tiger cat
Predict with NDArray
Probability : 0.30337515 Class : n02123159 tiger cat
```
the outputs come from the the input image, with top1 predictions picked.