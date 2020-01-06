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

# Benchmarking Scala Inference APIs 

This folder contains a base class [ScalaInferenceBenchmark](https://github.com/apache/incubator-mxnet/tree/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/benchmark/) and provides a mechanism for benchmarking [MXNet Inference APIs]((https://github.com/apache/incubator-mxnet/tree/master/scala-package/infer)) in Scala.
The benchmarking scripts provided runs an experiment for single inference calls and batch inference calls. It collects the time taken to perform an inference operation and emits the P99, P50 and Average values for these metrics.  One can easily add/modify any new/existing examples to the ScalaInferenceBenchmark framework in order to get the benchmark numbers for inference calls.
Currently the ScalaInferenceBenchmark script supports three Scala examples : 
1. [ImageClassification using ResNet-152](https://github.com/apache/incubator-mxnet/blob/master/scala-package/mxnet-demo/src/main/scala/sample/ImageClassificationExample.scala)
2. [Object Detection Example](https://github.com/apache/incubator-mxnet/blob/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/infer/objectdetector/SSDClassifierExample.scala)
3. [Text Generation through RNNs](https://github.com/apache/incubator-mxnet/blob/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/rnn/TestCharRnn.scala)

This script can be easily placed in an automated environment to run benchmark regressions on the Scala APIs. The script automatically picks up whether you are running it on a CPU machine or on a GPU machine and appropriately uses that.

## Contents

1. [Prerequisites](#prerequisites)
2. [Scripts](#scripts)

## Prerequisites

1. MXNet
2. MXNet Scala Package
3. [IntelliJ IDE (or alternative IDE) project setup](https://mxnet.apache.org/api/scala/docs/tutorials/mxnet_scala_on_intellij) with the MXNet Scala Package
4. Model files and datasets for the model one will try to benchmark

## Scripts
To help you easily run the benchmarks, a starter shell script has been provided for each of three examples mentioned above. The scripts can be found [here](https://github.com/apache/incubator-mxnet/blob/master/scala-package/examples/scripts/benchmark).
Each of the script takes some parameters as inputs, details of which can be found either in the bash scripts or in the example classes itself. 

* *ImageClassification Example*
<br> The following shows an example of running ImageClassifier under the benchmark script. The script takes as parameters, the platform type (cpu/gpu), number of iterations for inference calls, the batch size for batch inference calls, the model path, input file, and input directory. 
For more details to run ImageClassificationExample as a standalone file, refer to the [README](https://github.com/apache/incubator-mxnet/blob/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/infer/imageclassifier/README.md) for ImageClassifierExample.
You may need to run ```chmod u+x run_image_inference_bm.sh``` before running this script.
    ```bash
    cd <Path-To-MXNET-Repo>/scala-package/examples/scripts/infer/imageclassifier
    ./get_resnet_data.sh
    cd <Path-To-MXNET-Repo>/scala-package/examples/scripts/benchmark
    ./run_image_inference_bm.sh gpu ImageClassifierExample 100 10 ../infer/models/resnet-152/resnet-152 ../infer/images/kitten.jpg ../infer/images/
    ./run_image_inference_bm.sh cpu ImageClassifierExample 100 10 ../infer/models/resnet-152/resnet-152 ../infer/images/kitten.jpg ../infer/images/
    ```
    Upon running this script, you might see an output like this : 
    ```
    [main] INFO org.apache.mxnetexamples.benchmark.CLIParserBase - 
    single_inference_latency p99 1663, single_inference_p50 729, single_inference_average 755.17
    ...
        
    INFO org.apache.mxnetexamples.benchmark.CLIParserBase - 
    batch_inference_latency p99 4241, batch_inference_p50 4241, batch_inference_average 4241.00
    ```

* *Object Detection Example*
<br> The following shows an example of running SSDClassifier under the benchmark script. The script takes in the number of iterations for inference calls, the batch size for batch inference calls, the model path, input file, and input directory. 
For more details to run SSDClassifierExample as a standalone file, refer to the [README](https://github.com/apache/incubator-mxnet/blob/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/infer/objectdetector/README.md) for SSDClassifierExample.
You may need to run ```chmod u+x run_image_inference_bm.sh``` before running this script.
    ```bash
    cd <Path-To-MXNET-Repo>/scala-package/examples/scripts/infer/objectdetector
    ./get_ssd_data.sh
    cd <Path-To-MXNET-Repo>/scala-package/examples/scripts/benchmark
    ./run_image_inference_bm.sh cpu ObjectDetectionExample 100 10 ../infer/models/resnet50_ssd/resnet50_ssd_model ../infer/images/dog.jpg ../infer/images/ 
    ```
    Upon running this script, you might see an output like this : 
    ```
    [main] INFO org.apache.mxnetexamples.benchmark.CLIParserBase - 
    single_inference_latency p99 1663, single_inference_p50 729, single_inference_average 755.17
    ...
    
    INFO org.apache.mxnetexamples.benchmark.CLIParserBase - 
    batch_inference_latency p99 4241, batch_inference_p50 4241, batch_inference_average 4241.00
    ```
    
* *Text Generation through RNNs*
<br>The following shows an example of running TestCharRnn under the benchmark script. The script takes in the number of iterations for inference calls, the model path and the input text file. 
For more details to run TestCharRnn as a standalone file, refer to the [README](https://github.com/apache/incubator-mxnet/blob/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/rnn/README.md) for TextCharRnn.
You may need to run ```chmod u+x run_text_charrnn_bm.sh``` before running this script.
    ```bash
    wget https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/RNN/obama.zip
    unzip obama.zip
    cd <Path-To-MXNET-Repo>/scala-package/examples/scripts/benchmark
    ./run_text_charrnn_bm.sh cpu CharRnn 100 <path-to-model>/obama <path-to-model>/obama.txt 
    ```
    Upon running this script, you might see an output like this : 
    ```
    [main] INFO org.apache.mxnetexamples.benchmark.CLIParserBase - 
    single_inference_latency p99 4097, single_inference_p50 2560, single_inference_average 2673.720000 
    ```
