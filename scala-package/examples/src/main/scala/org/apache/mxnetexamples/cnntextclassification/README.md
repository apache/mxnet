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

# CNN Text Classification Example for Scala
This is the example using Scala type-safe api doing CNN text classification. 
This example is only for Illustration and not modeled to achieve the best accuracy.

Please contribute to improve the dev accuracy of the model.

## Setup

Please configure your maven project using our latest release. An tutorial to do that can be found here:
[IntelliJ IDE (or alternative IDE) project setup](http://mxnet.incubator.apache.org/tutorials/scala/mxnet_scala_on_intellij.html)

### Download the training files
```$xslt
https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/CNN/rt-polarity.pos
https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/CNN/rt-polarity.neg
```
### Download pretrained Word2Vec Model
I used the SLIM version, you can try with the full version to see if the accuracy can improve
```$xslt
https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/CNN/GoogleNews-vectors-negative300-SLIM.bin
```
### Train the model
Please configure the [args](https://github.com/apache/incubator-mxnet/blob/scala-package/examples/src/main/scala/org/apache/mxnet/examples/cnntextclassification/CNNTextClassification.scala#L299-L312) required for the model here and then run it.