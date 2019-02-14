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

# Custom Operator Example for Scala
This is the example using Custom Operator for type-safe api of Scala.
In the example, a `Softmax` operator is implemented to run the MNIST example.

There is also an example using RTC. However, the rtc module is depreciated and no longer can be used. Please contribute to use CudaModule operator to replace the rtc.

## Setup
### Download the source File
```$xslt
https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/mnist/mnist.zip
```
### Unzip the file
```$xslt
unzip mnist.zip
```
### Arguement Configuration
Then you need to define the arguments that you would like to pass in the model:
```$xslt
--data-path <location of your downloaded file>
```
 
you can find more in [here](https://github.com/apache/incubator-mxnet/blob/scala-package/examples/src/main/scala/org/apache/mxnet/examples/customop/ExampleCustomOp.scala#L218-L221)