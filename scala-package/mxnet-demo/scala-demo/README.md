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

# MXNet Scala Sample Project
This is an project created to use Maven-published Scala package with two Scala examples.
## Setup
You are required to use maven to build the package, by running the following:
```
mvn package
```
This command will pick the default values specified in the pom file.

Note: If you are planning to use GPU, please add `-Dmxnet.profile=linux-x86_64-gpu`

### Use customized version set
 which are shown below:
```Bash
export SCALA_VERSION_PROFILE=2.11 SCALA_VERSION=2.11.8
export SCALA_PKG_PROFILE=
mvn package -Dmxnet.profile=$(SCALA_PKG_PROFILE) \
		-Dmxnet.scalaprofile=$(SCALA_VERSION_PROFILE) \
		-Dscala.version=$(SCALA_VERSION)
```
These environment variable (`SCALA_PKG_PROFILE`, `SCALA_VERSION_PROFILE`, `SCALA_VERSION`)
should be set before executing the line above.

To obtain the most recent MXNet version, please click [here](https://mvnrepository.com/search?q=org.apache.mxnet)

## Run
### Hello World
The Scala file is being executed using Java. You can execute the helloWorld example as follows:
```Bash
java -cp $CLASSPATH sample.HelloWorld
```
However, you have to define the Classpath before you run the demo code. More information can be found in the `demo.sh` And you can run the bash script as follows:
```Bash
bash bin/demo.sh
```
It will load the library automatically and run the example
### Image Classification using Inference API
We also provide an example to do image classification, which downloads a ImageNet trained resnet18 model and runs inference on a cute puppy to return the classification result as
```Bash
Classes with top 5 probability = Vector((n02110958 pug, pug-dog,0.49161583), (n02108422 bull mastiff,0.40025946), (n02108089 boxer,0.04657662), (n04409515 tennis ball,0.028773671), (n02109047 Great Dane,0.009004086)) 
```
You can review the complete example [here](https://github.com/apache/incubator-mxnet/tree/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/infer/imageclassifier)

you can run using the command shown below:
```Bash
java -cp $CLASSPATH sample.ImageClassificationExample
```
or script as follows:
```Bash
bash bin/run_im.sh
```

If you want to test run on GPU, you can set a environment variable as follows:
```Bash
export SCALA_TEST_ON_GPU=1
```
## Clean up
To clean up a Maven package, run the following:
```Bash
mvn clean
```

## Q & A
If you are facing opencv issue on Ubuntu, please try as follows to install opencv 3.4 (required by 1.2.0 package and above)
```Bash
sudo add-apt-repository ppa:timsc/opencv-3.4
sudo apt-get update
sudo apt install libopencv-imgcodecs3.4
```