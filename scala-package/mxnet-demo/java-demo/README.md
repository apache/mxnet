# MXNet Java Sample Project
This is an project created to use Maven-published Scala/Java package with two Java examples.
## Setup
Please copy the downloaded MXNet Java package jar file under the `java-demo` folder.

User are required to use `mvn package` to build the package,
 which are shown below:
```Bash
export SCALA_VERSION_PROFILE=2.11 SCALA_VERSION=2.11.8 MXNET_VERSION=1.3.1-SNAPSHOT
export SCALA_PKG_PROFILE=
mvn package -Dmxnet.profile=$(SCALA_PKG_PROFILE) \
		-Dmxnet.scalaprofile=$(SCALA_VERSION_PROFILE) \
		-Dmxnet.version=$(MXNET_VERSION) \
		-Dscala.version=$(SCALA_VERSION)
```
These environment variable (`SCALA_PKG_PROFILE`, `SCALA_VERSION_PROFILE`, `MXNET_VERSION`, `SCALA_VERSION`)
should be set before executing the line above.
 
You can also use the `Makefile` as an alternative to do the same thing. Simply do the following:
```Bash
make javademo
```
This will load the default parameter for all the environment variable.
 If you want to run with GPU on Linux, just simply add `USE_CUDA=1` when you run the make file

## Run
### Hello World
The Scala file is being executed using Java. You can execute the helloWorld example as follows:
```Bash
java -cp $CLASSPATH sample.HelloWorld
```
However, you have to define the Classpath before you run the demo code. More information can be found in the `demo.sh` And you can run the bash script as follows:
```Bash
bash bin/java_sample.sh
```
It will load the library automatically and run the example
### Object Detection using Inference API
We also provide an example to do object detection, which downloads a ImageNet trained resnet50 model and runs inference on an image to return the classification result as
```Bash
Class: car
Probabilties: 0.99847263
Coord:312.21335, 72.02908, 456.01443, 150.66176
Class: bicycle
Probabilties: 0.9047381
Coord:155.9581, 149.96365, 383.83694, 418.94516
Class: dog
Probabilties: 0.82268167
Coord:83.82356, 179.14001, 206.63783, 476.78754
```

you can run using the command shown below:
```Bash
java -cp $CLASSPATH sample.ObjectDetection
```
or script as follows:
```Bash
bash bin/run_od.sh
```

If you want to test run on GPU, you can set a environment variable as follows:
```Bash
export SCALA_TEST_ON_GPU=1
```
## Clean up
Clean up for Maven package is simple, you can run the pre-configed `Makefile` as:
```Bash
make javaclean
```

## Q & A
If you are facing opencv issue on Ubuntu, please try as follows to install opencv 3.4 (required by 1.2.0 package and above)
```Bash
sudo add-apt-repository ppa:timsc/opencv-3.4
sudo apt-get update
sudo apt install libopencv-imgcodecs3.4
```