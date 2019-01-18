# MXNet Java Sample Project
This is an project created to use Maven-published Scala/Java package with two Java examples.
## Setup
You can use the `Makefile` to make the Java package. Simply do the following:
```Bash
make javademo
```
This will load the default parameter for all the environment variable.
If you want to run with GPU on Linux, just simply add `USE_CUDA=1` when you run the make file
 
You can use the following instruction as an alternative to achieve the same result:
User are required to use `mvn package` to build the package,
 which are shown below:
```Bash
export SCALA_VERSION_PROFILE=2.11
export SCALA_PKG_PROFILE=
mvn package -Dmxnet.profile=$SCALA_PKG_PROFILE \
		-Dmxnet.scalaprofile=$SCALA_VERSION_PROFILE
```
These environment variable (`SCALA_PKG_PROFILE`, `SCALA_VERSION_PROFILE`)
should be set before executing the line above.
The `SCALA_PKG_PROFILE` should be chosen from `osx-x86_64-cpu`, `linux-x86_64-cpu` or `linux-x86_64-gpu`.


## Run
### Hello World
The Scala file is being executed using Java. You can execute the helloWorld example as follows:
```Bash
bash bin/java_sample.sh
```
You can also run the following command manually:
```Bash
java -cp $CLASSPATH sample.HelloWorld
```
However, you have to define the Classpath before you run the demo code. More information can be found in the `java_sample.sh`.
The `CLASSPATH` should point to the jar file you have downloaded.

It will load the library automatically and run the example

In order to use the `Param Object`. We requires user to place this line in the front:
```
static NDArray$ NDArray = NDArray$.MODULE$;
```
It would help to have the NDArray companion object static and accessable from the outside.

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
bash bin/run_od.sh
```
or the command below as an alternative
```Bash
java -cp $CLASSPATH sample.ObjectDetection
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

Is there any other version available?

You can find nightly release version from [here](https://repository.apache.org/#nexus-search;gav~org.apache.mxnet~~1.5.0-SNAPSHOT~~).
Please keep the same version in the Makefile or [above version](https://repository.apache.org/#nexus-search;gav~org.apache.mxnet~~~~) to run this demo.
