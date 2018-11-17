# MXNet Scala Sample Project
This is an project created to use Maven-published Scala package with two Scala examples.
## Setup
User are required to use `mvn package` to build the package,
 which are shown below:
```Bash
export SCALA_VERSION_PROFILE=2.11 SCALA_VERSION=2.11.8 MXNET_VERSION=1.2.0
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
make scalademo
```
This will load the default parameter for all the environment variable.
 If you want to run with GPU on Linux, just simply add `USE_CUDA=1` when you run the make file

## Run
### Hello World
The Scala file is being executed using Java. You can execute the helloWorld example as follows:
```Bash
java -Xmx8G  -cp $CLASSPATH sample.HelloWorld
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
java -Xmx8G  -cp $CLASSPATH sample.ImageClassificationExample
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
Clean up for Maven package is simple, you can run the pre-configed `Makefile` as:
```Bash
make scalaclean
```

## Q & A
If you are facing opencv issue on Ubuntu, please try as follows to install opencv 3.4 (required by 1.2.0 package)
```Bash
sudo add-apt-repository ppa:timsc/opencv-3.4
sudo apt-get update
sudo apt install libopencv-imgcodecs3.4
```