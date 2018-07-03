# MXNet Scala Sample Project

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
The Scala file is being exectuted using Java. You can execute the helloWorld example as follows:
```Bash
java -Xmx8G  -cp $CLASSPATH sample.HelloWorld
```
However, you have to define the Classpath before you run the demo code. More information can be found in the `demo.sh` And you can run the bash script as follows:
```Bash
bash demo.sh
```
It will load the library automatically and run the example

## Clean up
Clean up for Maven package is simple, you can run the pre-configed `Makefile` as:
```Bash
make scalaclean
```