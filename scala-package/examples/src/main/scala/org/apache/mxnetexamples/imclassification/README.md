# MNIST Example for Scala
This is the MNIST Training Example implemented for Scala type-safe api
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
--data-dir <location of your downloaded file>
```
You can find more information [here](https://github.com/apache/incubator-mxnet/blob/scala-package/examples/src/main/scala/org/apache/mxnet/examples/imclassification/TrainMnist.scala#L169-L207)