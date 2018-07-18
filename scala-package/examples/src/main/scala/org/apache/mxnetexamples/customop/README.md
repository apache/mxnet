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