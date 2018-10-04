# MXNet C++ Package Examples

## Building C++ examples

The examples are built while building the MXNet library and cpp-package from source . However, they can be built manually as follows

From cpp-package/examples directory

-  Build all examples in release mode: **make all**
-  Build all examples in debug mode: **make debug**

By default, the examples are build to be run on GPU. To build examples to run on CPU:

-  Release: **make all MXNET\_USE\_CPU=1**
-  Debug: **make debug MXNET\_USE\_CPU=1**

The examples that are build to be run on GPU may not work on the non-GPU machines.
The makefile will also download the necessary data files and store in data folder. (The download will take couple of minutes, but will be done only once on a fresh installation.)


## Examples

This directory contains following examples. In order to run the examples, ensure that the path to the MXNet shared library is added to the OS specific environment variable such as _LD\_LIBRARY\_PATH_ .

### [alexnet.cpp](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/alexnet.cpp>)

The example implements C++ version of AlexNet. The networks trains the MNIST data. The number of epochs can be specified as command line arguement. For example:
	```
	./alexnet 10
	```

### [charRNN.cpp](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/chaRNN.cpp>)

The code implements C++ version charRNN for mxnet\example\rnn\char-rnn.ipynb with MXNet.cpp API. The generated params file is compatiable with python version. The train() and predict() has been verified with original data samples.

The example expects arguments as follows:

```
	./charRNN train [BuildIn\ [TImeMajor] {corpus file} { batch size} { max epoch} [{starting epoch}]
	./charRNN predict [BuildIn\ [TImeMajor] {param file} { batch size} { max epoch} [{starting epoch}]
```
 
### [googlenet.cpp](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/googlenet.cpp>)

The code implements GoogLeNet/Inception network using C++ API. The example uses MNIST data to train the network. The number of epochs can be specified in the command line as follows. If not specified, the model trains for 100 epochs.

```
./googlenet 10
```

### [mlp.cpp](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/mlp.cpp>)

The code implements multilayer perceptron from scratch. The example creates its own dummy data to train the model. The example does not require command line parameters. It trains the model for 20000 iterations.

```
./mlp
```

### [mlp_cpu.cpp](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/mlp_cpu.cpp>)

The code implements multilayer perceptron to train the MNIST data. The code demonstrates the use of "SimpleBind"  C++ API and MNISTIter. The example is designed to work on CPU. The example does not require command line parameters.

```
./mlp_cpu
```

### [mlp_gpu.cpp](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/mlp_gpu.cpp>)
The code implements multilayer perceptron to train the MNIST data. The code demonstrates the use of "SimpleBind"  C++ API and MNISTIter. The example is designed to work on GPU. The example does not require command line paratmeters.

```
./mlp_gpu
```

### [mlp_csv.cpp](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/mlp_csv.cpp>)
The code implements multilayer perceptron to train the MNIST data. The code demonstrates the use of "SimpleBind"  C++ API and CSVIter. The CSVIter can iterate data that is in CSV format. The example can be run on CPU or GPU. The example usage is as follows:

```
mlp_csv --train mnist_training_set.csv --test mnist_test_set.csv --epochs 10 --batch_size 100 --hidden_units "128,64,64 [--gpu]"
```

### [resnet.cpp](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/resnet.cpp>)

The code implements resnet model using C++ API. The model is used to train MNIST data. The number of epochs for training the model can be specified on the command line. By default, model is trained for 100 epochs.

```
./resnet 10
```

### [lenet.cpp](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/lenet.cpp>)

The code implements lenet model using C++ API. It uses MNIST training data in CSV format to train the network. The example does not use built-in CSVIter to read the data from CSV file. The number of epochs can be specified on the command line. By default, the mode is trained for 100000 epochs.

```
./lenet 10
```
### [lenet\_with\_mxdataiter.cpp](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/mlp_cpu.cpp>)

The code implements lenet model using C++ API. It uses MNIST training data to train the network. The example uses built-in MNISTIter to read the data. The number of epochs can be specified on the command line. By default, the mode is trained for 100 epochs.

```
./lenet\_with\_mxdataiter 10
```

In addition, there is `run_lenet_with_mxdataiter.sh` that downloads the mnist data and run `lenet_with_mxdataiter` example.

###[inception_bn.cpp](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/inception_bn.cpp>)

The code implements Inception network using C++ API with batch normalization. The example uses MNIST data to train the network. The model trains for 100 epochs.

```
./inception_bn
```
