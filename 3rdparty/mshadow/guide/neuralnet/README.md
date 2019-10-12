Example Neural Net code with MShadow
====

To compile the code, modify ```config.mk``` to the setting you like and type make
* You will need to have CUDA and  a version of BLAS

To run the demo, download  MNIST dataset from: http://yann.lecun.com/exdb/mnist/
unzip all the files into current folder

and run by  ./nnet cpu or ./nnet gpu. ./convnet cpu or ./convnet gpu

MultiGPU Version
====
* If you have two GPUs, you can run it by ```./nnet_ps gpu 0 1```.
* You can run it using CPUs ```./nnet_ps cpu 0 1```.
* This is an demonstration of mshadow-ps interface, see introduction in [../mshadow-ps](../mshadow-ps)
