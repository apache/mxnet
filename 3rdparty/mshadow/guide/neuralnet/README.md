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
