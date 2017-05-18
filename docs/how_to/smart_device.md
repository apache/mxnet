# Deep Learning in a Single File for Smart Devices

Deep learning (DL) systems are complex and often depend on a number of libraries.
Porting a DL library to different platforms can be painful, especially for smart devices.
One simple solution to this problem is to provide a light interface to the library, complete with all required code in a single file with minimal dependencies.
In this document, we explain how to amalgamate all necessary code into a single file,
and demonstrate the approach with an example in which we run image recognition on a mobile device.

## Amalgamation: Making the Whole System a Single File

We come to the idea of of amalgamation following the example of SQLite,
which pack all the code needed to run a simple database into a single source file.
All that's necessary to create the library is to compile that single file.
This simplifies the problem of porting to various platforms.

Thanks to [Jack Deng](https://github.com/jdeng),
MXNet provides an [amalgamation](https://github.com/dmlc/mxnet/tree/master/amalgamation) script
that compiles all code needed for prediction based on trained DL models into a single `.cc` file,
containing approximately 30K lines of code. This code only depends on the BLAS library.
Moreover, we've also created an even more minimal version,
with the BLAS dependency removed.
You can compile the single file into JavaScript by using [emscripten](https://github.com/kripken/emscripten).

The compiled library can be used by any other programming language.
The `.h` file contains a light prediction API.
Porting to another language with a C foreign function interface requires little effort.

For examples, see the following examples on GitHub:

- Go: [https://github.com/jdeng/gomxnet](https://github.com/jdeng/gomxnet)
- Java: [https://github.com/dmlc/mxnet/tree/master/amalgamation/jni](https://github.com/dmlc/mxnet/tree/master/amalgamation/jni)
- Python: [https://github.com/dmlc/mxnet/tree/master/amalgamation/python](https://github.com/dmlc/mxnet/tree/master/amalgamation/python)


If you plan to amalgamate your system, there are a few guidelines you ought to observe when building the project:

- Minimize dependence on other libraries.
- Use namespace to encapsulate the types and operators.
- Avoid running commands such as ```using namespace xyz``` on the global scope.
- Avoid cyclic include dependencies.


## Image Recognition Demo on Mobile Devices

With amalgamation, deploying the system on smart devices (such as Android or iOS) is simple. But there are two additional considerations:

- The model should be small enough to fit into the device's memory.
- The model shouldn't be too expensive to run given the relatively low computational power of these devices.

Let's use image recognition as an example.
We start with the state-of-the-art inception model.
We train it on an ImageNet dataset,
using multiple servers with GTX 980 cards.
The resulting model fits into memory,
but it's too expensive to run.
We remove some layers, but now the results are poor.

Finally, we show an Android example, thanks to Leliana, [https://github.com/Leliana/WhatsThis](https://github.com/Leliana/WhatsThis) to demonstrate how to run on Android.

<img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/apk/subinception.png" height="488" width="274">


By using amalgamation, we can easily port the prediction library to mobile devices, with nearly no dependency.
After compiling the library for smart platforms, the last thing we must do is to call C-API in the target language (Java/Swift).

<img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/apk/poolnet.png" height="488" width="274">

Besides this pre-trained Inception-BatchNorm network, we've provided two pre-trained models.

We tested our model on a Nexus 5:


|                  | Top-1 Validation on ILSVRC2012      | Time  | App Size  | Runtime Temp Memory Req |
| ---------------- | ----------------------------------- | ----- | --- | ------------ |
| FastPoorNet      | around 52%, similar to 2011 winner  | 1s    | <10MB    |  <5MB               |
| Sub InceptionBN  | around 64%, similar to 2013 winner  | 2.7s  | <40MB    |  <10MB              |
| InceptionBN      | around 70%                          | 4s-5s | <60MB    | 10MB               |

These models are for demonstration only.
They aren't fine-tuned for mobile devices,
and there is definitely room for improvement.  
We believe that making a lightweight, portable,
and fast deep learning library is fun and interesting,
and hope you enjoy using the library.

## Source Code
[https://github.com/Leliana/WhatsThis](https://github.com/Leliana/WhatsThis)


## Demo APK Download

- [FastPoorNet](https://github.com/dmlc/web-data/blob/master/mxnet/apk/fastpoornet.apk?raw=true)


- [SubInception](https://github.com/dmlc/web-data/blob/master/mxnet/apk/subinception.apk?raw=true)
