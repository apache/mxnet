MXNet Amalgamation
==================
This folder contains a amalgamation generation script to generate the entire mxnet library into one file.
Currently it supports generation for [predict API](../include/mxnet/c_predict_api.h),
which allows you to run prediction in platform independent way.

How to Generate the Amalgamation
--------------------------------
Type ```make``` will generate the following files
- mxnet_predict-all.cc
  - The file you can used to compile predict API
- ../lib/libmxnet_predict.so
  - The dynamic library generated for prediction.

You can also checkout the [Makefile](Makefile)

Dependency
----------
The only dependency is a BLAS library.

Acknowledgement
---------------
This module is created by [Jack Deng](https://github.com/jdeng).

Android
---------------
Setup NDK and build your standalone toolchain. [Instructions](http://developer.android.com/ndk/guides/standalone_toolchain.html#itc) Use the Advanced Method!!! In particular set PATH, CC and CXX.

Example:
```
export PATH=/tmp/my-android-toolchain/bin:$PATH
export CC=arm-linux-androideabi-gcc   # or export CC=arm-linux-androideabi-clang
export CXX=arm-linux-androideabi-g++  # or export CXX=arm-linux-androideabi-clang++
```

Build OpenBlas for Android: [Build OpenBlas](https://github.com/xianyi/OpenBLAS/wiki/How-to-build-OpenBLAS-for-Android)
Modify OPENBLAS_ROOT in Makefile
Type ```make ANDROID=1```

In most cases you will want to use jni_libmxnet_predict.so. It contains the JNIs. In case you want to build your own JNI, link with libmxnet_predict.o

You can use generated library in [Leliana WhatsThis Android app](https://github.com/Leliana/WhatsThis). Rename jni_libmxnet_predict.so to libmxnet_predict.so and overwrite default library to use up-to-date mxnet version.

Javascript
---------------
JS version works without OpenBLAS. You need [emscripten](http://kripken.github.io/emscripten-site/) to build it.
Type ```make MIN=1 libmxnet_predict.js```

You can use generated library in [mxnet.js](https://github.com/dmlc/mxnet.js)

iOS
---------------
[Chinese guide](http://www.liuxiao.org/2015/12/ios-mxnet-%E7%9A%84-ios-%E7%89%88%E6%9C%AC%E7%BC%96%E8%AF%91/)

Build OpenBlas for host machine [Instructions](https://github.com/xianyi/OpenBLAS/wiki/Installation-Guide)
Modify OPENBLAS_ROOT in Makefile.
Type ```make```
If the build process is successful you will see the following output:
```ar rcs libmxnet_predict.a mxnet_predict-all.o```

Modify mxnet_predict-all.cc:

If present comment
```
#include <cblas.h>
```

Add
```
#include <Accelerate/Accelerate.h>
```

Comment all occurences of
```
#include <emmintrin.h>
```

Change
```
#if defined(__ANDROID__) || defined(__MXNET_JS__)
#define MSHADOW_USE_SSE         0
#endif
```

To
```
#define MSHADOW_USE_SSE         0
```

Change
```
#ifdef __GNUC__
  #define MX_TREAD_LOCAL __thread
#elif __STDC_VERSION__ >= 201112L
  #define  MX_TREAD_LOCAL _Thread_local
#elif defined(_MSC_VER)
  #define MX_TREAD_LOCAL __declspec(thread)
#endif
```

To
```
#define MX_TREAD_LOCAL __declspec(thread)
```

**To build arm32 compatible version (e.g. iPhone 5):**

Change
```
typedef mxnet::common::ThreadLocalStore<ErrorEntry> MXAPIErrorStore;

const char *MXGetLastError() {
  return MXAPIErrorStore::Get()->last_error.c_str();
}

void MXAPISetLastError(const char* msg) {
  MXAPIErrorStore::Get()->last_error = msg;
}
```

To
```
//typedef mxnet::common::ThreadLocalStore<ErrorEntry> MXAPIErrorStore;

const char *MXGetLastError() {
  //return MXAPIErrorStore::Get()->last_error.c_str();
  return "";
}

void MXAPISetLastError(const char* msg) {
  //MXAPIErrorStore::Get()->last_error = msg;
  (void) msg;
}
```

You can use modified mxnet_predict-all.cc in [PPPOE WhatsThis iOS app](https://github.com/pppoe/WhatsThis-iOS).

