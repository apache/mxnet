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
Setup NDK and build your standalone toolchain. [Instructions](http://developer.android.com/ndk/guides/standalone_toolchain.html#itc) Use the Advanced Method!!!
Build OpenBlas for Android: [Build OpenBlas](https://github.com/xianyi/OpenBLAS/wiki/How-to-build-OpenBLAS-for-Android)
Type ```make ANDROID=1```
Build will FAIL the first time as expected. Before this issue gets fixed, you need to manually remove include statements from mxnext_predict-all.cc and add <common.h> back. Here is the diff you want to apply:

```
19,33c19
< #include <bits/exception_ptr.h>
< #include <bits/nested_exception.h>
< #include <c_asm.h>
< #include <common_alpha.h>
< #include <common_arm64.h>
< #include <common_ia64.h>
< #include <common_linux.h>
< #include <common_mips64.h>
< #include <common_power.h>
< #include <common_quad.h>
< #include <common_reference.h>
< #include <common_sparc.h>
< #include <common_thread.h>
< #include <common_x86.h>
< #include <common_x86_64.h>
---
> #include <common.h>
35d20
< #include <config_kernel.h>
41,42d25
< #include <machine/ansi.h>
< #include <machine/builtins.h>
44d26
< #include <packet/sse-inl.h>
51,55d32
< #include <sys/_types.h>
< #include <sys/mman.h>
< #include <sys/shm.h>
< #include <sys/time.h>
< #include <thread.h>
60d36
< #include <windows.h>

```

Type ```make ANDROID=1``` again to continue the build.

In most cases you will want to use jni_libmxnet_predict.so. It contains the JNIs. In case you want to build your own JNI, link with libmxnet_predict.o
