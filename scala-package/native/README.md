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

# MXNet Scala JNI

MXNet Scala JNI is a thin wrapper layer of underlying libmxnet.so.

## javah
JNI native code requires a header file that matches the java/scala interface,
this file is usually generated with javah.

In our case, org_apache_mxnet_native_c.h is generated and will be used to compile native code.

To improve build performance, we check in generated org_apache_mxnet_native_c.h file.
And we added a check to detect mismatch with Scala code and generated header. The checker will
make sure we won't forget to update org_apache_mxnet_native_c.h file.


## Linker options

Scala JNI (libmxnet-scala.so/libmxnet-scala.jnilib) is dynamically linked to libmxnet.so.
MXNet Scala will trying to load libmxnet.so from system LD_LIBRARY_PATH first.
If it failed, the try to resolve libmxnet.so in the same location as libmxnet-scala.so file.

### Linux
```
-Wl,-rpath=$ORIGIN -lmxnet
```
Above option will tell system to looking for libmxnet.so from the same location.


### Mac OSX
On Mac, we have to execute install_name_tool command to change library loading path:
```bash
install_name_tool -change lib/libmxnet.so @loader_path/libmxnet.so libmxnet-scala.jnilib
```

Other linker options:
* -shared : link as shared library
* -Wl,-install_name,libmxnet-scala.jnilib : avoid use build machine's absolute path
* -framework JavaVM : Stand jni options for mac
* -Wl,-exported_symbol,_Java_* : Stand jni options for mac
* -Wl,-x : Do not put non-global symbols in the output file's symbol table.


## Compiler flags

Scala JNI code technically doesn't need on any of MXNet make flags,
however c_api.h header links to many other dependencies header file,
which requires us to add DMSHADOW_USE_MKL and DMSHADOW_USE_CUDA to compile the JNI code.
These flags are not actually used by JNI and won't impact Scala's behavior.


### Linux

```
-DMSHADOW_USE_MKL=0
-DMSHADOW_USE_CUDA=0
-O3 -DNDEBUG=1 -fPIC -msse3 -mf16c
-Wall -Wsign-compare -Wno-unused-parameter -Wno-unknown-pragmas -Wno-unused-local-typedefs
```

### Mac OSX

```
-DMSHADOW_USE_MKL=0
-DMSHADOW_USE_CUDA=0
-g -O0 -fPIC -msse3 -mf16c
-Wall -Wsign-compare -Wno-unused-parameter -Wno-unknown-pragmas -Wno-unused-local-typedefs
```
