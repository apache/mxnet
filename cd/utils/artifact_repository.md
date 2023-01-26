<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
  ~
-->

# Artifact Repository - Pushing and Pulling libmxnet

The artifact repository is an S3 bucket accessible only to restricted Jenkins nodes. It is used to store compiled MXNet artifacts that can be used by downstream CD pipelines to package the compiled libraries for different delivery channels (e.g. DockerHub, PyPI, Maven, etc.). The S3 object keys for the files being posted will be prefixed with the following distinguishing characteristics of the binary: branch, commit id, operating system, variant and dependency linking strategy (static or dynamic). For instance, s3://bucket/73b29fa90d3eac0b1fae403b7583fdd1529942dc/ubuntu16.04/cu102mkl/static/libmxnet.so

An MXNet artifact is defined as the following set of files:

* The compiled libmxnet.so
* License files for dependencies that required their licenses to be shipped with the binary
* Dependencies that should be shipped together with the binary. For instance, for packaging the python wheel files, some dependencies that cannot be statically linked to the library need to also be included, see here (https://github.com/apache/mxnet/blob/master/tools/pip/setup.py#L142).

The artifact_repository.py script automates the upload and download of the specified files with the appropriate S3 object keys by taking explicitly set, or automatically derived, values for the different characteristics of the artifact.

### Determining Artifact Characteristics

An mxnet compiled library, or artifact for our purposes, is identified by the following distinguishing characteristics, which when not explicitly stated, will be (as much as possible) ascertained from the environment by the artifact_repository.py script: commit id, variant, operating system, and library type.

**Commit Id**

Manually configured through the --git-sha argument. 

If not set, derived by:

1. Using the values of the MXNET_SHA environment variable, which are set during the bootstrap process for *CD* Jenkins pipelines; otherwise
2. Using the values of the GIT_COMMIT environment variable, which are set automatically by Jenkins in the *CI* pipelines; otherwise
3. Using the output of git rev-parse HEAD for the commit id; otherwise
4. Fail with error

**Operating System**

Manually configured through the --os argument.

If not set, derived through the value of sys.platform (https://docs.python.org/3/library/sys.html#sys.platform). That is:

* if, linux*, extract the ID and VERSION_ID from /etc/*release, and return a concatenated string of these values, eg. ubuntu16.04, centos7, etc.
* otherwise, return the value given by sys.platform, eg. win32, darwin, etc.

**Variant**

Manually configured through the --variant argument. The current variants are: cpu, native, cu101, cu102, cu110, cu112.

As long as the tool is being run from the MXNet code base, the runtime feature detection tool (https://github.com/larroy/mxnet/blob/dd432b7f241c9da2c96bcb877c2dc84e6a1f74d4/docs/api/python/libinfo/libinfo.md) can be used to detect whether the library has been compiled with oneDNN (library has oneDNN feature enabled) and/or CUDA support (compiled with CUDA feature enabled).

If it has been compiled with CUDA support, the output of /usr/local/cuda/bin/nvcc --version can be mined for the exact CUDA version (eg. 8.0, 9.0, etc.).

By knowing which features are enabled on the binary, and if necessary, which CUDA version is installed on the machine, the value for the variant argument can be calculated. Eg. if CUDA features are enabled, and nvcc reports cuda version 10.2, then the variant would be cu102. If neither oneDNN nor CUDA features are enabled, the variant would be native. 

**Dependency Linking**

The library dependencies can be either statically or dynamically linked. This property will need to be manually set by user through either the `--static` or `--dynamic` arguments. There is no foolproof and programmatic way (that I could find) that can easily discern whether the library dependencies are statically or dynamically linked.

### Uploading an Artifact

The user must specify the path to the libmxnet.so, any license files, and any dependencies. The latter two are optional.
 
Example:

`./artifact_repository.py --push --static --libmxnet /path/to/libmxnet.so --licenses path/to/license1.txt /path/to/other_licenses/*.txt --dependencies /path/to/dependencies/*.so`

`./artifact_repository.py --push --dynamic --libmxnet /path/to/libmxnet.so`

NOTE: There is nothing stopping the user from uploading licenses and dependencies for dynamically linked libraries.

### Downloading An Artifact

The user must specify the directory to which the artifact should be downloaded. The user will also need to specify the variant, since different variants can work with the host operating system.

Example:

`./artifact_repository.py --pull --static --variant=cu102 ./dist`

This would result in the following directory structure:

```
dist
  |-----> libmxnet.so
  |-----> libmxnet.meta
  |-----> licenses
             |-----> MKL_LICENSE.txt
             |-----> CUP_LICENSE.txt
             |-----> ...
  |-----> dependencies
             |-----> libxxx.so
             |-----> libyyy.so
             |-----> ...
```

The libmxnet.meta file will include the characteristics of the artifact (ie. library type, variant, git commit id, etc.) in a “property” file format.

