<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# Release Python Docker Images for MXNet

The `docker-python` directory can be used to release mxnet python docker images to dockerhub after any mxnet release.  
It uses the appropriate pip binaries to build different docker images. Both python2 (default) and python3 images are available as -
* {version}_cpu
* {version}_cpu_mkl
* {version}_gpu_cu92
* {version}_gpu_cu92_mkl
* {version}_cpu_py3
* {version}_cpu_mkl_py3
* {version}_gpu_cu92_py3
* {version}_gpu_cu92_mkl_py3

And the following tags will be available without the version string in the image name (for Benchmarking and other use cases):
* latest (same as {version}_cpu)
* gpu (same as {version}_gpu_cu90)
* latest_cpu_mkl_py2 (same as {version}_cpu_mkl)
* latest_cpu_mkl_py3 (same as {version}_cpu_mkl_py3)
* latest_gpu_mkl_py2 (same as {version}_gpu_cu90_mkl)
* latest_gpu_mkl_py3 (same as {version}_gpu_cu90_mkl_py3)

Refer: https://pypi.org/project/mxnet/

### Using the Build Script
`./build_python_dockerfile.sh <mxnet_version> <pip_tag> <path_to_cloned_mxnet_repo>`

For example: 
`./build_python_dockerfile.sh 1.3.0 1.3.0.post0 ~/build-docker/mxnet`

### Tests run
* [test_mxnet.py](https://github.com/apache/mxnet/blob/master/docker/docker-python/test_mxnet.py): This script is used to make sure that the docker image builds the expected mxnet version. That is, the version picked by pip is the same as as the version passed as a parameter.

### Dockerhub Credentials
Dockerhub credentials will be required to push images at the end of this script.
Credentials can be provided in the following ways:
* **Interactive Login:** Run the script as is and it will ask you for credentials interactively.
* **Be Already Logged in:** Login to the mxnet dockerhub account before you run the build script and the script will complete build, test and push.
* **Set Environment Variables:** Set the following environment variables which the script will pick up to login to dockerhub at runtime -
    * $MXNET_DOCKERHUB_PASSWORD
    * $MXNET_DOCKERHUB_USERNAME
    

### Using the Docker Images
* The MXNet Python Docker images can be found here: https://hub.docker.com/r/mxnet/python/

* Docker Pull Command: `docker pull mxnet/python:<image_tag>`
* Get started: `docker run -it mxnet/python:<image_tag> bash`
