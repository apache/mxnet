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

# Installing MXNet

```eval_rst
.. toctree::
   :hidden:

   amazonlinux_setup.md
   build_from_source.md
   c_plus_plus.md
   centos_setup.md
   download.md
   java_setup.md
   osx_setup.md
   raspbian_setup.md
   scala_setup.md
   tx2_setup.md
   ubuntu_setup.md
   validate_mxnet.md
   windows_setup.md
```

Indicate your preferred configuration. Then, follow the customized commands to install MXNet.

<div class="dropdown">
  <button class="btn current-version btn-primary dropdown-toggle" type="button" data-toggle="dropdown">v1.3.1
  <span class="caret"></span></button>
  <ul class="dropdown-menu opt-group">
    <li class="opt active versions"><a href="#">v1.3.1</a></li>
    <li class="opt versions"><a href="#">v1.2.1</a></li>
    <li class="opt versions"><a href="#">v1.1.0</a></li>
    <li class="opt versions"><a href="#">v1.0.0</a></li>
    <li class="opt versions"><a href="#">v0.12.1</a></li>
    <li class="opt versions"><a href="#">v0.11.0</a></li>
    <li class="opt versions"><a href="#">master</a></li>
  </ul>
</div>

<script type="text/javascript" src='../_static/js/options.js'></script>

<!-- START - OS Menu -->

<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default opt active platforms">Linux</button>
  <button type="button" class="btn btn-default opt platforms">MacOS</button>
  <button type="button" class="btn btn-default opt platforms">Windows</button>
  <button type="button" class="btn btn-default opt platforms">Cloud</button>
  <button type="button" class="btn btn-default opt platforms">Devices</button>
</div>

<!-- START - Language Menu -->

<div class="linux macos windows">
<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default opt active languages">Python</button>
  <button type="button" class="btn btn-default opt languages">Scala</button>
  <button type="button" class="btn btn-default opt languages">Java</button>
  <button type="button" class="btn btn-default opt languages">Clojure</button>
  <button type="button" class="btn btn-default opt languages">R</button>
  <button type="button" class="btn btn-default opt languages">Julia</button>
  <button type="button" class="btn btn-default opt languages">Perl</button>
  <button type="button" class="btn btn-default opt languages">Cpp</button>
</div>
</div>

<!-- No CPU GPU for other Devices -->
<div class="linux macos windows cloud">
<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default processors opt active">GPU</button>
  <button type="button" class="btn btn-default processors opt">CPU</button>
</div>
</div>

<!-- other devices -->
<div class="devices">
<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default iots opt active">Raspberry Pi</button>
  <button type="button" class="btn btn-default iots opt">NVIDIA Jetson</button>
</div>
</div>

<!-- Linux Python GPU Options -->

<div class="linux macos windows">
<div class="python">
<div class="cpu gpu">
<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default environs opt active">Pip</button>
  <button type="button" class="btn btn-default environs opt">Docker</button>
  <button type="button" class="btn btn-default environs opt">Build from Source</button>
</div>
</div>
</div>
</div>
<hr>
<!-- END - Main Menu -->

<!-- START - Linux Python CPU Installation Instructions -->

<div class="linux">
<div class="python">
<div class="cpu">
<div class="pip">
<div class="v1-3-1">

```
$ pip install mxnet
```

</div> <!-- End of v1-3-1 -->
<div class="v1-2-1">

```
$ pip install mxnet==1.2.1
```

</div> <!-- End of v1-2-1 -->

<div class="v1-1-0">

```
$ pip install mxnet==1.1.0
```

</div> <!-- End of v1-1-0-->

<div class="v1-0-0">

```
$ pip install mxnet==1.0.0
```

</div> <!-- End of v1-0-0-->


<div class="v0-12-1">

```
$ pip install mxnet==0.12.1
```

For MXNet 0.12.0:

```
$ pip install mxnet==0.12.0
```

</div> <!-- End of v0-12-1-->

<div class="v0-11-0">

```
$ pip install mxnet==0.11.0
```

</div> <!-- End of v0-11-0-->

<div class="master">

```
$ pip install mxnet --pre
```

</div> <!-- End of master-->
<hr> <!-- pip footer -->
MXNet offers MKL pip packages that will be much faster when running on Intel hardware.
Check the chart below for other options, refer to <a href="https://pypi.org/project/mxnet/">PyPI for other MXNet pip packages</a>, or <a href="validate_mxnet.html">validate your MXNet installation</a>.

<img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/install/pip-packages.png" alt="pip packages"/>

**NOTES:**

*mxnet-cu92mkl* means the package is built with CUDA/cuDNN and MKL-DNN enabled and the CUDA version is 9.2.

All MKL pip packages are experimental prior to version 1.3.0.

</div> <!-- End of pip -->


<div class="docker">
<br/>

Docker images with *MXNet* are available at [Docker Hub](https://hub.docker.com/r/mxnet/).

**Step 1**  Install Docker on your machine by following the [docker installation instructions](https://docs.docker.com/engine/installation/linux/ubuntu/#install-using-the-repository).

*Note* - You can install Community Edition (CE) to get started with *MXNet*.

**Step 2** [Optional] Post installation steps to manage Docker as a non-root user.

Follow the four steps in this [docker documentation](https://docs.docker.com/engine/installation/linux/linux-postinstall/#manage-docker-as-a-non-root-user) to allow managing docker containers without *sudo*.

If you skip this step, you need to use *sudo* each time you invoke Docker.

**Step 3** Pull the MXNet docker image.

```
$ docker pull mxnet/python # Use sudo if you skip Step 2
```

You can list docker images to see if mxnet/python docker image pull was successful.

```
$ docker images # Use sudo if you skip Step 2

REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
mxnet/python        latest              00d026968b3c        3 weeks ago         1.41 GB
```

Using the latest MXNet with [Intel MKL-DNN](https://github.com/intel/mkl-dnn) is recommended for the fastest inference speeds with MXNet.

```
$ docker pull mxnet/python:1.3.0_cpu_mkl # Use sudo if you skip Step 2
$ docker images # Use sudo if you skip Step 2

REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
mxnet/python        1.3.0_cpu_mkl       deaf9bf61d29        4 days ago          678 MB
```

**Step 4** <a href="validate_mxnet.html">Validate the installation</a>.

</div> <!-- END of docker -->

<div class="build-from-source">
<br/>

To build from source, refer to the <a href="ubuntu_setup.html">MXNet Ubuntu installation guide</a>.

</div><!-- END of build from source -->

</div><!-- END of CPU -->
<!-- END - Linux Python CPU Installation Instructions -->

<!-- START - Linux Python GPU Installation Instructions -->

<div class="gpu">
<div class="pip">
<div class="v1-3-1">

```
$ pip install mxnet-cu92
```

</div> <!-- End of v1-3-1-->
<div class="v1-2-1">

```
$ pip install mxnet-cu92==1.2.1
```

</div> <!-- End of v1-2-1-->

<div class="v1-1-0">

```
$ pip install mxnet-cu91==1.1.0
```

</div> <!-- End of v1-1-0-->

<div class="v1-0-0">

```
$ pip install mxnet-cu90==1.0.0
```

</div> <!-- End of v1-0-0-->

<div class="v0-12-1">

```
$ pip install mxnet-cu90==0.12.1
```

</div> <!-- End of v0-12-1-->

<div class="v0-11-0">

```
$ pip install mxnet-cu80==0.11.0
```

</div> <!-- End of v0-11-0-->

<div class="master">

```
$ pip install mxnet-cu92 --pre
```

</div> <!-- End of master-->
<hr> <!-- pip footer -->
MXNet offers MKL pip packages that will be much faster when running on Intel hardware.
Check the chart below for other options, refer to <a href="https://pypi.org/project/mxnet/">PyPI for other MXNet pip packages</a>, or <a href="validate_mxnet.html">validate your MXNet installation</a>.

<img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/install/pip-packages.png" alt="pip packages"/>

**NOTES:**

*mxnet-cu92mkl* means the package is built with CUDA/cuDNN and MKL-DNN enabled and the CUDA version is 9.2.

All MKL pip packages are experimental prior to version 1.3.0.

CUDA should be installed first. Instructions can be found in the <a href="ubuntu_setup.html#cuda-dependencies">CUDA dependencies section of the MXNet Ubuntu installation guide</a>.

**Important:** Make sure your installed CUDA version matches the CUDA version in the pip package. Check your CUDA version with the following command:

```
nvcc --version
```

You can either upgrade your CUDA install or install the MXNet package that supports your CUDA version.

</div> <!-- END of pip -->

<div class="docker">

<br/>

Docker images with *MXNet* are available at [Docker Hub](https://hub.docker.com/r/mxnet/).

**Step 1**  Install Docker on your machine by following the [docker installation instructions](https://docs.docker.com/engine/installation/linux/ubuntu/#install-using-the-repository).

*Note* - You can install Community Edition (CE) to get started with *MXNet*.

**Step 2** [Optional] Post installation steps to manage Docker as a non-root user.

Follow the four steps in this [docker documentation](https://docs.docker.com/engine/installation/linux/linux-postinstall/#manage-docker-as-a-non-root-user) to allow managing docker containers without *sudo*.

If you skip this step, you need to use *sudo* each time you invoke Docker.

**Step 3** Install *nvidia-docker-plugin* following the [installation instructions](https://github.com/NVIDIA/nvidia-docker/wiki). *nvidia-docker-plugin* is required to enable the usage of GPUs from the docker containers.

**Step 4** Pull the MXNet docker image.

```
$ docker pull mxnet/python:gpu # Use sudo if you skip Step 2
```

You can list docker images to see if mxnet/python docker image pull was successful.

```
$ docker images # Use sudo if you skip Step 2

REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
mxnet/python        gpu                 493b2683c269        3 weeks ago         4.77 GB
```

Using the latest MXNet with [Intel MKL-DNN](https://github.com/intel/mkl-dnn) is recommended for the fastest inference speeds with MXNet.

```
$ docker pull mxnet/python:1.3.0_cpu_mkl # Use sudo if you skip Step 2
$ docker images # Use sudo if you skip Step 2

REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
mxnet/python        1.3.0_gpu_cu92_mkl  adcb3ab19f50        4 days ago          4.23 GB
```

**Step 5** <a href="validate_mxnet.html">Validate the installation</a>.

</div> <!-- END of docker -->

<div class="build-from-source">
<br/>
Refer to the <a href="ubuntu_setup.html">MXNet Ubuntu installation guide</a>.


</div> <!-- END of build from source -->
</div> <!-- END of GPU -->
</div> <!-- END of Python -->
<!-- END - Linux Python GPU Installation Instructions -->


<div class="r">
<div class="cpu">

The default version of R that is installed with `apt-get` is insufficient. You will need to first [install R v3.4.4+ and build MXNet from source](ubuntu_setup.html#install-the-mxnet-package-for-r).

After you have setup R v3.4.4+ and MXNet, you can build and install the MXNet R bindings with the following, assuming that `incubator-mxnet` is the source directory you used to build MXNet as follows:

```
$ cd incubator-mxnet
$ make rpkg
```

</div> <!-- END of CPU -->


<div class="gpu">

The default version of R that is installed with `apt-get` is insufficient. You will need to first [install R v3.4.4+ and build MXNet from source](ubuntu_setup.html#install-the-mxnet-package-for-r).

After you have setup R v3.4.4+ and MXNet, you can build and install the MXNet R bindings with the following, assuming that `incubator-mxnet` is the source directory you used to build MXNet as follows:

```
$ cd incubator-mxnet
$ make rpkg
```

</div> <!-- END of GPU -->
</div> <!-- END of R -->


<div class="scala">
<div class="gpu">
<br/>
You can use the Maven packages defined in the following dependency to include MXNet in your Scala project. Please refer to the <a href="scala_setup.html">MXNet-Scala setup guide</a> for a detailed set of instructions to help you with the setup process.

<a href="https://mvnrepository.com/artifact/org.apache.mxnet/mxnet-full_2.11-linux-x86_64-gpu"><img src="https://img.shields.io/badge/org.apache.mxnet-linux gpu-green.svg" alt="maven badge"/></a>

```html
<dependency>
    <groupId>org.apache.mxnet</groupId>
    <artifactId>mxnet-full_2.11-linux-x86_64-gpu</artifactId>
</dependency>
```
<br>
</div> <!-- End of gpu -->

<div class="cpu">
<br/>
You can use the Maven packages defined in the following dependency to include MXNet in your Scala project. Please refer to the <a href="scala_setup.html">MXNet-Scala setup guide</a> for a detailed set of instructions to help you with the setup process.

<a href="https://mvnrepository.com/artifact/org.apache.mxnet/mxnet-full_2.11-linux-x86_64-cpu"><img src="https://img.shields.io/badge/org.apache.mxnet-linux cpu-green.svg" alt="maven badge"/></a>

```html
<dependency>
    <groupId>org.apache.mxnet</groupId>
    <artifactId>mxnet-full_2.11-linux-x86_64-cpu</artifactId>
</dependency>
```
<br>
</div> <!-- End of cpu -->
</div> <!-- End of scala -->


<div class="clojure">
<div class="gpu">
<br/>

You can use the Maven packages defined in the following dependency to include MXNet in your Clojure project. To maximize leverage, the Clojure package has been built on the existing Scala package. Please refer to the <a href="scala_setup.html">MXNet-Scala setup guide</a> for a detailed set of instructions to help you with the setup process that is required to use the Clojure dependency.

<a href="https://mvnrepository.com/artifact/org.apache.mxnet.contrib.clojure/clojure-mxnet-linux-gpu"><img src="https://img.shields.io/badge/org.apache.mxnet-linux gpu-green.svg" alt="maven badge"/></a>

```html
<dependency>
    <groupId>org.apache.mxnet.contrib.clojure</groupId>
    <artifactId>clojure-mxnet-linux-gpu</artifactId>
</dependency>
```

<br>
</div> <!-- End of gpu -->
<div class="cpu">
<br/>
You can use the Maven packages defined in the following dependency to include MXNet in your Clojure project. To maximize leverage, the Clojure package has been built on the existing Scala package. Please refer to the <a href="scala_setup.html">MXNet-Scala setup guide</a> for a detailed set of instructions to help you with the setup process that is required to use the Clojure dependency.

<a href="https://mvnrepository.com/artifact/org.apache.mxnet.contrib.clojure/clojure-mxnet-linux-cpu"><img src="https://img.shields.io/badge/org.apache.mxnet-linux cpu-green.svg" alt="maven badge"/></a>

```html
<dependency>
    <groupId>org.apache.mxnet.contrib.clojure</groupId>
    <artifactId>clojure-mxnet-linux-cpu</artifactId>
</dependency>
```

<br>
</div> <!-- End of cpu -->
</div> <!-- End of clojure -->


<div class="java">
<div class="gpu">
<br/>
You can use the Maven packages defined in the following dependency to include MXNet in your Java project. The Java API is provided as a subset of the Scala API and is intended for inference only. Please refer to the <a href="java_setup.html">MXNet-Java setup guide</a> for a detailed set of instructions to help you with the setup process.

<a href="https://repository.apache.org/#nexus-search;gav~org.apache.mxnet~~1.4.0-SNAPSHOT~~"><img src="https://img.shields.io/badge/org.apache.mxnet-linux gpu-green.svg" alt="maven badge"/></a>

```html
<dependency>
    <groupId>org.apache.mxnet</groupId>
    <artifactId>mxnet-full_2.11-linux-x86_64-gpu</artifactId>
    <version>[1.4.0, )</version>
</dependency>
```

<br>
</div> <!-- End of gpu -->

<div class="cpu">
<br/>
You can use the Maven packages defined in the following dependency to include MXNet in your Java project. The Java API is provided as a subset of the Scala API and is intended for inference only. Please refer to the <a href="java_setup.html">MXNet-Java setup guide</a> for a detailed set of instructions to help you with the setup process.

<a href="https://repository.apache.org/#nexus-search;gav~org.apache.mxnet~~1.4.0-SNAPSHOT~~"><img src="https://img.shields.io/badge/org.apache.mxnet-linux cpu-green.svg" alt="maven badge"/></a>

```html
<dependency>
    <groupId>org.apache.mxnet</groupId>
    <artifactId>mxnet-full_2.11-linux-x86_64-cpu</artifactId>
    <version>[1.4.0, )</version>
</dependency>
```
<br>
</div> <!-- End of cpu -->
</div> <!-- End of java -->


<div class="julia">
<div class="cpu gpu">
</br>
Refer to the <a href="ubuntu_setup.html#install-the-mxnet-package-for-julia">Julia section of the MXNet Ubuntu installation guide</a>.

</div> <!-- End of cpu gpu -->
</div> <!-- End of julia -->

<div class="perl">
<div class="cpu gpu">
</br>
Refer to the <a href="ubuntu_setup.html#install-the-mxnet-package-for-perl">Perl section of the MXNet Ubuntu installation guide</a>.

</div> <!-- End of cpu gpu -->
</div> <!-- End of julia -->



<div class="cpp">
<div class="cpu gpu">
<br/>
<p>To enable the C++ package, build from source using `make USE_CPP_PACKAGE=1`.
<br/>Refer to the <a href="c_plus_plus.html">MXNet C++ setup guide</a> for more info.</p>
<br/>
</div> <!-- End of cpu gpu -->
</div> <!-- END - C++-->
<hr>

</div> <!-- END - Linux -->


<!-- START - MacOS Python CPU Installation Instructions -->

<div class="macos">
<div class="python">
<div class="cpu">
<div class="pip">
<div class="v1-3-1">

```
$ pip install mxnet
```

</div> <!-- End of v1-3-1 -->
<div class="v1-2-1">

```
$ pip install mxnet==1.2.1
```

</div> <!-- End of v1-2-1 -->


<div class="v1-1-0">

```
$ pip install mxnet==1.1.0
```

</div> <!-- End of v1-1-0-->


<div class="v1-0-0">

```
$ pip install mxnet==1.0.0
```

</div> <!-- End of v1-0-0-->

<div class="v0-12-1">

```
$ pip install mxnet=0.12.1
```

</div> <!-- End of v0-12-1-->


<div class="v0-11-0">

```
$ pip install mxnet==0.11.0
```

</div> <!-- End of v0-11-0-->

<div class="master">

```
$ pip install mxnet --pre
```

</div> <!-- End of master-->
<hr> <!-- pip footer -->
MXNet offers MKL pip packages that will be much faster when running on Intel hardware.
Check the chart below for other options, refer to <a href="https://pypi.org/project/mxnet/">PyPI for other MXNet pip packages</a>, or <a href="validate_mxnet.html">validate your MXNet installation</a>.

<img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/install/pip-packages.png" alt="pip packages"/>

**NOTES:**

*mxnet-cu92mkl* means the package is built with CUDA/cuDNN and MKL-DNN enabled and the CUDA version is 9.2.

All MKL pip packages are experimental prior to version 1.3.0.

</div> <!-- END of pip -->


<div class="docker">
<br/>

Docker images with *MXNet* are available at [Docker Hub](https://hub.docker.com/r/mxnet/).

**Step 1**  Install Docker on your machine by following the [docker installation instructions](https://docs.docker.com/docker-for-mac/install/#install-and-run-docker-for-mac).

*Note* - You can install Community Edition (CE) to get started with *MXNet*.

**Step 2** Pull the MXNet docker image.

```
$ docker pull mxnet/python
```

You can list docker images to see if mxnet/python docker image pull was successful.

```
$ docker images

REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
mxnet/python        latest              00d026968b3c        3 weeks ago         1.41 GB
```

Using the latest MXNet with [Intel MKL-DNN](https://github.com/intel/mkl-dnn) is recommended for the fastest inference speeds with MXNet.

```
$ docker pull mxnet/python:1.3.0_cpu_mkl # Use sudo if you skip Step 2
$ docker images # Use sudo if you skip Step 2

REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
mxnet/python        1.3.0_cpu_mkl       deaf9bf61d29        4 days ago          678 MB
```

**Step 4** <a href="validate_mxnet.html">Validate the installation</a>.

</div> <!-- END of docker -->


<div class="build-from-source">
<br/>

To build from source, refer to the <a href="osx_setup.html">MXNet macOS installation guide</a>.

MXNet developers should refer to the MXNet wiki's <a href="https://cwiki.apache.org/confluence/display/MXNET/MXNet+Developer+Setup+on+Mac">Developer Setup on Mac</a>.
<br/>
</div> <!-- END of build from source -->
</div> <!-- END of CPU -->


<!-- START - Mac OS Python GPU Installation Instructions -->
<div class="gpu">
<div class="pip docker">
<br/>
This option is only available by building from source. Refer to the <a href="osx_setup.html">MXNet macOS installation guide</a>.
<br/>
</div>

<div class="build-from-source">
<br/>

Refer to the <a href="osx_setup.html">MXNet macOS installation guide</a>.

MXNet developers should refer to the MXNet wiki's <a href="https://cwiki.apache.org/confluence/display/MXNET/MXNet+Developer+Setup+on+Mac">Developer Setup on Mac</a>.
<br/>
</div> <!-- END of build from source -->
</div> <!-- END of GPU -->
</div> <!-- END of Python -->


<!-- START - MacOS R CPU Installation Instructions -->

<div class="r">
<div class="cpu">
</br>

To run MXNet you also should have OpenCV and OpenBLAS installed. You may install them with `brew` as follows:

```bash
brew install opencv
brew install openblas
```

To ensure MXNet R package runs with the version of OpenBLAS installed, create a symbolic link as follows:

```bash
ln -sf /usr/local/opt/openblas/lib/libopenblas.dylib /usr/local/opt/openblas/lib/libopenblasp-r0.3.1.dylib
```

Install the latest version (3.5.1+) of R from [CRAN](https://cran.r-project.org/bin/macosx/).
You can [build MXNet-R from source](osx_setup.html#install-the-mxnet-package-for-r), or you can use a pre-built binary:

```r
cran <- getOption("repos")
cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/"
options(repos = cran)
install.packages("mxnet")
```

</div> <!-- END of CPU -->


<div class="gpu">
</br>
Will be available soon.

</div> <!-- END of GPU -->
</div> <!-- END of R -->

<div class="scala">
<div class="cpu">
</br>
You can use the Maven packages defined in the following dependency to include MXNet in your Scala project. Please refer to the <a href="scala_setup.html">MXNet-Scala setup guide</a> for a detailed set of instructions to help you with the setup process.

<a href="https://mvnrepository.com/artifact/org.apache.mxnet/mxnet-full_2.11-osx-x86_64-cpu"><img src="https://img.shields.io/badge/org.apache.mxnet-mac cpu-green.svg" alt="maven badge"/></a>

```html
<dependency>
    <groupId>org.apache.mxnet</groupId>
    <artifactId>mxnet-full_2.11-osx-x86_64-cpu</artifactId>
</dependency>
```
<br>
</div> <!-- End of cpu  -->
<div class="gpu">

Not available at this time. <br>

</div>
</div> <!-- End of scala -->


<div class="clojure">
<div class="cpu">
</br>
You can use the Maven packages defined in the following dependency to include MXNet in your Clojure project. To maximize leverage, the Clojure package has been built on the existing Scala package. Please refer to the <a href="scala_setup.html">MXNet-Scala setup guide</a> for a detailed set of instructions to help you with the setup process that is required to use the Clojure dependency.

<a href="https://mvnrepository.com/artifact/org.apache.mxnet.contrib.clojure/clojure-mxnet-osx-cpu"><img src="https://img.shields.io/badge/org.apache.mxnet-mac cpu-green.svg" alt="maven badge"/></a>

```html
<dependency>
    <groupId>org.apache.mxnet.contrib.clojure</groupId>
    <artifactId>clojure-mxnet-osx-cpu</artifactId>
</dependency>
```

<br>
</div> <!-- End of cpu  -->
<div class="gpu">
Not available at this time. <br>
</div> <!-- End of gpu -->
</div> <!-- End of clojure -->


<div class="java">
<div class="cpu">
</br>
You can use the Maven packages defined in the following dependency to include MXNet in your Java project. The Java API is provided as a subset of the Scala API and is intended for inference only. Please refer to the <a href="java_setup.html">MXNet-Java setup guide</a> for a detailed set of instructions to help you with the setup process.

<a href="https://repository.apache.org/#nexus-search;gav~org.apache.mxnet~~1.4.0-SNAPSHOT~~"><img src="https://img.shields.io/badge/org.apache.mxnet-mac cpu-green.svg" alt="maven badge"/></a>

```html
<dependency>
    <groupId>org.apache.mxnet</groupId>
    <artifactId>mxnet-full_2.11-linux-x86_64-cpu</artifactId>
    <version>[1.4.0, )</version>
</dependency>
```
<br>
</div> <!-- End of cpu  -->
<div class="gpu">

Not available at this time. <br>

</div>
</div> <!-- End of java -->


<div class="julia">
<div class="cpu gpu">
</br>
Refer to the <a href="osx_setup.html#install-the-mxnet-package-for-julia">Julia section of the MXNet macOS installation guide</a>.

</div> <!-- End of cpu gpu -->
</div> <!-- End of julia -->

<div class="perl">
<div class="cpu gpu">
</br>
Refer to the <a href="osx_setup.html#install-the-mxnet-package-for-perl">Perl section of the MXNet macOS installation guide</a>.

</div> <!-- End of cpu gpu -->
</div> <!-- End of julia -->



<div class="cpp">
<br/>
<p>To enable the C++ package, build from source using `make USE_CPP_PACKAGE=1`.
<br/>Refer to the <a href="c_plus_plus.html">MXNet C++ setup guide</a> for more info.</p>
<br/>
</div>
<hr>
For more installation options, refer to the <a href="osx_setup.html">MXNet macOS installation guide</a>.
</div> <!-- END - Mac OS -->



<div class="windows">
<div class="python">
<div class="cpu">
<div class="pip">
<div class="v1-3-1">

```
$ pip install mxnet
```

</div> <!-- End of v1-3-1 -->
<div class="v1-2-1">

```
$ pip install mxnet==1.2.1
```

</div> <!-- End of v1-2-1 -->

<div class="v1-1-0">

```
$ pip install mxnet==1.1.0
```

</div> <!-- End of v1-1-0-->

<div class="v1-0-0">

```
$ pip install mxnet==1.0.0
```

</div> <!-- End of v1-0-0-->

<div class="v0-12-1">

```
$ pip install mxnet==0.12.1
```

</div> <!-- End of v0-12-1-->

<div class="v0-11-0">

```
$ pip install mxnet==0.11.0
```

</div> <!-- End of v0-11-0-->

<div class="master">

```
$ pip install mxnet --pre
```

</div> <!-- End of master-->
<hr> <!-- pip footer -->
MXNet offers MKL pip packages that will be much faster when running on Intel hardware.
Check the chart below for other options, refer to <a href="https://pypi.org/project/mxnet/">PyPI for other MXNet pip packages</a>, or <a href="validate_mxnet.html">validate your MXNet installation</a>.

<img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/install/pip-packages.png" alt="pip packages"/>

**NOTES:**

*mxnet-cu92mkl* means the package is built with CUDA/cuDNN and MKL-DNN enabled and the CUDA version is 9.2.

All MKL pip packages are experimental prior to version 1.3.0.

</div> <!-- End of pip -->


<div class="docker">
<br/>

Docker images with *MXNet* are available at [Docker Hub](https://hub.docker.com/r/mxnet/).

**Step 1**  Install Docker on your machine by following the [docker installation instructions](https://docs.docker.com/engine/installation/linux/ubuntu/#install-using-the-repository).

*Note* - You can install Community Edition (CE) to get started with *MXNet*.

**Step 2** [Optional] Post installation steps to manage Docker as a non-root user.

Follow the four steps in this [docker documentation](https://docs.docker.com/engine/installation/linux/linux-postinstall/#manage-docker-as-a-non-root-user) to allow managing docker containers without *sudo*.

If you skip this step, you need to use *sudo* each time you invoke Docker.

**Step 3** Pull the MXNet docker image.

```
$ docker pull mxnet/python # Use sudo if you skip Step 2
```

You can list docker images to see if mxnet/python docker image pull was successful.

```
$ docker images # Use sudo if you skip Step 2

REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
mxnet/python        latest              00d026968b3c        3 weeks ago         1.41 GB
```

Using the latest MXNet with [Intel MKL-DNN](https://github.com/intel/mkl-dnn) is recommended for the fastest inference speeds with MXNet.

```
$ docker pull mxnet/python:1.3.0_cpu_mkl # Use sudo if you skip Step 2
$ docker images # Use sudo if you skip Step 2

REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
mxnet/python        1.3.0_cpu_mkl       deaf9bf61d29        4 days ago          678 MB
```

**Step 4** <a href="validate_mxnet.html">Validate the installation</a>.


</div> <!-- End of docker -->

<div class="build-from-source">
<br/>
Refer to the <a href="windows_setup.html">MXNet Windows installation guide</a>

</div> <!-- End of Build from source -->
</div> <!-- End of CPU -->


<div class="gpu">
<div class="pip">
<div class="v1-3-1">

```
$ pip install mxnet-cu92
```

</div> <!-- End of v1-3-1 -->
<div class="v1-2-1">

```
$ pip install mxnet-cu92==1.2.1
```

</div> <!-- End of v1-2-1 -->

<div class="v1-1-0">

```
$ pip install mxnet-cu91==1.1.0
```

</div> <!-- End of v1-1-0-->

<div class="v1-0-0">

```
$ pip install mxnet-cu90==1.0.0
```

</div> <!-- End of v1-0-0-->

<div class="v0-12-1">

```
$ pip install mxnet-cu90==0.12.1
```

</div> <!-- End of v0-12-1-->

<div class="v0-11-0">

```
$ pip install mxnet-cu80==0.11.0
```

</div> <!-- End of v0-11-0-->

<div class="master">

```
$ pip install mxnet-cu92 --pre
```

</div> <!-- End of master-->
<hr> <!-- pip footer -->
MXNet offers MKL pip packages that will be much faster when running on Intel hardware.
Check the chart below for other options, refer to <a href="https://pypi.org/project/mxnet/">PyPI for other MXNet pip packages</a>, or <a href="validate_mxnet.html">validate your MXNet installation</a>.

<img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/install/pip-packages.png" alt="pip packages"/>

**NOTES:**

*mxnet-cu92mkl* means the package is built with CUDA/cuDNN and MKL-DNN enabled and the CUDA version is 9.2.

All MKL pip packages are experimental prior to version 1.3.0.

[Anaconda](https://www.anaconda.com/download/) is recommended.

CUDA should be installed first. Instructions can be found in the <a href="ubuntu_setup.html#cuda-dependencies">CUDA dependencies section of the MXNet Ubuntu installation guide</a>.

**Important:** Make sure your installed CUDA version matches the CUDA version in the pip package. Check your CUDA version with the following command:

```
nvcc --version
```

Refer to [#8671](https://github.com/apache/incubator-mxnet/issues/8671) for status on CUDA 9.1 support.

You can either upgrade your CUDA install or install the MXNet package that supports your CUDA version.

</div> <!-- End of pip -->

<div class="build-from-source">
<br/>

To build from source, refer to the <a href="windows_setup.html">MXNet Windows installation guide</a>.


</div> <!-- End of build from source -->
</div> <!-- End of GPU -->
</div> <!-- End of Python -->


<!-- START - Windows R CPU Installation Instructions -->

<div class="r">
<div class="cpu">
</br>

Install the latest version (3.5.1+) of R from [CRAN](https://cran.r-project.org/bin/windows/).
You can [build MXNet-R from source](windows_setup.html#install-mxnet-package-for-r), or you can use a pre-built binary:

```r
cran <- getOption("repos")
cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/"
options(repos = cran)
install.packages("mxnet")
```

To run MXNet you also should have OpenCV and OpenBLAS installed.

</div> <!-- END - Windows R CPU -->

<div class="gpu">
</br>

You can [build MXNet-R from source](windows_setup.html#install-mxnet-package-for-r), or you can use a pre-built binary:

```r
  cran <- getOption("repos")
  cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/GPU/cu92"
  options(repos = cran)
  install.packages("mxnet")
```
Change cu92 to cu80, cu90 or cu91 based on your CUDA toolkit version. Currently, MXNet supports these versions of CUDA.

</div> <!-- END of GPU -->
</div> <!-- END - Windows R -->

<div class="scala">
<div class="cpu gpu">
<br/>
MXNet-Scala for Windows is not yet available.
<br/>
</div> <!-- End of cpu gpu -->
</div> <!-- End of scala -->

<div class="clojure">
<div class="cpu gpu">
<br/>
MXNet-Clojure for Windows is not yet available.
<br/>
</div> <!-- End of cpu gpu -->
</div> <!-- End of clojure -->

<div class="java">
<div class="cpu gpu">
<br/>
MXNet-Java for Windows is not yet available.
<br/>
</div> <!-- End of cpu gpu -->
</div> <!-- End of java -->

<div class="julia">
<div class="cpu gpu">
</br>
Refer to the <a href="windows_setup.html#install-the-mxnet-package-for-julia">Julia section of the MXNet Windows installation guide</a>.

</div> <!-- End of cpu gpu -->
</div> <!-- End of julia -->

<div class="perl">
<div class="cpu gpu">
</br>
Refer to the <a href="windows_setup.html#install-the-mxnet-package-for-perl">Perl section of the MXNet Windows installation guide</a>.

</div> <!-- End of cpu gpu -->
</div> <!-- End of julia -->

<div class="cpp">
<div class="cpu gpu">
</br>
<p>To enable the C++ package, build from source using `make USE_CPP_PACKAGE=1`.
<br/>Refer to the <a href="c_plus_plus.html">MXNet C++ setup guide</a> for more info.</p>
<br/>
</div> <!-- End of cpu gpu -->
</div> <!-- End of C++ -->
<hr>
For more installation options, refer to the <a href="windows_setup.html">MXNet Windows installation guide</a>.
</div> <!-- End of Windows -->


<!-- START - Cloud Python Installation Instructions -->

<div class="cloud">
<div class="gpu">

MXNet is available on several cloud providers with GPU support. You can also find GPU/CPU-hybrid support for use cases like scalable inference, or even fractional GPU support with AWS Elastic Inference.

* **Alibaba**
    - [NVIDIA VM](https://docs.nvidia.com/ngc/ngc-alibaba-setup-guide/launching-nv-cloud-vm-console.html#launching-nv-cloud-vm-console)
* **Amazon Web Services**
    - [Amazon SageMaker](https://aws.amazon.com/sagemaker/) - Managed training and deployment of MXNet models
    - [AWS Deep Learning AMI](https://aws.amazon.com/machine-learning/amis/) - Preinstalled Conda environments for Python 2 or 3 with MXNet, CUDA, cuDNN, MKL-DNN, and AWS Elastic Inference
    - [Dynamic Training on AWS](https://github.com/awslabs/dynamic-training-with-apache-mxnet-on-aws) - experimental manual EC2 setup or semi-automated CloudFormation setup
    - [NVIDIA VM](https://aws.amazon.com/marketplace/pp/B076K31M1S)
* **Google Cloud Platform**
    - [NVIDIA VM](https://console.cloud.google.com/marketplace/details/nvidia-ngc-public/nvidia_gpu_cloud_image)
* **Microsoft Azure**
    - [NVIDIA VM](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/nvidia.ngc_azure_17_11?tab=Overview)
* **Oracle Cloud**
    - [NVIDIA VM](https://docs.cloud.oracle.com/iaas/Content/Compute/References/ngcimage.htm)

All NVIDIA VMs use the [NVIDIA MXNet Docker container](https://ngc.nvidia.com/catalog/containers/nvidia:mxnet).
Follow the [container usage instructions](https://ngc.nvidia.com/catalog/containers/nvidia:mxnet) found in [NVIDIA's container repository](https://ngc.nvidia.com/).

</div> <!-- END gpu -->

<div class="cpu">
MXNet should work on any cloud provider's CPU-only instances. Follow the Python pip install instructions, Docker instructions, or try the following preinstalled option.

* **Amazon Web Services**
    - [AWS Deep Learning AMI](https://aws.amazon.com/machine-learning/amis/) - Preinstalled Conda environments for Python 2 or 3 with MXNet and MKL-DNN.

</div> <!-- end cpu -->
</div> <!-- END - Cloud Python Installation Instructions -->


<!-- DEVICES -->
<div class="devices">
  <div class="raspberry-pi">

MXNet supports the Debian based Raspbian ARM based operating system so you can run MXNet on Raspberry Pi 3B devices.

These instructions will walk through how to build MXNet for the Raspberry Pi and install the Python bindings for the library.

You can do a dockerized cross compilation build on your local machine or a native build on-device.

The complete MXNet library and its requirements can take almost 200MB of RAM, and loading large models with the library can take over 1GB of RAM. Because of this, we recommend running MXNet on the Raspberry Pi 3 or an equivalent device that has more than 1 GB of RAM and a Secure Digital (SD) card that has at least 4 GB of free memory.

## Quick installation
You can use this [pre-built Python wheel](wget https://mxnet-public.s3.amazonaws.com/install/raspbian/mxnet-1.5.0-py2.py3-none-any.whl) on a Raspberry Pi 3B with Stretch. You will likely need to install several dependencies to get MXNet to work. Refer to the following **Build** section for details.

**Cross compilation build (Experimental)**

## Docker installation
**Step 1**  Install Docker on your machine by following the [docker installation instructions](https://docs.docker.com/engine/installation/linux/ubuntu/#install-using-the-repository).

*Note* - You can install Community Edition (CE)

**Step 2** [Optional] Post installation steps to manage Docker as a non-root user.

Follow the four steps in this [docker documentation](https://docs.docker.com/engine/installation/linux/linux-postinstall/#manage-docker-as-a-non-root-user) to allow managing docker containers without *sudo*.

## Build

**Please use a Native build with gcc 4 as explained below, higher compiler versions currently cause test
failures on ARM**

The following command will build a container with dependencies and tools and then compile MXNet for
ARMv7. The resulting artifact will be located in `build/mxnet-x.x.x-py2.py3-none-any.whl`, copy this
file to your Raspberry Pi.

```
ci/build.py -p armv7
```

## Install

Your Pi will need several dependencies.

Install MXNet dependencies with the following:
```
sudo apt-get update
sudo apt-get install -y \
    apt-transport-https \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libatlas-base-dev \
    libcurl4-openssl-dev \
    libjemalloc-dev \
    liblapack-dev \
    libopenblas-dev \
    libopencv-dev \
    libzmq3-dev \
    ninja-build \
    python-dev \
    software-properties-common \
    sudo \
    unzip \
    virtualenv \
    wget
```
Install virtualenv with:
```
sudo pip install virtualenv
```
Create a Python 2.7 environment for MXNet with:
```
virtualenv -p `which python` mxnet_py27
```
You may use Python 3, however the [wine bottle detection example](https://mxnet.incubator.apache.org/versions/master/tutorials/embedded/wine_detector.html) for the Pi with camera requires Python 2.7.

Create a virtualenv and install the wheel we created previously, or the wheel that you downloaded.

```
virtualenv -p `which python3` mxnet_py27
source mxnet_py27/bin/activate
pip install mxnet-x.x.x-py2.py3-none-any.whl
```


**Native Build**

Installing MXNet is a two-step process:

1. Build the shared library from the MXNet C++ source code.
2. Install the supported language-specific packages for MXNet.

**Step 1** Build the Shared Library

On Raspbian versions Wheezy and later, you need the following dependencies:

- Git (to pull code from GitHub)

- libblas (for linear algebraic operations)

- libopencv (for computer vision operations. This is optional if you want to save RAM and Disk Space)

- A C++ compiler that supports C++ 11. The C++ compiler compiles and builds MXNet source code. Supported compilers include the following:

    - [G++ (4.8 or later)](https://gcc.gnu.org/gcc-4.8/). Make sure to use gcc 4 and not 5 or 6 as there are known bugs with these compilers.
    - [Clang (3.9 - 6)](https://clang.llvm.org/)

Install these dependencies using the following commands in any directory:

```
    sudo apt-get update
    sudo apt-get -y install git cmake ninja-build build-essential g++-4.9 c++-4.9 liblapack* libblas* libopencv* libopenblas* python3-dev python-dev virtualenv
```

Clone the MXNet source code repository using the following `git` command in your home directory:
```
    git clone https://github.com/apache/incubator-mxnet.git --recursive
    cd incubator-mxnet
```

Build:
```
    mkdir -p build && cd build
    cmake \
        -DUSE_SSE=OFF \
        -DUSE_CUDA=OFF \
        -DUSE_OPENCV=ON \
        -DUSE_OPENMP=ON \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_SIGNAL_HANDLER=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -GNinja ..
    ninja -j$(nproc)
```
Some compilation units require memory close to 1GB, so it's recommended that you enable swap as
explained below and be cautious about increasing the number of jobs when building (-j)

Executing these commands start the build process, which can take up to a couple hours, and creates a file called `libmxnet.so` in the build directory.

If you are getting build errors in which the compiler is being killed, it is likely that the
compiler is running out of memory (especially if you are on Raspberry Pi 1, 2 or Zero, which have
less than 1GB of RAM), this can often be rectified by increasing the swapfile size on the Pi by
editing the file /etc/dphys-swapfile and changing the line CONF_SWAPSIZE=100 to CONF_SWAPSIZE=1024,
then running:
```
  sudo /etc/init.d/dphys-swapfile stop
  sudo /etc/init.d/dphys-swapfile start
  free -m # to verify the swapfile size has been increased
```

**Step 2** Install MXNet Python Bindings

To install Python bindings run the following commands in the MXNet directory:

```
    cd python
    pip install --upgrade pip
    pip install -e .
```

Note that the `-e` flag is optional. It is equivalent to `--editable` and means that if you edit the source files, these changes will be reflected in the package installed.

Alternatively you can create a whl package installable with pip with the following command:
```
ci/docker/runtime_functions.sh build_wheel python/ $(realpath build)
```


You are now ready to run MXNet on your Raspberry Pi device. You can get started by following the tutorial on [Real-time Object Detection with MXNet On The Raspberry Pi](http://mxnet.io/tutorials/embedded/wine_detector.html).

*Note - Because the complete MXNet library takes up a significant amount of the Raspberry Pi's limited RAM, when loading training data or large models into memory, you might have to turn off the GUI and terminate running processes to free RAM.*

</div> <!-- End of raspberry pi -->


<div class="nvidia-jetson">

# Nvidia Jetson TX family

MXNet supports the Ubuntu Arch64 based operating system so you can run MXNet on NVIDIA Jetson Devices.

These instructions will walk through how to build MXNet for the Pascal based [NVIDIA Jetson TX2](http://www.nvidia.com/object/embedded-systems-dev-kits-modules.html) and install the corresponding python language bindings.

For the purposes of this install guide we will assume that CUDA is already installed on your Jetson device.

**Install MXNet**

Installing MXNet is a two-step process:

1. Build the shared library from the MXNet C++ source code.
2. Install the supported language-specific packages for MXNet.

**Step 1** Build the Shared Library

You need the following additional dependencies:

- Git (to pull code from GitHub)

- libatlas (for linear algebraic operations)

- libopencv (for computer vision operations)

- python pip (to load relevant python packages for our language bindings)

Install these dependencies using the following commands in any directory:

```
    sudo apt-get update
    sudo apt-get -y install git build-essential libatlas-base-dev libopencv-dev graphviz python-pip
    sudo pip install pip --upgrade
    sudo pip install setuptools numpy --upgrade
    sudo pip install graphviz jupyter
```

Clone the MXNet source code repository using the following `git` command in your home directory:
```
    git clone https://github.com/apache/incubator-mxnet.git --recursive
    cd incubator-mxnet
```

Edit the Makefile to install the MXNet with CUDA bindings to leverage the GPU on the Jetson:
```
    cp make/crosscompile.jetson.mk config.mk
```

Edit the Mshadow Makefile to ensure MXNet builds with Pascal's hardware level low precision acceleration by editing 3rdparty/mshadow/make/mshadow.mk and adding the following after line 122:
```
MSHADOW_CFLAGS += -DMSHADOW_USE_PASCAL=1
```

Now you can build the complete MXNet library with the following command:
```
    make -j $(nproc)
```

Executing this command creates a file called `libmxnet.so` in the mxnet/lib directory.

**Step 2** Install MXNet Python Bindings

To install Python bindings run the following commands in the MXNet directory:

```
    cd python
    pip install --upgrade pip
    pip install -e .
```

Note that the `-e` flag is optional. It is equivalent to `--editable` and means that if you edit the source files, these changes will be reflected in the package installed.

Add the mxnet folder to the path:

```
    cd ..
    export MXNET_HOME=$(pwd)
    echo "export PYTHONPATH=$MXNET_HOME/python:$PYTHONPATH" >> ~/.rc
    source ~/.rc
```

You are now ready to run MXNet on your NVIDIA Jetson TX2 device.

</div> <!-- End of jetson -->
</div> <!-- End of devices -->


<!-- This # tag restarts the page and allows reuse
 of the div classes for validation sections, etc -->


<!-- Download -->
<hr>


# Source Download

<a href="download.html">Download</a> your required version of MXNet and <a href="build_from_source.html">build from source</a>.
