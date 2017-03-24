# MXNet: A Scalable Deep Learning Framework

MXNet is an open-source deep learning framework that allows you to define,
train, and deploy deep neural networks on a wide array of devices, from cloud
infrastructure to mobile devices.  It is highly scalable, allowing for fast
model training, and supports a flexible programming model and multiple
languages. MXNet allows you to mix symbolic and imperative programming flavors
to maximize both efficiency and productivity.  MXNet is built on a dynamic
dependency scheduler that automatically parallelizes both symbolic and
imperative operations on the fly.  A graph optimization layer on top of that
makes symbolic execution fast and memory efficient. The MXNet library is
portable and lightweight, and it scales to multiple GPUs and multiple machines.

## Quick Overview

<div id="lang-demo">
<ul class="nav nav-tabs" role="tablist">
<li role="presentation" class="active">
<a href="#python-demo" role="tab" data-toggle="tab">Python</a>
</li>
<li role="presentation">
<a href="#scala-demo" role="tab" data-toggle="tab">Scala</a>
</li>
<li role="presentation">
<a href="#r-demo" role="tab" data-toggle="tab">R</a>
</li>
<li role="presentation">
<a href="#julia-demo" role="tab" data-toggle="tab">Julia</a>
</li>
</ul>
<div class="tab-content">
<div role="tabpanel" class="tab-pane active" id="python-demo">

```python
>>> import mxnet as mx
>>> a = mx.nd.ones((2, 3), mx.gpu())
>>> print ((a * 2).asnumpy())
[[ 2.  2.  2.]
 [ 2.  2.  2.]]
```

</div> <!-- python-demo -->
<div role="tabpanel" class="tab-pane" id="scala-demo">

```scala
scala> import ml.dmlc.mxnet._
import ml.dmlc.mxnet._

scala> val arr = NDArray.ones(2, 3)
arr: ml.dmlc.mxnet.NDArray = ml.dmlc.mxnet.NDArray@f5e74790

scala> arr.shape
res0: ml.dmlc.mxnet.Shape = (2,3)

scala> (arr * 2).toArray
res2: Array[Float] = Array(2.0, 2.0, 2.0, 2.0, 2.0, 2.0)

scala> (arr * 2).shape
res3: ml.dmlc.mxnet.Shape = (2,3)
```

</div> <!-- scala-demo -->
<div role="tabpanel" class="tab-pane" id="r-demo">

```r
> require(mxnet)
Loading required package: mxnet
> a <- mx.nd.ones(c(2,3))
> a
     [,1] [,2] [,3]
[1,]    1    1    1
[2,]    1    1    1
> a + 1
     [,1] [,2] [,3]
[1,]    2    2    2
[2,]    2    2    2
```

</div> <!-- r-demo -->
<div role="tabpanel" class="tab-pane" id="julia-demo">

```julia
julia> using MXNet

julia> a = mx.ones((2,3), mx.gpu())
mx.NDArray{Float32}(2,3)

julia> Array{Float32}(a * 2)
2×3 Array{Float32,2}:
 2.0  2.0  2.0
 2.0  2.0  2.0
```

</div> <!-- julia-demo -->
</div>
</div>

## Setup MXNet

Get the instructions to setup MXNet by selecting your preference.

<div id="setup-options">
<div class="option-row", data-key="os">
  <div class="option-title">OS:</div>
  <div class="option-select">
    <div class="btn-group"  role="group">
      <button type="button" class="btn btn-default active">Linux</button>
      <button type="button" class="btn btn-default">Mac OS X</button>
      <button type="button" class="btn btn-default">Windows</button>
    </div>
  </div>
</div>

<div class="option-row", data-key="lang">
  <div class="option-title">Language:</div>
  <div class="option-select">
    <div class="btn-group"  role="group">
      <button type="button" class="btn btn-default active">Python</button>
      <button type="button" class="btn btn-default">Scala</button>
      <button type="button" class="btn btn-default">R</button>
      <button type="button" class="btn btn-default">Julia</button>
    </div>
  </div>
</div>

<!-- <div class="option-row", data-key="driver"> -->
<!--   <div class="option-title">Driver:</div> -->
<!--   <div class="option-select"> -->
<!--     <div class="btn-group"  role="group"> -->
<!--       <button type="button" class="btn btn-default active">CPU</button> -->
<!--       <button type="button" class="btn btn-default">MKL-DNN</button> -->
<!--       <button type="button" class="btn btn-default">CUDA</button> -->
<!--     </div> -->
<!--   </div> -->
<!-- </div> -->

<div class="option-row", data-key="type">
  <div class="option-title">Install type:</div>
  <div class="option-select">
    <div class="btn-group"  role="group">
      <button type="button" class="btn btn-default">Build From Source</button>
      <button type="button" class="btn btn-default active">Pre-Built Binaries</button>
      <button type="button" class="btn btn-default">Docker</button>
      <button type="button" class="btn btn-default">Cloud</button>
    </div>
  </div>
</div>
</div> <!-- setup-options -->

<div class="alert alert-info" role="alert" id="not_found">
Instructions will come soon!
</div>

<!-- <h2 id="install-inst-title"></h2> -->

<div class="install-inst linux_python_pre-built-binaries"

<h3>Install with pip</h3>

First install `pip` if necessary. For example

```bash
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
```

There are multiple pre-builds based on different drivers:

- **default**: supports CPU, suits for fast developing.
- **mkl-dnn**: compiled with [MKL-DNN](https://github.com/01org/mkl-dnn) to accelerate
  the Intel CPU performance.
- **cuda**: compiled with both [CUDA](https://developer.nvidia.com/cuda-toolkit)
  and [cuDNN](https://developer.nvidia.com/cudnn) to accelerate the performance on
  Nvidia GPUs. It requires both CUDA and cuDNN are installed.

Use one of following commands to install the desired release:

```bash
pip install mxnet       # default
pip install mxnet-mkl   # for MKL-DNN
pip install mxnet-cu75  # for CUDA 7.5
pip install mxnet-cu80  # for CUDA 8.0
```

Troubleshoot:

- If you see the following error:

  ```bash
  Downloading/unpacking mxnet
    Could not find any downloads that satisfy the requirement mxnet
  Cleaning up...
  No distributions at all found for mxnet
  ```

  A possible reason is that your `pip` version is too low, e.g., if you
  installed it by `apt-get install python-pip` on Ubuntu 14.04. One way is
  first removing the old version by `sudo apt-get purge python-pip`. Next
  install the new version using the above instructions.

</div>

<div class="install-inst mac-os-x_python_pre-built-binaries">

<h3>Install with pip</h3>

First install `pip` if necessary. For example

```bash
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
```

Then install the CPU version by

```bash
pip install mxnet
```
</div>

<div class="install-inst mac-os-x_python_docker
mac-os-x_scala_docker mac-os-x_r_docker
mac-os-x_julia_docker linux_python_docker
linux_scala_docker linux_r_docker
linux_julia_docker windows_python_docker
windows_julia_docker windows_scala_docker
windows_r_docker">

To run MXNet within a docker contrainer, we should first have
[docker](https://www.docker.com/) installed. If we want to use Nvida GPUs,we can
use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker/wiki).  to launch
the docker containers.

The dockerfiles are available at [mxnet/docker](https://github.com/dmlc/mxnet/tree/master/docker)

</div>

<div class="install-inst linux_python_cloud
linux_scala_cloud linux_r_cloud
linux_julia_cloud">

AWS images with MXNet installed:

- [Deep Learning AMI for Ubuntu](https://aws.amazon.com/marketplace/pp/B06VSPXKDX)
- [Deep Learning AMI for Amazon Linux](https://aws.amazon.com/marketplace/pp/B01M0AXXQB)

</div>

<div class="install-inst linux_python_build-from-source
linux_scala_build-from-source linux_r_build-from-source
linux_julia_build-from-source">

**Step 1**: build the shared library `libmxnet.so` from C++ source files
  according to the Linux release:

- [Ubuntu](http://mxnet.io/get_started/ubuntu_setup.html#build-the-shared-library)
- [CentOS](http://mxnet.io/get_started/centos_setup.html#build-mxnet-shared-library)
- [Amazon Linux]()
- [Raspbian for Raspberry Pi]()

</div>


<div class="install-inst linux_python_build-from-source">

**Step 2**: setup the Python package according to the Linux release:

- [Ubuntu](http://mxnet.io/get_started/ubuntu_setup.html#install-the-mxnet-package-for-python)
- [CentOS, Amazon Linux](http://mxnet.io/get_started/amazonlinux_setup.html#install-the-mxnet-package-for-python)
- [Raspbian for Raspberry Pi](http://mxnet.io/get_started/raspbian_setup.html#install-mxnet-python-bindings)

</div>

<div class="install-inst linux_scala_build-from-source">

**Step 2**: setup the Scala package.

- [Ubuntu](http://mxnet.io/get_started/ubuntu_setup.html#install-the-mxnet-package-for-scala)
- [CentOS, Amazon Linux](http://mxnet.io/get_started/amazonlinux_setup.html#install-the-mxnet-package-for-scala)

</div>

<div class="install-inst linux_r_build-from-source">

**Step 2**: setup the R package.

- [Ubuntu](http://mxnet.io/get_started/ubuntu_setup.html#install-the-mxnet-package-for-r)
- [CentOS, Amazon Linux](http://mxnet.io/get_started/amazonlinux_setup.html#install-the-mxnet-package-for-r)

</div>

<div class="install-inst linux_julia_build-from-source">

**Step 2**: setup the Julia package.

- [Ubuntu](http://mxnet.io/get_started/ubuntu_setup.html#install-the-mxnet-package-for-julia)
- [CentOS, Amazon Linux](http://mxnet.io/get_started/amazonlinux_setup.html#install-the-mxnet-package-for-julia)

</div>

<div class="install-inst mac-os-x_python_build-from-source
mac-os-x_scala_build-from-source mac-os-x_r_build-from-source
mac-os-x_julia_build-from-source">

**Step 1**: [build the shared library from C++ source files](http://mxnet.io/get_started/osx_setup.html#build-the-shared-library)

</div>

<div class="install-inst mac-os-x_python_build-from-source">

**Step 2**: [setup the Python package](http://mxnet.io/get_started/osx_setup.html#install-the-mxnet-package-for-python)

</div>

<div class="install-inst mac-os-x_scala_build-from-source">

**Step 2**: [setup the Scala package](http://mxnet.io/get_started/osx_setup.html#install-the-mxnet-package-for-scala)

</div>

<div class="install-inst mac-os-x_r_build-from-source">

**Step 2**: [setup the R package](http://mxnet.io/get_started/osx_setup.html#install-the-mxnet-package-for-r)

</div>

<div class="install-inst mac-os-x_julia_build-from-source">

**Step 2**: [setup the Julia package](http://mxnet.io/get_started/osx_setup.html#install-the-mxnet-package-for-julia)

</div>

<div class="install-inst windows_python_build-from-source
windows_scala_build-from-source windows_r_build-from-source
windows_julia_build-from-source">

**Step 1**: [build the shared library from C++ source files](http://mxnet.io/get_started/windows_setup.html#build-the-shared-library)

</div>

<div class="install-inst windows_python_build-from-source">

**Step 2**: [setup the Python package](http://mxnet.io/get_started/windows_setup.html#install-mxnet-for-python)

</div>

<div class="install-inst windows_scala_build-from-source">

**Step 2**: [setup the Scala package](http://mxnet.io/get_started/windows_setup.html#install-mxnet-for-scala)

</div>

<div class="install-inst windows_r_build-from-source">

**Step 2**: [setup the R package](http://mxnet.io/get_started/windows_setup.html#install-mxnet-for-r)

</div>

<div class="install-inst windows_julia_build-from-source">

**Step 2**: [setup the Julia package](http://mxnet.io/get_started/windows_setup.html#install-mxnet-for-julia)

</div>

<script>
  var opts = {
  os: 'Linux', lang: 'Python', type: 'Pre-built Binaries'
  };
  function getLabelArray() {
  return [opts.os, opts.lang, opts.type];
  }
  function updateInstruction() {
  var label = '.' + getLabelArray().join('_').replace(/[ .]/g, '-').toLowerCase();
  $('#install-inst-title').html(label);
  $('.install-inst').hide();
  $("#not_found").hide();
  if ($(label).length == 0) {
    $("#not_found").show();
  } else {
    $(label).show();
  }
  }
  updateInstruction();
  function selectOption(ev) {
  var el = $(this);
  el.siblings().removeClass('active');
  el.addClass('active');
  opts[el.parents('.option-row').data('key')] = el.text();
  updateInstruction();
  }
  $('.btn-group').on('click', '.btn', selectOption);
</script>

## Next Steps

* [Tutorials](http://mxnet.io/tutorials/index.html)
* [How To](http://mxnet.io/how_to/index.html)
* [API Documents](http://mxnet.io/api/index.html)
