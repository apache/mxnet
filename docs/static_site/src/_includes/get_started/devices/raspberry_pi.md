
MXNet supports the Debian based Raspbian ARM based operating system so you can run MXNet on
Raspberry Pi 3B
devices.

These instructions will walk through how to build MXNet for the Raspberry Pi and install the
Python bindings
for the library.

You can do a dockerized cross compilation build on your local machine or a native build
on-device.

The complete MXNet library and its requirements can take almost 200MB of RAM, and loading
large models with
the library can take over 1GB of RAM. Because of this, we recommend running MXNet on the
Raspberry Pi 3 or
an equivalent device that has more than 1 GB of RAM and a Secure Digital (SD) card that has
at least 4 GB of
free memory.

## Quick installation
You can use this [pre-built Python
wheel](https://mxnet-public.s3.amazonaws.com/install/raspbian/mxnet-1.5.0-py2.py3-none-any.whl)
on a
Raspberry Pi 3B with Stretch. You will likely need to install several dependencies to get
MXNet to work.
Refer to the following **Build** section for details.

## Docker installation
**Step 1** Install Docker on your machine by following the [docker installation
instructions](https://docs.docker.com/engine/installation/linux/ubuntu/#install-using-the-repository).

*Note* - You can install Community Edition (CE)

**Step 2** [Optional] Post installation steps to manage Docker as a non-root user.

Follow the four steps in this [docker
documentation](https://docs.docker.com/engine/installation/linux/linux-postinstall/#manage-docker-as-a-non-root-user)
to allow managing docker containers without *sudo*.

## Build

**This cross compilation build is experimental.**

**Please use a Native build with gcc 4 as explained below, higher compiler versions
currently cause test
failures on ARM.**

The following command will build a container with dependencies and tools,
and then compile MXNet for ARMv7.
You will want to run this on a fast cloud instance or locally on a fast PC to save time.
The resulting artifact will be located in `build/mxnet-x.x.x-py2.py3-none-any.whl`.
Copy this file to your Raspberry Pi.
The previously mentioned pre-built wheel was created using this method.

{% highlight bash %}
ci/build.py -p armv7
            {% endhighlight %}

## Install using a pip wheel

Your Pi will need several dependencies.

Install MXNet dependencies with the following:

{% highlight bash %}
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
python-pip \
software-properties-common \
sudo \
unzip \
virtualenv \
wget
{% endhighlight %}

Install virtualenv with:

{% highlight bash %}
sudo pip install virtualenv
{% endhighlight %}

Create a Python 2.7 environment for MXNet with:

{% highlight bash %}
virtualenv -p `which python` mxnet_py27
{% endhighlight %}

You may use Python 3, however the [wine bottle detection
example](https://mxnet.apache.org/api/python/docs/tutorials/deploy/inference/wine_detector.html)
for the
Pi with camera requires Python 2.7.

Activate the environment, then install the wheel we created previously, or install this
[prebuilt
wheel](https://mxnet-public.s3.amazonaws.com/install/raspbian/mxnet-1.5.0-py2.py3-none-any.whl).

{% highlight bash %}
source mxnet_py27/bin/activate
pip install mxnet-x.x.x-py2.py3-none-any.whl
{% endhighlight %}

Test MXNet with the Python interpreter:

{% highlight python %}
$ python

>>> import mxnet
{% endhighlight %}

If there are no errors then you're ready to start using MXNet on your Pi!

## Native Build

Installing MXNet from source is a two-step process:

1. Build the shared library from the MXNet C++ source code.
2. Install the supported language-specific packages for MXNet.

**Step 1** Build the Shared Library

On Raspbian versions Wheezy and later, you need the following dependencies:

- Git (to pull code from GitHub)

- libblas (for linear algebraic operations)

- libopencv (for computer vision operations. This is optional if you want to save RAM and
Disk Space)

- A C++ compiler that supports C++ 11. The C++ compiler compiles and builds MXNet source
code. Supported
compilers include the following:

- [G++ (4.8 or later)](https://gcc.gnu.org/gcc-4.8/). Make sure to use gcc 4 and not 5 or 6
as there are
known bugs with these compilers.
- [Clang (3.9 - 6)](https://clang.llvm.org/)

Install these dependencies using the following commands in any directory:

{% highlight bash %}
sudo apt-get update
sudo apt-get -y install git cmake ninja-build build-essential g++-4.9 c++-4.9 liblapack*
libblas* libopencv*
libopenblas* python3-dev python-dev virtualenv
{% endhighlight %}

Clone the MXNet source code repository using the following `git` command in your home
directory:

{% highlight bash %}
git clone https://github.com/apache/incubator-mxnet.git --recursive
cd incubator-mxnet
{% endhighlight %}

Build:

{% highlight bash %}
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
{% endhighlight %}
Some compilation units require memory close to 1GB, so it's recommended that you enable swap
as
explained below and be cautious about increasing the number of jobs when building (-j)

Executing these commands start the build process, which can take up to a couple hours, and
creates a file
called `libmxnet.so` in the build directory.

If you are getting build errors in which the compiler is being killed, it is likely that the
compiler is running out of memory (especially if you are on Raspberry Pi 1, 2 or Zero, which
have
less than 1GB of RAM), this can often be rectified by increasing the swapfile size on the Pi
by
editing the file /etc/dphys-swapfile and changing the line CONF_SWAPSIZE=100 to
CONF_SWAPSIZE=1024,
then running:
{% highlight bash %}sudo /etc/init.d/dphys-swapfile stop
sudo /etc/init.d/dphys-swapfile start
free -m # to verify the swapfile size has been increased
{% endhighlight %}

**Step 2** Build cython modules (optional)

{% highlight bash %}
$ pip install Cython
$ make cython # You can set the python executable with `PYTHON` flag, e.g., make cython
PYTHON=python3
{% endhighlight %}

*MXNet* tries to use the cython modules unless the environment variable
`MXNET_ENABLE_CYTHON` is set to `0`.
If loading the cython modules fails, the default behavior is falling back to ctypes without
any warning. To
raise an exception at the failure, set the environment variable `MXNET_ENFORCE_CYTHON` to
`1`. See
[here](https://mxnet.apache.org/api/faq/env_var) for more details.


**Step 3** Install MXNet Python Bindings

To install Python bindings run the following commands in the MXNet directory:

{% highlight bash %}
cd python
pip install --upgrade pip
pip install -e .
{% endhighlight %}

Note that the `-e` flag is optional. It is equivalent to `--editable` and means that if you
edit the source
files, these changes will be reflected in the package installed.

Alternatively you can create a whl package installable with pip with the following command:

{% highlight bash %}ci/docker/runtime_functions.sh build_wheel python/ $(realpath build)
{% endhighlight %}


You are now ready to run MXNet on your Raspberry Pi device. You can get started by following
the tutorial on
[Real-time Object Detection with MXNet On The Raspberry
Pi](https://mxnet.io/api/python/docs/tutorials/deploy/inference/wine_detector.html).

*Note - Because the complete MXNet library takes up a significant amount of the Raspberry
Pi's limited RAM,
when loading training data or large models into memory, you might have to turn off the GUI
and terminate
running processes to free RAM.*
