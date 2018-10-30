# Clojure MXNet

A Clojure Package Built on the MXNet Deep Learning Library

## Introduction

MXNet is a flexible and efficient deep learning library. While its core is built in C++ for maximum performance, support for multiple programming languages in intermediate and high-level APIs is a first-class feature. MXNet is currently an incubating Apache project.

The motivation for creating a Clojure package was to give Clojurians access to a world-class deep learning platform, thereby building bridges for future development and innovation in the community. The Clojure package provides all the essential tools, including low-level and high-level APIs, dynamic graphs, and things like GANs and natural language processing.

To maximize leverage, the Clojure package has been built on the existing Scala package using interop. This approach has allowed rapid initial development and close parity with the Scala package functionality. It also leaves the door open to directly developing code against the JNI bindings with Clojure in the future in an incremental fashion, using the test suites as a refactoring guide.

For a **video introduction**, see [Clojure MXNet with Carin Meier - Clojure Virtual Meetup](https://www.crowdcast.io/e/clojure-mxnet-with-carin) (setup instructions from 20:49)

## Current State and Plans

Help is needed testing and generally making the package better. A list of the pacakge status and contribution needs can be found under [Clojure Package Contribution Needs](https://cwiki.apache.org/confluence/display/MXNET/Clojure+Package+Contribution+Needs). Please get involved :)

## Getting Started

The Clojure MXNet framework consists of a core C library, a Scala API that talks to the core through [JNI (Java Native Interface)](https://en.wikipedia.org/wiki/Java_Native_Interface) bindings, and finally a Clojure wrapper around the Scala API.

Since there is a native code involved in the framework, what OS you are running matters.

The following combinations of operating system and compute device are supported:

- Linux CPU
- Linux GPU
- OSX CPU

There are three ways of getting started:

1. Install [prebuilt Clojure jars](https://search.maven.org/search?q=clojure%20mxnet) with the native dependencies baked in. This the quickest way to get going.
2. Install the Clojure package from source, but use prebuilt jars for the native dependencies. Choose this option if you want pre-release features of the Clojure package but don't want to build (compile native dependencies yourself.
3. Build everything from source. This option is for developers or advanced users who want cutting-edge features in all parts of the dependency chain.

**Note:** This guide assumes that you are familiar with the basics of creating Clojure projects and managing dependencies. See [here](https://github.com/technomancy/leiningen/blob/stable/doc/TUTORIAL.md) for the official Leiningen tutorial.

### Option 1: Clojure Package from Prebuilt Jar

If you are new to MXNet and just want to try things out, this option is the best way to get started. You will install release versions of MXNet core, MXNet Scala and MXNet Clojure.

For reference, the Clojure MXNet jars can be found on [maven.org](https://search.maven.org/search?q=clojure%20mxnet).

#### Installing additional dependencies

Depending on your operating system, you will need a couple of packages that are not distributed through Maven:

- [OpenCV](https://opencv.org/) version 3.4
- [OpenBLAS](http://www.openblas.net/)
- [ATLAS](http://math-atlas.sourceforge.net/)
- [cURL](https://curl.haxx.se/) library version 3

##### Linux (Ubuntu)

As of writing this, OpenCV 3.4 is not available in the default repositories. Therefore, a third-party repository is needed.

```bash
sudo add-apt-repository ppa:timsc/opencv
sudo apt-get update
sudo apt install libopencv-imgcodecs3.4 libopenblas-base libatlas3-base libcurl3
```

Note: `libcurl3` may conflict with other packages on your system. [Here](https://github.com/apache/incubator-mxnet/issues/12822) is a possible workaround.

##### Linux (Arch)

```bash
yaourt -S openblas-lapack
yaourt -S libcurl-compat
export LD_PRELOAD=libcurl.so.3
```

To enable GPU support, you will additionally need the CUDA toolkit:

```bash
wget https://archive.archlinux.org/packages/c/cuda/cuda-9.0.176-4-x86_64.pkg.tar.xz
sudo pacman -U cuda-9.0.176-4-x86_64.pkg.tar.xz
```

##### OSX

```bash
brew install wget
brew install opencv
```

#### Installing the Clojure package

- Create a new project with `lein new my-mxnet`
- Edit your `project.clj` based on your system with the desired jar from maven. It will be one of the following:

  - `[org.apache.mxnet.contrib.clojure/clojure-mxnet-linux-cpu "1.3.0"]`
  - `[org.apache.mxnet.contrib.clojure/clojure-mxnet-linux-gpu "1.3.0"]`
  - `[org.apache.mxnet.contrib.clojure/clojure-mxnet-osx-cpu "1.3.0"]`

At this point you should be able to run your own example like this [NDArray Tutorial](https://github.com/apache/incubator-mxnet/blob/master/contrib/clojure-package/examples/tutorial/src/tutorial/ndarray.clj).

### Option 2: Clojure package from Source, Scala Package from Jar

With this option, you will install a Git revision of the Clojure package source and a [Scala package jar from Maven](https://search.maven.org/search?q=g:org.apache.mxnet) with native dependencies baked in.

- Install additional dependencies as described in [the corresponding section for Option 1](#installing-additional-dependencies),
- Recursively clone the MXNet repository and checkout the desired revision. Here we assume the `1.3.0` tag and a clone into the `~/mxnet` directory:

  ```bash
  git clone --recursive https://github.com/apache/incubator-mxnet.git ~/mxnet
  cd ~/mxnet
  git tag --list  # Get the tag that matches the Scala package version
  git checkout tags/1.3.0 -b my_mxnet
  cd contrib/clojure
  ```

- Edit `project.clj` to include the desired Scala jar from Maven:

    [org.apache.mxnet/mxnet-full_2.11-linux-x86_64-cpu "1.3.0”]

- Run `lein test`. All the tests should run without error.
- At this point you can run `lein install` to build and install the Clojure jar locally.

To run examples, you can now use `lein run` in any of the example directories, e.g., `examples/imclassification`. You can also specify the compute device, e.g., `lein run :cpu 2` (for 2 CPUs) or `lein run :gpu` (1 GPU).

### Option 3: Everything from Source

This option will build the C lib locally and then the Scala jars as well. You can use this option when you are developing changes for the project.

For your dependencies look at https://mxnet.incubator.apache.org/install/osx_setup.html or https://mxnet.incubator.apache.org/install/ubuntu_setup.html
about _Prepare Environment for GPU Installation_
and _Install MXNet dependencies_

Also, ensure you have JDK 8 on your system. Later versions may produce cryptic build errors mentioning `scala.reflect.internal.MissingRequirementError`. 

Checkout the latest SHA from the main package:

`git clone --recursive https://github.com/apache/incubator-mxnet.git ~/mxnet`
`cd ~/mxnet`

If you need to checkout a particular release you can do it with:

`git checkout tags/<tag_name> -b <branch_name>`

`git submodule update --init --recursive`

Sometimes it useful to use this script to clean hard using this [gist](https://gist.github.com/nicktoumpelis/11214362)


Go here to do the base package installation https://mxnet.incubator.apache.org/install/index.html.
This will create an `lib/libmxnet.so` file that is the c API we need for the next step.

Run `make scalapkg` then `make scalainstall`. This will create the Scala jars and install them in your local maven.
 
Next, `cd contrib/clojure` and edit the project.clj to include the Scala jar that was just created and installed in your maven. Example `[org.apache.mxnet/mxnet-full_2.11-osx-x86_64-cpu "1.3.0-SNAPSHOT"]`.

- run `lein test`. All the tests should run without an error
- At this point you can do `lein install` to build and install the Clojure jar locally. Now, you can run the examples by doing `cd examples/imclassification` and then `lein run`  or `lein run :cpu 2` (for 2 CPUs) (for GPU `lein run :gpu`)

## Docker Files

There are Dockerfiles available as well.

- [Community Provided by Magnet](https://hub.docker.com/u/magnetcoop/)
- [MXNet CI](https://github.com/apache/incubator-mxnet/blob/master/ci/docker/Dockerfile.build.ubuntu_cpu) and the install scripts
  - [Ubuntu core](https://github.com/apache/incubator-mxnet/blob/master/ci/docker/install/ubuntu_core.sh)
  - [Ubuntu Scala](https://github.com/apache/incubator-mxnet/blob/master/ci/docker/install/ubuntu_scala.sh)
  - [Ubuntu Clojure](https://github.com/apache/incubator-mxnet/blob/master/ci/docker/install/ubuntu_clojure.sh)

## Need Help?

If you are having trouble getting started or have a question, feel free to reach out at:

- Clojurian Slack #mxnet channel. To join, go to [http://clojurians.net/](http://clojurians.net/).
- Apache Slack #mxnet and #mxnet-scala channel (To join this slack send an email to dev@mxnet.apache.org)
- Create an Issue on [https://github.com/apache/incubator-mxnet/issues](https://github.com/apache/incubator-mxnet/issues)


## Examples
There are quite a few examples in the examples directory. To use.

`lein install` in the main project
`cd` in the the example project of interest

There are README is every directory outlining instructions.

A good place to get started is the module example.
Do `lein run` for the CPU version or `lein run :gpu` for GPU.

## Generating documentation

To generate API docs, run `lein codox`. The html docs will be generated in the target/docs directory.

## Code Coverage

To run the Code Coverage tool. Run `lein cloverage`.

## Tools to keep style consistent

To keep the style consistent for the project we include the script that make it easier.
There are two script in the base of the project and in each examples.

To run it just see the following file. `lein-cljfmt-check` and `lein-cljfmt-fix`.
The first command will run and check and confirm if the code needed to be updated to reflect the community style guide.
The second command will apply the change and fix any inconsistent indentation in place. This is recommendd to be done
before the submit a new pull request so we can keep the style consistent throughout the project.

## FAQ

**Why build on the Scala package?**

The motivation section addresses this, but the main reason is high leverage is using the great work that the Scala package has already done.

**How can I tell if the GPU is being used?**

CUDA is finding a best algorithm... As long as a Context.gpu() passed in the code as a context, GPU should be used.

This command can be very handy too

`nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 5
timestamp, name, utilization.gpu [%], utilization.memory [%], memory.total [MiB], memory.free [MiB], memory.used [MiB]`

**Supported APIs**
There are 3 high level APIs supported in MXNet: (Model/FeedForward), Module, and Gluon. The Module API is supported in the Clojure package because of the existing support for it in the Scala package. The Module API is very similar to the Gluon API and examples of the usage can be found in the examples directory. The Model/FeedForward API is deprected.

Gluon support will come later and may or may not be built on the Scala gluon API (when it lands there)

## Architecture & Design

See the Confluence page: https://cwiki.apache.org/confluence/display/MXNET/MXNet+Clojure

## Building and Deploying Jars

The release process for deploying the Clojure jars is on the [wiki](https://cwiki.apache.org/confluence/display/MXNET/Clojure+Release+Process).


## Special Thanks
Special thanks to people that provided testing and feedback to make this possible

- Chris Hodapp
- Iñaki Arenaza & Magnet Coop
- r0man
- Ben Kamphaus
- Sivaram Konanki
- Rustam Gilaztdinov
- Kamil Hryniewicz
- Christian Weilbach
- Burin Choomnuan
- Avram Aelony
- Jim Dunn
- Kovas Boguta
