# Clojure MXNet

A clojure package to the MXNet Deep Learning library

## Introduction

MXNet is a first class, modern deep learning library. It supports multiple languages on a first class basis and is incubating as an Apache project.

The motivation for creating a Clojure package is to be able to open the deep learning library to the Clojure ecosystem and build bridges for future development and innovation for the community. It provides all the needed tools including low level and high level apis, dynamic graphs, and things like GAN and natural language support.

For high leverage, the Clojure package has been built on the existing Scala package using interop. This has allowed rapid development and close parity with the Scala functionality. This also leaves the door open to directly developing code against the jni-bindings with Clojure in the future in an incremental fashion, using the test suites as a refactoring guide.

For a **video introduction**, see [Clojure MXNet with Carin Meier - Clojure Virtual Meetup](https://www.crowdcast.io/e/clojure-mxnet-with-carin) (setup instructions from 20:49)

## Current State and Plans

Help is needed testing and generally making the package better. A list of the pacakge status and contribution needs can be found here [Clojure Package Contribution Needs](https://cwiki.apache.org/confluence/display/MXNET/Clojure+Package+Contribution+Needs). Please get involved :)


## Getting Started

The Clojure MXNet framework consists of a core C library, a Scala Api that talks to it through JNI bindings, and finally a Clojure wrapper around the Scala Api.

Since there is a native code involved in the framework, what OS you are running matters.

The following systems are supported:

- OSX cpu
- Linux cpu
- Linux gpu

There are three ways of getting started:

* [Use the Clojure jars with the native dependencies baked in](#getting-started-with-the-clojure-jars-on-maven). This the easiest way to get going.
* [Checkout the MXNet project from a release tag using the Scala jars with native dependencies](#getting-started-with-mxnet-project-with-the-prebuilt-scala-jars). This is also a pretty easy way to get started.
* [Build from the MXNet project master](#getting-started-with-mxnet-project-with-building-from-source). This option can be used to build the whole project yourself.


### Getting Started with the Clojure Jars on Maven

This option is one of the fastest. If you are looking to use MXNet in your own Clojure project, this is your best bet.

The Clojure MXNet jars are found on [maven.org](https://search.maven.org/search?q=clojure%20mxnet).

- Create your own project with `lein new my-mxnet`
- Edit your `project.clj` based on your system with the desired jar from maven. Replace `x.x.x` with the latest version on [maven](https://search.maven.org/search?q=g:org.apache.mxnet.contrib.clojure).

   - [org.apache.mxnet.contrib.clojure/clojure-mxnet-linux-gpu "x.x.x"]
   - [org.apache.mxnet.contrib.clojure/clojure-mxnet-linux-cpu "x.x.x"]
   - [org.apache.mxnet.contrib.clojure/clojure-mxnet-osx-cpu "x.x.0"]


You also might need the following dependencies installed depending on your system because of the native dependencies assumed in the clojure-mxnet jars.

*For OSX you will need:*

`brew install opencv`

*For Ubuntu Linux you will need:*

```
sudo add-apt-repository ppa:timsc/opencv-3.4
sudo apt-get update
sudo apt install libopencv-imgcodecs3.4
```

You may also need

```
libopenblas-base
libatlas3-base
libcurl3
```

*For Arch Linux you will need:*

_CPU_

```
yaourt -S openblas-lapack
yaourt -S libcurl-compat
export LD_PRELOAD=libcurl.so.3
```
_GPU_

```
wget https://archive.archlinux.org/packages/c/cuda/cuda-9.0.176-4-x86_64.pkg.tar.xz
sudo pacman -U cuda-9.0.176-4-x86_64.pkg.tar.xz
```

At this point you should be able to run your own example like this [NDArray Tutorial](https://github.com/apache/incubator-mxnet/blob/master/contrib/clojure-package/examples/tutorial/src/tutorial/ndarray.clj)

### Getting Started with MXNet project with the Prebuilt Scala Jars

This option is also fast. It doesn't require you to build the native C library or the Scala jars, it gets them from [Maven](https://search.maven.org/search?q=g:org.apache.mxnet) as well. Use this if you want to clone the repo and be able to run the tests and examples in the Clojure package.

- `git clone --recursive https://github.com/apache/incubator-mxnet.git ~/mxnet`
- `cd mxnet`
- `git tag —list` (find the tag that corresponds the version of the latest Scala jar)
- `git checkout tags/<tag_name> -b <branch_name>`
- `cd contrib/clojure`
- edit `project.clj` to include your Scala jar from maven. It should match your system. Example `[org.apache.mxnet/mxnet-full_2.11-linux-x86_64-cpu "x.y.z”]`
- run `lein test`. All the tests should run without an error
- At this point you can do `lein install` to build and install the Clojure jar locally. Now, you can run the examples by doing `cd examples/imclassification` and then `lein run`  or `lein run :cpu 2` (for 2 cpus) (for gpu `lein run :gpu`)

You also might need dependencies installed based on your system. Please see the dependency section above in [Getting Started with the Clojure Jars on Maven](#getting-started-with-the-clojure-jars-on-maven))


### Getting Started with MXNet project with Building from Source

This option will build the C lib locally and then the Scala jars as well. You can use this option when you are developing changes for the project.

* For your dependencies look at https://mxnet.incubator.apache.org/install/osx_setup.html or https://mx	net.incubator.apache.org/install/ubuntu_setup.html
	about _Prepare Environment for GPU Installation_
	and	 _Install MXNet dependencies_
* Also, ensure you have JDK 8 on your system. Later versions may produce cryptic build errors mentioning `scala.reflect.internal.MissingRequirementError`. 
* Checkout the latest SHA from the main package:
* `git clone --recursive https://github.com/apache/incubator-mxnet.git ~/mxnet`
* `cd ~/mxnet`
* `git tag —list` (find the tag that corresponds the version of the latest Scala jar)
* `git checkout tags/<tag_name> -b <branch_name>` To checkout a particular release
* `git submodule update --init --recursive` - To updated all the submodules to the commit
  * Sometimes it is useful to use this script to clean hard, (remove any extra files and reset everything), using this [gist](https://gist.github.com/nick	toumpelis/11214362)
* Go here to do the base package installation https://mxnet.incubator.apache.org/install/index.html.
  * This will create an `lib/libmxnet.so` file that is the c api we need for the next step.
* Run `make scalapkg` This will create the Scala jars.
* Run `make scalainstall`. This will install them in your local maven.
* `cd contrib/clojure` and edit the project.clj to include the Scala jar that was just created and installed in your maven. Example `[org.apache.mxnet/mxnet-full_2.11-osx-x86_64-cpu "1.3.0-SNAPSHOT"]`.
* run `lein test`. All the tests should run without an error
*  At this point you can do `lein install` to build and install the clojure jar locally
*  Now, you can run the examples by doing `cd examples/imclassification` and then `lein run`  or `lein run :cpu 2` (for 2 cpus) (for gpu `lein run :gpu`)


## Docker Files

There are Dockerfiles available as well.

- [Community Provided by Magnet](https://hub.docker.com/u/magnetcoop/)
- [MXNet CI](https://github.com/apache/incubator-mxnet/blob/master/ci/docker/Dockerfile.build.ubuntu_cpu) and the install scripts
  - [ubuntu core](https://github.com/apache/incubator-mxnet/blob/master/ci/docker/install/ubuntu_core.sh)
  - [ubuntu scala](https://github.com/apache/incubator-mxnet/blob/master/ci/docker/install/ubuntu_scala.sh)
  - [ubuntu clojure](https://github.com/apache/incubator-mxnet/blob/master/ci/docker/install/ubuntu_clojure.sh)

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
Do `lein run` for the cpu version or `lein run :gpu` for gpu.

## Generating documentation

To generate api docs, run `lein codox`. The html docs will be generated in the target/docs directory.

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

**How can I tell if the gpu is being used?**

CUDA is finding a best algorithm... As long as a Context.gpu() passed in the code as a context, GPU should be used.

This command can be very handy too

`nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 5
timestamp, name, utilization.gpu [%], utilization.memory [%], memory.total [MiB], memory.free [MiB], memory.used [MiB]`

**Supported APIs**
There are 3 high level apis supported in MXNet: (Model/FeedForward), Module, and Gluon. The Module api is supported in the Clojure package because of the existing support for it in the Scala package. The Module api is very similar to the Gluon api and examples of the usage can be found in the examples directory. The Model/FeedForward Api is deprected.

Gluon support will come later and may or may not be built on the Scala gluon api (when it lands there)

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
