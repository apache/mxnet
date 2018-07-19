# Clojure MXNet

A clojure package to the MXNet Deep Learning library

## Introduction

MXNet is a first class, modern deep learning library. It supports multiple languages on a first class basis and is incubating as an Apache project.

The motivation for creating a Clojure package is to be able to open the deep learning library to the Clojure ecosystem and build bridges for future development and innovation for the community. It provides all the needed tools including low level and high level apis, dynamic graphs, and things like GAN and natural language support.

For high leverage, the Clojure package has been built on the existing Scala package using interop. This has allowed rapid development and close parity with the Scala functionality. This also leaves the door open to directly developing code against the jni-bindings with Clojure in the future in an incremental fashion, using the test suites as a refactoring guide.

## Current State and Plans

The Clojure package is nearing the end of its first development milestone which is to achieve a close parity with the Scala package.

Help is needed testing and generally making the package better. A list of the pacakge status and contribution needs can be found here [Clojure Package Contribution Needs](https://cwiki.apache.org/confluence/display/MXNET/Clojure+Package+Contribution+Needs). Please get involved :)

Testing instructions can be found in the testing.md.

## Getting Started

The following systems are supported:

- OSX cpu
- Linux cpu
- Linux gpu

There are two ways of getting going. The first way is the easiest and that is to use the pre-built jars from Maven. The second way is to build from source. In both cases, you will need to load the prereqs and dependencies, (like opencv).



### Prerequisites


Follow the instructions from https://mxnet.incubator.apache.org/install/osx_setup.html or https://mxnet.incubator.apache.org/install/ubuntu_setup.html
about _Prepare Environment for GPU Installation_
and _Install MXNet dependencies_


#### Cloning the repo and running from source

To use the prebuilt jars (easiest), you will need to replace the native version of the line in the project dependencies with your configuration.

`[org.apache.mxnet/mxnet-full_2.11-linux-x86_64-gpu "1.2.1"]`
or
`[org.apache.mxnet/mxnet-full_2.11-linux-x86_64-cpu "1.2.1"]`
or
`[org.apache.mxnet/mxnet-full_2.11-osx-x86_64-cpu "1.2.1"]`

If you are using the prebuilt jars they may have a slightly different dependencies then building from source:

*For OSX you will need:*

`brew install opencv`

*For Ubuntu Linux you will need:*

```
sudo add-apt-repository ppa:timsc/opencv-3.4
sudo apt-get update
sudo apt install libopencv-imgcodecs3.4
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

If you want to see the exact versions and flags that the jars were built with, look here:
[Scala Release Process](https://cwiki.apache.org/confluence/display/MXNET/MXNet-Scala+Release+Process)


Check your installation with `lein test`. If that works alright then, you can try some code!

```clojure

(ns tutorial.ndarray
  (:require [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.context :as context]))

;;Create NDArray
(def a (ndarray/zeros [100 50])) ;;all zero arrray of dimension 100 x 50
(def b (ndarray/ones [256 32 128 1])) ;; all one array of dimension
(def c (ndarray/array [1 2 3 4 5 6] [2 3])) ;; array with contents of a shape 2 x 3

;;; There are also ways to convert to a vec or get the shape as an object or vec
(ndarray/->vec c) ;=> [1.0 2.0 3.0 4.0 5.0 6.0]
```

See the examples/tutorial section for more.


The jars from maven with the needed MXNet native binaries in it. On startup, the native libraries are extracted from the jar and copied into a temporary location on your path. On termination, they are deleted.


### Build from MXNET Source

Checkout the latest sha from the main package

`git clone --recursive https://github.com/apache/incubator-mxnet.git ~/mxnet`
`cd ~/mxnet`

If you need to checkout a particular release you can do it with:

`git checkout tags/1.2.1 -b release-1.2.1`

`git submodule update --init --recursive`

Sometimes it useful to use this script to clean hard
https://gist.github.com/nicktoumpelis/11214362


Go here to do the base package installation https://mxnet.incubator.apache.org/install/index.html

 Run `make scalapkg` then `make scalainstall`

then replace the correct jar for your architecture in the project.clj, example `[org.apache.mxnet/mxnet-full_2.11-osx-x86_64-cpu "1.3.0-SNAPSHOT"]`

#### Test your installation

To test your installation, you should run `lein test`. This will run the test suite (CPU) for the clojure package.


#### Generation of NDArray and Symbol apis

The bulk of the ndarray and symbol apis are generated via java reflection into the Scala classes. The files are generated as a compile time step (AOT) in the `dev.generator` namespace.

You may also run this manually with the repl functions:

`(generate-ndarray-file)`
and
`(generate-symbol-file)`


These will generate the files under `src/org.apache.clojure-mxnet/gen/` that are loaded by the `src/org.apache.clojure-mxnet/ndarray.clj` and `src/org.apache.clojure-mxnet/symbol.clj` files.


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
The process to build and deploy the jars currently is a manual process using the `lein` build tool and `Clojars`, the Clojure dependency hosting platform.

There is one jar for every system supported.

- Comment out the line in the `project.clj` for the system that you are targeting, (example OSX cpu you would uncomment out ` [org.apache.mxnet/mxnet-full_2.11-osx-x86_64-cpu "1.2.0"]` but leave the linux deps commented)
- Change the `defproject org.apache.mxnet.contrib.clojure/clojure-mxnet "0.1.1-SNAPSHOT"` in the project to reference the correct version number and jar description. For example changing the line to be `org.apache.mxnet.contrib.clojure/mxnet-osx-cpu "0.1.2"` would create a jar with the group id of `org.apache.mxnet.contrib.clojure` and the artifact name of `mxnet-osx-cpu` and the version of `0.1.2`
- Run `lein clean`
- Run `lein jar` to create the jar
- Check that the jar looks alright in the `/target` directory.

To deploy the jar to Clojars, you do `lein deploy clojars` and it will prompt you for your username and password.

_Note: Integration with deployment to Nexus can be enabled too for the future [https://help.sonatype.com/repomanager2/maven-and-other-build-tools/leiningen](https://help.sonatype.com/repomanager2/maven-and-other-build-tools/leiningen)_

You would repeat this process for all the build system types.


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
