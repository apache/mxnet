# Clojure MXNet

A Clojure Package Built on the MXNet Deep Learning Library

## Introduction

MXNet is a flexible and efficient deep learning library. While its core is built in C++ for maximum performance, support for multiple programming languages in intermediate and high-level APIs is a first-class feature. MXNet is currently an incubating Apache project.

The motivation for creating a Clojure package was to give Clojurians access to a world-class deep learning platform, thereby building bridges for future development and innovation in the community. The Clojure package provides all the essential tools, including low-level and high-level APIs, dynamic graphs, etc., and enables building advanced architectures like GANs or LSTM to tackle challenging applications such as image recognition or natural language processing.

To maximize leverage, the Clojure package has been built on the existing Scala package using [Java Interop](https://clojure.org/reference/java_interop). This approach has allowed rapid initial development and close parity with the Scala package functionality. It also leaves the door open to incrementally developing Clojure code that directly interfaces MXNet core using [JNI](https://en.wikipedia.org/wiki/Java_Native_Interface).

For a **video introduction**, see [Clojure MXNet with Carin Meier - Clojure Virtual Meetup](https://www.crowdcast.io/e/clojure-mxnet-with-carin) (setup instructions from 20:49).

## Current State and Plans

The Clojure MXNet package is currently treated as *user-contributed code* within MXNet, as can be seen from its placement under `contrib` in the source tree. This means that it should first undergo a stabilization period and receive feedback from users before it can graduate to a fully integrated and supported part of MXNet.

That said, because it closely tracks the Scala package, Clojure MXNet can be expected to have a similar level of maturity and stability regarding the low-level functionality. It is mostly in the hand-written Java interop part of the Clojure wrapper where bugs are more likely to be encountered. Such bugs tend to be fixed rather quickly once they are known and their origin is clear (see also [Getting Involved](#getting-involved)).

For an overview of the development status and open problems, please refer to [Clojure Package Contribution Needs](https://cwiki.apache.org/confluence/display/MXNET/Clojure+Package+Contribution+Needs).

## Getting Involved

By far the best way to get involved with this project is to install the Clojure MXNet package, run the examples, play around, build new things with it, and get back to the development team with feedback! Your input can not only help to identify current issues, but also guide the future development of the Clojure package by pointing out must-have features that are currently missing, or by identifying usability or performace problems of high impact.

There are two main ways of reaching out to other users and the package maintainers:

- If you have a question or general feedback, or you encountered a problem but are not sure if it's a bug or a misunderstanding, then the *Apache Slack* (channels `#mxnet` and `#mxnet-scala`) is the best place to turn check out. To join, [ask for an invitation](https://mxnet.apache.org/community/contribute.html#slack) at `dev@mxnet.apache.org`.
- If you found a bug, miss an important feature or want to give feedback directly relevant for development, please head over to the MXNet [GitHub issue page](https://github.com/apache/incubator-mxnet/issues) and create a new issue. If the issue is specific to the Clojure package, consider using a title starting with `[Clojure]` to make it easily discoverable among the many other, mostly generic issues.

Of course, contributions to code or documentation are also more than welcome! Please check out the [Clojure Package Contribution Needs](https://cwiki.apache.org/confluence/display/MXNET/Clojure+Package+Contribution+Needs) to get an idea about where and how to contribute code.

For a more comprehensive overview of different ways to contribute, see [Contributing to MXNet](https://mxnet.apache.org/community/contribute.html).

## Getting Started

The Clojure MXNet framework consists of a core C library, a Scala API that talks to the core through [JNI (Java Native Interface)](https://en.wikipedia.org/wiki/Java_Native_Interface) bindings, and finally a Clojure wrapper around the Scala API.

Since the core contains native (compiled) code and is bundled with the language bindings, your hardware and OS matter to the choices to be made during installation. The following combinations of operating system and compute device are supported:

- Linux CPU
- Linux GPU
- OSX CPU

There are three ways of getting started:

1. [Install prebuilt Clojure jars](#option-1-clojure-package-from-prebuilt-jar) with the native dependencies baked in. This the quickest way to get going.
2. [Install the Clojure package from source, but use prebuilt jars for the native dependencies](#option-2-clojure-package-from-source-scala-package-from-jar). Choose this option if you want pre-release features of the Clojure package but don't want to build (compile) native dependencies yourself.
3. [Build everything from source](#option-3-everything-from-source). This option is for developers or advanced users who want cutting-edge features in all parts of the dependency chain.

**Note:** This guide assumes that you are familiar with the basics of creating Clojure projects and managing dependencies. See [here](https://github.com/technomancy/leiningen/blob/stable/doc/TUTORIAL.md) for the official Leiningen tutorial.

### Option 1: Clojure Package from Prebuilt Jar

If you are new to MXNet and just want to try things out, this option is the best way to get started. You will install release versions of MXNet core, MXNet Scala and MXNet Clojure.

For reference, the Clojure MXNet jars can be found on [maven.org](https://search.maven.org/search?q=clojure%20mxnet).

#### Installing additional dependencies

Depending on your operating system, you will need a couple of packages that are not distributed through Maven:

- [OpenCV](https://opencv.org/) version 3.4
- [OpenBLAS](https://www.openblas.net/)
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
- Edit your `project.clj` and add one of the following entries to `:dependencies`, based on your system and the compute device you want to use:


  - `[org.apache.mxnet.contrib.clojure/clojure-mxnet-linux-cpu <latest-version>]`
  - `[org.apache.mxnet.contrib.clojure/clojure-mxnet-linux-gpu <latest-version>]`
  - `[org.apache.mxnet.contrib.clojure/clojure-mxnet-osx-cpu <latest-version>]`

You can find the latest version out on [maven central- clojure-mxnet latest](https://search.maven.org/search?q=clojure-mxnet)

After making this change and running `lein deps`, you should be able to run example code like this [NDArray Tutorial](https://github.com/apache/incubator-mxnet/blob/master/contrib/clojure-package/examples/tutorial/src/tutorial/ndarray.clj).

### Option 2: Clojure package from Source, Scala Package from Jar

With this option, you will install a Git revision of the Clojure package source and a [Scala package jar from Maven](https://search.maven.org/search?q=g:org.apache.mxnet) with native dependencies baked in.

- Install additional dependencies as described in [the corresponding section for Option 1](#installing-additional-dependencies),

- Recursively clone the MXNet repository and checkout the desired version, (example 1.3.1). You should use the latest [version](https://search.maven.org/search?q=clojure-mxnet)), and a clone into the `~/mxnet` directory:

  ```bash
  git clone --recursive https://github.com/apache/incubator-mxnet.git ~/mxnet
  cd ~/mxnet
  git tag --list  # Find the tag that matches the Scala package version

  git checkout tags/<version> -b my_mxnet
  git submodule update --init --recursive
  cd contrib/clojure
  ```

- Edit `project.clj` to include the desired Scala jar from Maven:


      [org.apache.mxnet/mxnet-full_2.11-linux-x86_64-cpu <latest-version>]

- Run `lein test`. All the tests should run without error.
- At this point you can run `lein install` to build and install the Clojure jar locally.

To run examples, you can now use `lein run` in any of the example directories, e.g., `examples/imclassification`. You can also specify the compute device, e.g., `lein run :cpu 2` (for 2 CPUs) or `lein run :gpu` (for 1 GPU).

#### Experimental: Using Scala Snapshot Jars
**Note:** Instead of a release tag, you can also use a development version of the Clojure package, e.g., Git `master`, together with the prebuilt Scala jar. There is a repo of nightly built snapshots of Scala jars. You can use them in your `project.clj` by adding a repository:

```
["snapshots" {:url "https://repository.apache.org/content/repositories/snapshots"
                              :snapshots true
                              :sign-releases false
                              :checksum :fail
                              :update :always
                              :releases {:checksum :fail :update :always}}]
```

Then you should be able to run with your dependency:

    [org.apache.mxnet/mxnet-full_2.11-osx-x86_64-cpu "latest-version-SNAPSHOT"]


In that case, however, breakage can happen at any point, for instance when the Scala development version adds, changes or removes an interface and the Clojure development version moves along. If you really need the most recent version, you should consider [installation option 3](#option-3-everything-from-source).

### Option 3: Everything from Source

With this option, you will compile the core MXNet C++ package and jars for both Scala and Clojure language bindings from source. If you intend to make changes to the code in any of the parts, or if you simply want the latest and greatest features, this choice is for you.

The first step is to recursively clone the MXNet repository and checkout the desired version, (example 1.3.1). You should use the latest [version](https://search.maven.org/search?q=clojure-mxnet)), and clone into the `~/mxnet` directory:

  ```bash
  git clone --recursive https://github.com/apache/incubator-mxnet.git ~/mxnet
  cd ~/mxnet
  git checkout tags/version -b my_mxnet  # this is optional
  git submodule update --init --recursive
  ```

If you have previous builds and other unwanted files lying around in the working directory and would like to clean up, [here](https://gist.github.com/nicktoumpelis/11214362) is a useful script for that task. However, be aware that this recipe will remove any untracked files and reset uncommitted changes in the working directory.

#### Building the core library

Detailed instructions for building MXNet core from source can be found [in the MXNet installation documentation](https://mxnet.incubator.apache.org/install/index.html). The relevant sections are:

- For Ubuntu Linux: [CUDA Dependencies](https://mxnet.incubator.apache.org/install/ubuntu_setup.html#cuda-dependencies) and [Building MXNet from Source](https://mxnet.incubator.apache.org/install/ubuntu_setup.html#build-mxnet-from-source)
- For Mac OSX: [Build the Shared Library](https://mxnet.incubator.apache.org/install/osx_setup.html#build-the-shared-library)

In particular, ignore all of the language-interface-specific sections.

The outcome of this step will be a shared library `lib/libmxnet.so` that is used in the next step.

#### Building the Scala jar

- Ensure you have JDK 8 on your system. Later versions may produce cryptic build errors mentioning `scala.reflect.internal.MissingRequirementError`. 
- Build and install the Scala package in your local Maven directory using the following commands:

  ```bash
  cd scala-package
  mvn install
  ```

#### Building the Clojure jar
 
- Enter the `contrib/clojure` directory and edit the `project.clj` file. Add the Scala jar that was just created and installed, e.g., `[org.apache.mxnet/mxnet-full_2.11-osx-x86_64-cpu "latest-version-SNAPSHOT"]`, to the `:dependencies`.
- Run `lein test`. All the tests should run without an error.
- Run `lein install` to build and install the Clojure jar locally.

To run examples, you can now use `lein run` in any of the example directories, e.g., `examples/imclassification`. You can also specify the compute device, e.g., `lein run :cpu 2` (for 2 CPUs) or `lein run :gpu` (for 1 GPU).

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
- Apache Slack #mxnet and #mxnet-scala channel. To join this slack send an email to dev@mxnet.apache.org.
- Create an Issue on [https://github.com/apache/incubator-mxnet/issues](https://github.com/apache/incubator-mxnet/issues).


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

The release process for deploying the Clojure jars is on the [Apache MXNet developer wiki](https://cwiki.apache.org/confluence/display/MXNET/Clojure+Release+Process).


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
