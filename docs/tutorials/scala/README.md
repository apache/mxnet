# MXNet-Scala Tutorials

## Installation & Setup

Using MXNet-Scala is easiest with Maven. You have a couple of options for setting up that depend on your environment.

**Note:** Windows is not yet supported.

* [MXNet-Scala Setup Guide Using Maven](../install/scala_setup.md)
* [Setup Scala with MXNet and Create a MXNet-Scala Project with IntelliJ](mxnet_scala_on_intellij.md)


### Build the MXNet Shared Library and Scala Package

The [MXNet-Scala Setup Guide Using Maven](../install/scala_setup.md) is recommended. Otherwise, the following instructions for macOS, Ubuntu, and Windows are provided for reference only:

| OS | Step 1 | Step 2 |
|---|---|---|
|macOS | [Shared Library for macOS](http://mxnet.incubator.apache.org/install/osx_setup.html#build-the-shared-library) | [Scala Package for macOS](http://mxnet.incubator.apache.org/install/osx_setup.html#install-the-mxnet-package-for-scala) |
| Ubuntu | [Shared Library for Ubuntu](http://mxnet.incubator.apache.org/install/ubuntu_setup.html#installing-mxnet-on-ubuntu) | [Scala Package for Ubuntu](http://mxnet.incubator.apache.org/install/ubuntu_setup.html#install-the-mxnet-package-for-scala) |
| Windows | [Shared Library for Windows](http://mxnet.incubator.apache.org/install/windows_setup.html#build-the-shared-library) | <a class="github-button" href="https://github.com/apache/incubator-mxnet/issues/10549" data-size="large" data-show-count="true" aria-label="Issue apache/incubator-mxnet on GitHub">Call for Contribution</a> |


### Build Scala from an Existing MXNet Installation
If you have already built MXNet **from source** and are looking to setup Scala from that point, you may simply run the following from the MXNet source root:

```
make scalapkg
make scalainstall
```

## Tutorials

* [MNIST with MXNet-Scala](mnist.md)
* [Character-level Language Model with MXNet-Scala](char_lstm.md)

See the [tutorials](http://mxnet.incubator.apache.org/tutorials/index.html#other-languages-api-tutorials) page on MXNet.io for more.
