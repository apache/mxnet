# MXNet-Scala Tutorials

## Installation & Setup

Using MXNet-Scala is easiest with Maven. Refer to the following tutorial to see how you can use Maven with IntelliJ.

* [Setup Scala with MXNet and Create a MXNet-Scala Project with IntelliJ (macOS)](mxnet_scala_on_intellij.md)

You can also build the Scala package yourself.

### Build the MXNet Shared Library and Scala Package

This depends on your operating system. Instructions for macOS, Ubuntu, and Windows are provided:

| OS | Step 1 | Step 2 |
|---|---|---|
|macOS | [Shared Library for macOS](http://mxnet.incubator.apache.org/install/osx_setup.html#build-the-shared-library) | [Scala Package for macOS](http://mxnet.incubator.apache.org/install/osx_setup.html#install-the-mxnet-package-for-scala) |
| Ubuntu | [Shared Library for Ubuntu](http://mxnet.incubator.apache.org/install/ubuntu_setup.html#installing-mxnet-on-ubuntu) | [Scala Package for Ubuntu](http://mxnet.incubator.apache.org/install/ubuntu_setup.html#install-the-mxnet-package-for-scala) |
| Windows | [Shared Library for Windows](http://mxnet.incubator.apache.org/install/windows_setup.html#build-the-shared-library) | [Scala Package for Windows](http://mxnet.incubator.apache.org/install/windows_setup.html#installing-the-mxnet-package-for-scala) |


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
