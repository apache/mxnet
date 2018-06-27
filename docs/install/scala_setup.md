# Setup the MXNet Package for Scala

The following instructions are provided for macOS and Ubuntu. Windows is not yet available.

**Note:** If you use IntelliJ or a similar IDE, you may want to follow the [MXNet-Scala on IntelliJ tutorial](../tutorials/scala/mxnet_scala_on_intellij.html) instead of these instructions.

## Setup Instructions
**Step 1.** Download the MXNet source.

```bash
git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet
cd mxnet
```

**Step 2.** Install dependencies:

**macOS Steps**

```bash
brew update
brew tap caskroom/versions
brew cask install java8
brew install maven
brew install opencv@2
```

**Ubuntu Steps**

```bash
sudo ./ci/docker/install/ubuntu_core.sh
sudo ./ci/docker/install/ubuntu_scala.sh
```

**Step 3.** Run the demo MXNet-Scala project.

Go to the [MXNet-Scala demo project's README](https://github.com/apache/incubator-mxnet/tree/master/scala-package/mxnet-demo) and follow the directions to test the MXNet-Scala package installation.


## Example MXNet-Scala Dependencies Definition

The following is an example excerpt from an MXNet-Scala project's `pom.xml` file.

```
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-core_${scala.binary.version}</artifactId>
  <version>1.3.0-SNAPSHOT</version>
  <scope>provided</scope>
</dependency>
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-infer_${scala.binary.version}</artifactId>
  <version>1.3.0-SNAPSHOT</version>
  <scope>provided</scope>
</dependency>
```

## Build the MXNet Shared Library and Scala Package

The previously mentioned setup with Maven is recommended. Otherwise, the following instructions for macOS, Ubuntu, and Windows are provided for reference only:

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

## Documentation

Scaladocs are generated as part of the docs build pipeline. You can find them published in the [Scala API](http://mxnet.incubator.apache.org/api/scala/index.html) section of the website or by going to the [scaladocs output](https://mxnet.incubator.apache.org/api/scala/docs/index.html#org.apache.mxnet.package) directly.

To build the docs yourself, follow the [developer build docs instructions](https://github.com/apache/incubator-mxnet/tree/master/docs/build_version_doc#developer-instructions).

## Resources

* [Scala API](http://mxnet.incubator.apache.org/api/scala/index.html)
* [scaladocs](https://mxnet.incubator.apache.org/api/scala/docs/index.html#org.apache.mxnet.package)
* [MXNet-Scala Tutorials](../tutorials/scala)
