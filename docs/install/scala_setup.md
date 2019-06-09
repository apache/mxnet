# Setup the MXNet Package for Scala

The following instructions are provided for macOS and Ubuntu. Windows is not yet available.

**Note:** If you use IntelliJ or a similar IDE, you may want to follow the [MXNet-Scala on IntelliJ tutorial](../tutorials/scala/mxnet_scala_on_intellij.html) instead of these instructions.

<hr>

## Maven

### Setup Instructions

**Step 1.** Install dependencies:

**macOS Steps**

```bash
brew update
brew tap caskroom/versions
brew cask install java8
brew install opencv
brew install maven
```

**Ubuntu Steps**

These scripts will install Maven and its dependencies.

```bash
wget https://raw.githubusercontent.com/apache/incubator-mxnet/master/ci/docker/install/ubuntu_core.sh
wget https://raw.githubusercontent.com/apache/incubator-mxnet/master/ci/docker/install/ubuntu_scala.sh
chmod +x ubuntu_core.sh
chmod +x ubuntu_scala.sh
sudo ./ubuntu_core.sh
sudo ./ubuntu_scala.sh
```

**Step 2.** Run the demo MXNet-Scala project.

Go to the [MXNet-Scala demo project's README](https://github.com/apache/incubator-mxnet/tree/master/scala-package/mxnet-demo) and follow the directions to test the MXNet-Scala package installation.

#### Maven Repository

Package information can be found in the Maven Repository:
https://mvnrepository.com/artifact/org.apache.mxnet

**Linux CPU**
```html
<!-- https://mvnrepository.com/artifact/org.apache.mxnet/mxnet-full_2.11-linux-x86_64-cpu -->
<dependency>
    <groupId>org.apache.mxnet</groupId>
    <artifactId>mxnet-full_2.11-linux-x86_64-cpu</artifactId>
</dependency>
```

**Linux GPU**
```html
<!-- https://mvnrepository.com/artifact/org.apache.mxnet/mxnet-full_2.11-linux-x86_64-gpu -->
<dependency>
    <groupId>org.apache.mxnet</groupId>
    <artifactId>mxnet-full_2.11-linux-x86_64-gpu</artifactId>
</dependency>
```

**macOS CPU**
```html
<!-- https://mvnrepository.com/artifact/org.apache.mxnet/mxnet-full_2.11-osx-x86_64-cpu -->
<dependency>
    <groupId>org.apache.mxnet</groupId>
    <artifactId>mxnet-full_2.11-osx-x86_64-cpu</artifactId>
</dependency>
```

**NOTE:** You may specify the version you wish to use by adding the version number to the `dependency` block. For example, to use v1.2.0 you would add `<version>1.2.0</version>`. Otherwise Maven will use the latest version available.

<hr>

## Source

The previously mentioned setup with Maven is recommended. Otherwise, the following instructions for macOS, Ubuntu, and Windows are provided for reference only:

**If you have already built mxnet from source using `cmake`, run `make clean` and then follow the appropriate guide below***

| OS | Step 1 | Step 2 |
|---|---|---|
|macOS | [Shared Library for macOS](http://mxnet.incubator.apache.org/install/osx_setup.html#build-the-shared-library) | [Scala Package for macOS](http://mxnet.incubator.apache.org/install/osx_setup.html#install-the-mxnet-package-for-scala) |
| Ubuntu | [Shared Library for Ubuntu](http://mxnet.incubator.apache.org/install/ubuntu_setup.html#installing-mxnet-on-ubuntu) | [Scala Package for Ubuntu](http://mxnet.incubator.apache.org/install/ubuntu_setup.html#install-the-mxnet-package-for-scala) |
| Windows | <a class="github-button" href="https://github.com/apache/incubator-mxnet/issues/10549" data-size="large" data-show-count="true" aria-label="Issue apache/incubator-mxnet on GitHub"> | <a class="github-button" href="https://github.com/apache/incubator-mxnet/issues/10549" data-size="large" data-show-count="true" aria-label="Issue apache/incubator-mxnet on GitHub">Call for Contribution</a> |


#### Build Scala from an Existing MXNet Installation
If you have already built MXNet **from source** and are looking to setup Scala from that point, you may simply run the following from the MXNet source root:

```
make scalapkg
make scalainstall
```

<hr>

## Documentation

Scaladocs are generated as part of the docs build pipeline. You can find them published in the [Scala API](http://mxnet.incubator.apache.org/api/scala/index.html) section of the website or by going to the [scaladocs output](https://mxnet.incubator.apache.org/api/scala/docs/index.html#org.apache.mxnet.package) directly.

To build the docs yourself, follow the [developer build docs instructions](https://github.com/apache/incubator-mxnet/tree/master/docs/build_version_doc#developer-instructions).

<hr>

## Resources

* [Scala API](http://mxnet.incubator.apache.org/api/scala/index.html)
* [scaladocs](https://mxnet.incubator.apache.org/api/scala/docs/index.html#org.apache.mxnet.package)
* [MXNet-Scala Tutorials](../tutorials/scala)
