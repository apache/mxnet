# Setup the MXNet Package for Scala

The following instructions are provided for macOS and Ubuntu. Windows is not yet available.

If you use IntelliJ or a similar IDE, you may want to follow the [MXNet-Scala on IntelliJ tutorial](../tutorials/scala/mxnet_scala_on_intellij.html) instead.

**Step 1.**: Download the MXNet source.

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

**Step 3.**: Run the example project.

Go to the [example project's README](https://github.com/apache/incubator-mxnet/tree/master/scala-package/mxnet-demo) and follow the directions to test the MXNet-Scala package installation.


### Example MXNet-Scala Dependencies Definition

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
