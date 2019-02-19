<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Setup the MXNet Package for Java

The following instructions are provided for macOS and Ubuntu. Windows is not yet available.

**Note:** If you use IntelliJ or a similar IDE, you may want to follow the [MXNet-Java on IntelliJ tutorial](../tutorials/java/mxnet_java_on_intellij.html) instead of these instructions.

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

These scripts will install Maven and its dependencies. You will be running the Scala scripts because the MXNet-Java project has a dependency on the MXNet-Scala project.

```bash
wget https://raw.githubusercontent.com/apache/incubator-mxnet/master/ci/docker/install/ubuntu_core.sh
wget https://raw.githubusercontent.com/apache/incubator-mxnet/master/ci/docker/install/ubuntu_scala.sh
chmod +x ubuntu_core.sh
chmod +x ubuntu_scala.sh
sudo ./ubuntu_core.sh
sudo ./ubuntu_scala.sh
```

**Step 2.** Run the demo MXNet-Java project.

Go to the [MXNet-Java demo project's README](https://github.com/apache/incubator-mxnet/tree/master/scala-package/mxnet-demo/java-demo) and follow the directions to test the MXNet-Java package installation.

#### Maven Repository

MXNet-Java can be easily included in your Maven managed project. The Java packages are currently available as nightly builds on Maven. Add the following Maven repository to your `pom.xml` to fetch the Java packages :

```html
<repositories>
    <repository>
      <id>Apache Snapshot</id>
      <url>https://repository.apache.org/content/groups/snapshots</url>
    </repository>
</repositories>
```

Also, add the dependency which corresponds to your platform to the `dependencies` tag :

**Linux CPU**
```html
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.11-linux-x86_64-cpu</artifactId>
  <version>1.4.0-SNAPSHOT</version>
</dependency>
```

**Linux GPU**
```html
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.11-linux-x86_64-gpu</artifactId>
  <version>1.4.0-SNAPSHOT</version>
</dependency>
```

**macOS CPU**
```html
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.11-osx-x86_64-cpu</artifactId>
  <version>1.4.0-SNAPSHOT</version>
</dependency>
```


The official Java Packages will be released with the release of MXNet 1.4 and will be available on  [MXNet Maven package repository](https://search.maven.org/#search%7Cga%7C1%7Cg%3A%22org.apache.mxnet%22).
<hr>

### Eclipse IDE Support
You can convert your existing Maven project to a project that can run in Eclipse by:
```
mvn eclipse:eclipse
```
This can be done once you have your maven project properly configured.

## Source

The previously mentioned setup with Maven is recommended. Otherwise, the following instructions for macOS and Ubuntu are provided for reference only:

**If you have already built mxnet from source using `cmake`, run `make clean` and then follow the appropriate guide below***

| OS | Step 1 | Step 2 |
|---|---|---|
|macOS | [Shared Library for macOS](../install/osx_setup.html#build-the-shared-library) | [Scala Package for macOS](http://mxnet.incubator.apache.org/install/osx_setup.html#install-the-mxnet-package-for-scala) |
| Ubuntu | [Shared Library for Ubuntu](../install/ubuntu_setup.html#installing-mxnet-on-ubuntu) | [Scala Package for Ubuntu](http://mxnet.incubator.apache.org/install/ubuntu_setup.html#install-the-mxnet-package-for-scala) |
| Windows | <a class="github-button" href="https://github.com/apache/incubator-mxnet/issues/10549" data-size="large" data-show-count="true" aria-label="Issue apache/incubator-mxnet on GitHub"> | <a class="github-button" href="https://github.com/apache/incubator-mxnet/issues/10549" data-size="large" data-show-count="true" aria-label="Issue apache/incubator-mxnet on GitHub">Call for Contribution</a> |


#### Build Java from an Existing MXNet Installation
If you have already built MXNet **from source** and are looking to setup Java from that point, you may simply run the following from the MXNet `scala-package` folder:

```
mvn install
```
This will install both the Java Inference API and the required MXNet-Scala package. 
<hr>

## Documentation

Javadocs are generated as part of the docs build pipeline. You can find them published in the [Java API](../api/java/index.html) section of the website or by going to the [scaladocs output](https://mxnet.incubator.apache.org/api/scala/docs/index.html#org.apache.mxnet.package) directly.

To build the docs yourself, follow the [developer build docs instructions](https://github.com/apache/incubator-mxnet/tree/master/docs/build_version_doc#developer-instructions).

<hr>

## Resources

* [Java API](../api/java/index.html)
* [javadocs](../api/java/docs/index.html#org.apache.mxnet.package)
* [MXNet-Java Tutorials](../../tutorials/index.html#java-tutorials)
