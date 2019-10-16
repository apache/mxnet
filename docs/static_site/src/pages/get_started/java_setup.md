---
layout: page
title: Java Setup
action: Get Started
action_url: /get_started
permalink: /get_started/java_setup
---
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

**Note:** If you use IntelliJ or a similar IDE, you may want to follow the [MXNet-Java on IntelliJ tutorial]({{'/api/java/docs/tutorials/mxnet_java_on_intellij'|relative_url}}) instead of these instructions.

<hr>

## Maven

### Setup Instructions

**Step 1.** Install dependencies:

**macOS Steps**

```bash
brew update
brew tap caskroom/versions
brew cask install java8
brew install maven
```

**Ubuntu Steps**

Please run the following lines:

```bash
sudo apt-get install openjdk-8-jdk maven
```

**Step 2.** Run the demo MXNet-Java project.

Go to the [MXNet-Java demo project's README](https://github.com/apache/incubator-mxnet/tree/master/scala-package/mxnet-demo/java-demo) and follow the directions to test the MXNet-Java package installation.

#### Maven Repository

MXNet-Java can be easily included in your Maven managed project. The Java packages are currently available on Maven. Add the dependency which corresponds to your platform to the `dependencies` tag :

**Linux CPU**
```html
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.11-linux-x86_64-cpu</artifactId>
  <version>1.4.0</version>
</dependency>
```

**Linux GPU**
```html
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.11-linux-x86_64-gpu</artifactId>
  <version>1.4.0</version>
</dependency>
```

**macOS CPU**
```html
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.11-osx-x86_64-cpu</artifactId>
  <version>1.4.0</version>
</dependency>
```

The official Java Packages have been released as part of MXNet 1.4 and are available on the [MXNet Maven package repository](https://search.maven.org/#search%7Cga%7C1%7Cg%3A%22org.apache.mxnet%22).
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
|macOS | [Shared Library for macOS](osx_setup.html#build-the-shared-library) | [Scala Package for macOS](osx_setup.html#install-the-mxnet-package-for-scala) |
| Ubuntu | [Shared Library for Ubuntu](ubuntu_setup.html#installing-mxnet-on-ubuntu) | [Scala Package for Ubuntu](ubuntu_setup.html#install-the-mxnet-package-for-scala) |
| Windows | <a class="github-button" href="https://github.com/apache/incubator-mxnet/issues/10549" data-size="large" data-show-count="true" aria-label="Issue apache/incubator-mxnet on GitHub"> | <a class="github-button" href="https://github.com/apache/incubator-mxnet/issues/10549" data-size="large" data-show-count="true" aria-label="Issue apache/incubator-mxnet on GitHub">Call for Contribution</a> |


#### Build Java from an Existing MXNet Installation
If you have already built MXNet **from source** and are looking to setup Java from that point, you may simply run the following from the MXNet `scala-package` folder:

```
mvn install
```
This will install both the Java Inference API and the required MXNet-Scala package.
<hr>

## Documentation

Javadocs are generated as part of the docs build pipeline. You can find them published in the [Java API]({{'/api/java'|relative_url}}) section of the website or by going to the [scaladocs output]({{'/api/scala/docs/api/#org.apache.mxnet.package'|relative_url}}) directly.

To build the docs yourself, follow the [developer build docs instructions](https://github.com/apache/incubator-mxnet/tree/master/docs/README.md).

<hr>

## Resources

* [Java API]({{'/api/java'|relative_url}})
* [javadocs]({{'/api/scala/docs/api/#org.apache.mxnet.package'|relative_url}})
* [MXNet-Java Tutorials]({{'/api/java/docs/tutorials'|relative_url}})
