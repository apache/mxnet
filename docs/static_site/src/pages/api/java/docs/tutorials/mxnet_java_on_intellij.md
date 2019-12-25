---
layout: page_api
title: Java with IntelliJ
permalink: /api/java/docs/tutorials/mxnet_java_on_intellij
is_tutorial: true
tag: java
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


# Run MXNet Java Examples Using the IntelliJ IDE (macOS)

This tutorial guides you through setting up a simple Java project in IntelliJ IDE on macOS and demonstrates usage of the MXNet Java APIs.

## Prerequisites:
To use this tutorial you need the following pre-requisites:

- [Java 8 JDK](http://www.oracle.com/technetwork/java/javase/downloads/index.html)
- [Maven](https://maven.apache.org/install.html)
- [OpenCV](https://opencv.org/)
- [IntelliJ IDEA](https://www.jetbrains.com/idea/) (One can download the community edition from [here](https://www.jetbrains.com/idea/download))

### MacOS Prerequisites

Run the following commands to install the prerequisites on MacOS.
```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew update
brew tap caskroom/versions
brew cask install java8
brew install maven
brew install opencv
```

### Ubuntu Prerequisites

Run the following commands to install the prerequisites on Ubuntu.

```
sudo apt-get install openjdk-8-jdk maven
```


## Set Up Your Project

**Step 1.** Install and setup [IntelliJ IDEA](https://www.jetbrains.com/idea/)

**Step 2.** Create a new Project:

![intellij welcome](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/scala/intellij-welcome.png)

From the IntelliJ welcome screen, select "Create New Project".

Choose the Maven project type.

Select the checkbox for `Create from archetype`, then choose `org.apache.maven.archetypes:maven-archetype-quickstart` from the list below. More on this can be found on a Maven tutorial : [Maven in 5 Minutes](https://maven.apache.org/guides/getting-started/maven-in-five-minutes.html).

![maven project type - archetype](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tutorials/java/project-archetype.png)

click `Next`.

![project metadata](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tutorials/java/intellij-project-metadata.png)

Set the project's metadata. For this tutorial, use the following:

**GroupId**
```
mxnet
```
**ArtifactId**
```
javaMXNet
```
**Version**
```
1.0-SNAPSHOT
```

![project properties](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tutorials/java/intellij-project-properties.png)

Review the project's properties. The settings can be left as their default.

![project location](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tutorials/java/intellij-project-location.png)

Set the project's location. The rest of the settings can be left as their default.

![project 1](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tutorials/java/intellij-project-pom.png)

After clicking Finish, you will be presented with the project's first view.
The project's `pom.xml` will be open for editing.

**Step 3.** The Java packages are currently available on Maven. Add the following under the `dependencies` tag :

```html
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.11-osx-x86_64-cpu</artifactId>
  <version>1.4.0</version>
</dependency>
```
The official Java Packages have been released as part of MXNet 1.4 and are available on the [MXNet Maven package repository](https://search.maven.org/#search%7Cga%7C1%7Cg%3A%22org.apache.mxnet%22).

Note :
- Change the osx-x86_64 to linux-x86_64 if your platform is linux.
- Change cpu into gpu if you have a gpu backed machine and want to use gpu.


**Step 4.** Import dependencies with Maven:

  - Note the prompt in the lower right corner that states "Maven projects need to be imported". If this is not visible, click on the little greed balloon that appears in the lower right corner.

![import_dependencies](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tutorials/java/project-import-changes.png)

Click "Import Changes" in this prompt.

**Step 5.** Build the project:
- To build the project, from the menu choose Build, and then choose Build Project.

**Step 6.** Navigate to the App.java class in the project and paste the code in `main` method from HelloWorld.java from [Java Demo project](https://github.com/apache/incubator-mxnet/tree/master/scala-package/mxnet-demo/java-demo/src/main/java/mxnet/HelloWorld.java) on MXNet repository, overwriting the original hello world code.
You can also grab the entire [Java Demo project](https://github.com/apache/incubator-mxnet/tree/master/scala-package/mxnet-demo/java-demo) and run it by following the instructions on the [README](https://github.com/apache/incubator-mxnet/blob/master/scala-package/mxnet-demo/java-demo/README.md).

**Step 7.** Now run the App.java.

The result should be something similar to this:

```
Hello World!
(1,2)
Process finished with exit code 0
```

### Troubleshooting

If you get an error, check the dependencies at the beginning of this tutorial. For example, you might see the following in the middle of the error messages, where `x.x` would the version it's looking for.

```
...
Library not loaded: /usr/local/opt/opencv/lib/libopencv_calib3d.x.x.dylib
...
```

This can be resolved be installing OpenCV.

### Command Line Build Option

- You can also compile the project by using the following command at the command line. Change directories to this project's root folder then run the following:

```bash
mvn clean install dependency:copy-dependencies
```
If the command succeeds, you should see a lot of info and some warning messages, followed by:

```bash
[INFO] ------------------------------------------------------------------------
[INFO] BUILD SUCCESS
[INFO] ------------------------------------------------------------------------
[INFO] Total time: 3.475 s
[INFO] Finished at: 2018-11-08T05:06:31-08:00
[INFO] ------------------------------------------------------------------------
```
The build generates a new jar file in the `target` folder called `javaMXNet-1.0-SNAPSHOT.jar`.

To run the App.java use the following command from the project's root folder and you should see the same output as we got when the project was run from IntelliJ.
```bash
java -cp "target/javaMXNet-1.0-SNAPSHOT.jar:target/dependency/*" mxnet.App
```

## Next Steps
For more information about MXNet Java resources, see the following:

* [Java Inference API]({{'/api/java'|relative_url}})
* [Java Inference Examples](https://github.com/apache/incubator-mxnet/tree/master/scala-package/examples/src/main/java/org/apache/mxnetexamples/javaapi/infer)
* [MXNet Tutorials Index]({{'/api'|relative_url}})
