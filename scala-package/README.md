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

MXNet Package for Scala/Java
=====

The MXNet Scala/Java Package brings flexible and efficient GPU/CPU computing and state-of-art deep learning to the JVM.

- It enables you to write seamless tensor/matrix computation with multiple GPUs
  in Scala, Java and other JVM languages.
- It also enables you to construct and customize the state-of-art deep learning models in JVM languages,
  and apply them to tasks such as image classification and data science challenges.
- The Scala/Java _Inference API_ provides an easy out of the box solution for performing inference tasks using pre-trained MXNet models.

Pre-Built Maven Packages
------------------------

### Stable ###

The MXNet Scala/Java packages can be easily included in your Maven managed project.
The stable jar files for the packages are available on the [MXNet Maven Package Repository](https://search.maven.org/search?q=g:org.apache.mxnet).
Currently we provide packages for Linux (Ubuntu 16.04) (CPU and GPU) and macOS (CPU only). Stable packages for Windows and CentOS will come soon. For now, if you have a CentOS machine, follow the ```Build From Source``` section below. 

To add the MXNet Scala/Java packages to your project, add the dependency as shown below corresponding to your platform, under the ```dependencies``` tag in your project's ```pom.xml``` :

**Linux GPU**

<a href="https://mvnrepository.com/artifact/org.apache.mxnet/mxnet-full_2.11-linux-x86_64-gpu"><img src="https://img.shields.io/badge/org.apache.mxnet-linux gpu-green.svg" alt="maven badge"/></a>

```HTML
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.11-linux-x86_64-gpu</artifactId>
  <version>[1.4.0,)</version>
</dependency>
```

**Linux CPU**

<a href="https://mvnrepository.com/artifact/org.apache.mxnet/mxnet-full_2.11-linux-x86_64-cpu"><img src="https://img.shields.io/badge/org.apache.mxnet-linux cpu-green.svg" alt="maven badge"/></a>

```HTML
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.11-linux-x86_64-cpu</artifactId>
  <version>[1.4.0,)</version>
</dependency>
```

**macOS CPU**

<a href="https://mvnrepository.com/artifact/org.apache.mxnet/mxnet-full_2.11-osx-x86_64-cpu"><img src="https://img.shields.io/badge/org.apache.mxnet-macOS cpu-green.svg" alt="maven badge"/></a>

```HTML
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.11-osx-x86_64-cpu</artifactId>
  <version>[1.4.0,)</version>
</dependency>
```

**Note:** ```<version>[1.4.0,)<\version>``` indicates that we will fetch packages with version 1.4.0 or higher. This will always ensure that the pom.xml is able to fetch the latest and greatest jar files from Maven.  

### Nightly ###

Apart from these, the nightly builds representing the bleeding edge development on Scala/Java packages are also available on the [MXNet Maven Nexus Package Repository](https://repository.apache.org/#nexus-search;gav~org.apache.mxnet~~~~). 
Currently we provide nightly packages for Linux (CPU and GPU) and MacOS (CPU only). The Linux nightly jar files also work on CentOS. Nightly packages for Windows will come soon.

Add the following ```repository``` to your project's ```pom.xml``` file : 

````html
<repositories>
    <repository>
      <id>Apache Snapshot</id>
      <url>https://repository.apache.org/content/groups/snapshots</url>
    </repository>
</repositories>
````

Also, add the dependency which corresponds to your platform to the ```dependencies``` tag :

**Linux GPU**

<a href="https://repository.apache.org/#nexus-search;gav~org.apache.mxnet~mxnet-full_2.11-linux-x86_64-gpu~~~"><img src="https://img.shields.io/badge/org.apache.mxnet-linux gpu-green.svg" alt="maven badge"/></a>

```HTML
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.11-linux-x86_64-gpu</artifactId>
  <version>[1.6.0-SNAPSHOT,)</version>
</dependency>
```

**Linux CPU**

<a href="https://repository.apache.org/#nexus-search;gav~org.apache.mxnet~mxnet-full_2.11-osx-x86_64-cpu~~~"><img src="https://img.shields.io/badge/org.apache.mxnet-linux cpu-green.svg" alt="maven badge"/></a>

```HTML
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.11-linux-x86_64-cpu</artifactId>
  <version>[1.6.0-SNAPSHOT,)</version>
</dependency>
```

**macOS CPU**

<a href="https://mvnrepository.com/artifact/org.apache.mxnet/mxnet-full_2.11-osx-x86_64-cpu"><img src="https://img.shields.io/badge/org.apache.mxnet-macOS cpu-green.svg" alt="maven badge"/></a>
```HTML
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.11-osx-x86_64-cpu</artifactId>
  <version>[1.6.0-SNAPSHOT,)</version>
</dependency>
```

**Note:** ```<version>[1.6.0-SNAPSHOT,)</version>``` indicates that we will fetch packages with version 1.6.0 or higher. This will always ensure that the pom.xml is able to fetch the latest and greatest jar files from Maven Snapshot repository.

Build From Source
-----------------

The [Installation Guide](https://mxnet.apache.org/get_started) contains instructions to install mxnet or build it from source. The Scala/Java package is built from source using Maven. The maven build assumes you already have a ``lib/libmxnet.so`` file.
If you have built MXNet from source and are looking to set up Scala\Java from that point, you may simply run the following from the MXNet source root, the build will detect your platform (OSX/Linux) and libmxnet.so flavor (CPU/GPU):

```bash
cd scala-package
mvn install
```

You can also run the unit tests and integration tests on the Scala Package by :

```bash
cd scala-package
mvn integration-test -DskipTests=false
```

Or run a subset of unit tests, for e.g.,

```bash
cd scala-package
mvn -Dsuites=org.apache.mxnet.NDArraySuite integration-test
```

If everything goes well, you will find jars for `assembly`, `core` and `example` modules.
Also it produces the native library in `native/target`, which you can use in conjunction with the `core` module.

Deploy to repository
--------------------

By default, `maven deploy` will deploy artifacts to local file system, you can find them in the ``scala-package/deploy/target/repo`` folder.

For nightly builds (typically done by CI), a snapshot build will be uploaded to an apache snapshot repository with the following command:

```bash
cd scala-package
mvn deploy -Pnightly
```

Use the following command when performing a release (pushes artifacts to an apache staging repository):

```bash
cd scala-package
mvn deploy -Pstaging
```

Examples & Usage
-------
Assuming you use `mvn install`, you can find the `mxnet-full_scala_version-INTERNAL.jar` e.g. `mxnet-full_2.11-INTERNAL.jar` under the path `incubator-mxnet/scala-package/assembly/target`.

Adding the following configuration in `pom.xml`
```HTML
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.11-INTERNAL</artifactId>
  <version>1.6.0</version>
  <scope>system</scope>
  <systemPath>path_to_jar/mxnet-full_2.11-INTERNAL.jar</systemPath>
</dependency>
```
If you have following error message
```
Error: A JNI error has occurred, please check your installation and try again
Exception in thread "main" java.lang.NoClassDefFoundError: org/apache/mxnet/NDArray
        at java.lang.Class.getDeclaredMethods0(Native Method)
        at java.lang.Class.privateGetDeclaredMethods(Class.java:2701)
        at java.lang.Class.privateGetMethodRecursive(Class.java:3048)
        at java.lang.Class.getMethod0(Class.java:3018)
        at java.lang.Class.getMethod(Class.java:1784)
        at sun.launcher.LauncherHelper.validateMainClass(LauncherHelper.java:544)
        at sun.launcher.LauncherHelper.checkAndLoadMain(LauncherHelper.java:526)
Caused by: java.lang.ClassNotFoundException: org.apache.mxnet.NDArray
        at java.net.URLClassLoader.findClass(URLClassLoader.java:381)
        at java.lang.ClassLoader.loadClass(ClassLoader.java:424)
        at sun.misc.Launcher$AppClassLoader.loadClass(Launcher.java:331)
        at java.lang.ClassLoader.loadClass(ClassLoader.java:357)
```
Please make sure your $CLASSPATH contains `mxnet-full_scala_version-INTERNAL.jar`.

- To set up the Scala Project using IntelliJ IDE on macOS follow the instructions [here](https://mxnet.apache.org/tutorials/scala/mxnet_scala_on_intellij.html).
- Several examples on using the Scala APIs are provided in the [Scala Examples Folder](https://github.com/apache/incubator-mxnet/tree/master/scala-package/examples/)

Scala Training APIs
-------
- Module API :
[The Module API](https://mxnet.apache.org/api/scala/module.html) provides an intermediate and high-level interface for performing computation with neural networks in MXNet. Modules provide high-level APIs for training, predicting, and evaluating.

- KVStore API : 
To run training over multiple GPUs and multiple hosts, one can use the [KVStore API](https://mxnet.apache.org/api/scala/kvstore.html).

- IO/Data Loading : 
MXNet Scala provides APIs for preparing data to feed as an input to models. Check out [Data Loading API](https://mxnet.apache.org/api/scala/io.html) for more info.
 
Other available Scala APIs for training can be found [here](https://mxnet.apache.org/api/scala/index.html).  
 

Scala Inference APIs
-------
The [Scala Inference APIs](https://mxnet.apache.org/api/scala/infer.html) provide an easy, out of the box solution to load a pre-trained MXNet model and run inference on it. The Inference APIs are present in the [Infer Package](https://github.com/apache/incubator-mxnet/tree/master/scala-package/infer) under the MXNet Scala Package repository, while the documentation for the Infer API is available [here](https://mxnet.apache.org/api/scala/docs/index.html#org.apache.mxnet.infer.package).  

Java Inference APIs
-------
The [Java Inference APIs](https://mxnet.apache.org/api/java/index.html) also provide an easy, out of the box solution to load a pre-trained MXNet model and run inference on it. The Inference APIs are present in the [Infer Package](https://github.com/apache/incubator-mxnet/tree/master/scala-package/infer/src/main/scala/org/apache/mxnet/infer/javaapi) under the MXNet Scala Package repository, while the documentation for the Infer API is available [here](https://mxnet.apache.org/api/java/docs/index.html#org.apache.mxnet.infer.package).
More APIs will be added to the Java Inference APIs soon.

JVM Memory Management
-------
The Scala/Java APIs also provide an automated resource management system, thus making it easy to manage the native memory footprint without any degradation in performance.
More details about JVM Memory Management are available [here](https://github.com/apache/incubator-mxnet/blob/master/scala-package/memory-management.md).

License
-------
MXNet Scala Package is licensed under [Apache-2](https://github.com/apache/incubator-mxnet/blob/master/scala-package/LICENSE) license.

MXNet uses some 3rd party softwares. Following 3rd party license files are bundled inside Scala jar file:
* cub/LICENSE.TXT
* mkldnn/external/mklml_mac_2019.0.1.20180928/license.txt
