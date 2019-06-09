MXNet Package for Scala/Java
=====

The MXNet Scala/Java Package brings flexible and efficient GPU/CPU computing and state-of-art deep learning to JVM.

- It enables you to write seamless tensor/matrix computation with multiple GPUs
  in Scala, Java and other languages built on JVM.
- It also enables you to construct and customize the state-of-art deep learning models in JVM languages,
  and apply them to tasks such as image classification and data science challenges.
- The Scala/Java Inferece APIs provides an easy out of the box solution for loading pre-trained MXNet models and running inference on them.
  
Pre-Built Maven Packages
------------

### Stable ###

The MXNet Scala/Java packages can be easily included in your Maven managed project.
The stable jar files for the packages are available on the [MXNet Maven Package Repository](https://search.maven.org/search?q=g:org.apache.mxnet)
Currently we provide packages for Linux (Ubuntu 16.04) (CPU and GPU) and macOS (CPU only). Stable packages for Windows and CentOS will come soon. For now, if you have a CentOS machine, follow the ```Build From Source``` section below. 

To add MXNet Scala/Java package to your project, add the dependency as shown below corresponding to your platform, under the ```dependencies``` tag in your project's ```pom.xml``` :

**Linux GPU**

<a href="https://mvnrepository.com/artifact/org.apache.mxnet/mxnet-full_2.11-linux-x86_64-gpu"><img src="https://img.shields.io/badge/org.apache.mxnet-linux gpu-green.svg" alt="maven badge"/></a>

```HTML
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.11-linux-x86_64-gpu</artifactId>
  <version>[1.3.1,)</version>
</dependency>
```

**Linux CPU**

<a href="https://mvnrepository.com/artifact/org.apache.mxnet/mxnet-full_2.11-linux-x86_64-cpu"><img src="https://img.shields.io/badge/org.apache.mxnet-linux cpu-green.svg" alt="maven badge"/></a>

```HTML
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.11-linux-x86_64-cpu</artifactId>
  <version>[1.3.1,)</version>
</dependency>
```

**macOS CPU**

<a href="https://mvnrepository.com/artifact/org.apache.mxnet/mxnet-full_2.11-osx-x86_64-cpu"><img src="https://img.shields.io/badge/org.apache.mxnet-macOS cpu-green.svg" alt="maven badge"/></a>

```HTML
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.11-osx-x86_64-cpu</artifactId>
  <version>[1.3.1,)</version>
</dependency>
```

**Note:** ```<version>[1.3.1,)<\version>``` indicates that we will fetch packages with version 1.3.1 or higher. This will always ensure that the pom.xml is able to fetch the latest and greatest jar files from Maven.  

### Nightly ###

Apart from these, the nightly builds representing the bleeding edge development  on Scala/Java packages are also available on the [MXNet Maven Nexus Package Repository](https://repository.apache.org/#nexus-search;gav~org.apache.mxnet~~~~). 
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
  <version>[1.5.0,)</version>
</dependency>
```

**Linux CPU**

<a href="https://repository.apache.org/#nexus-search;gav~org.apache.mxnet~mxnet-full_2.11-osx-x86_64-cpu~~~"><img src="https://img.shields.io/badge/org.apache.mxnet-linux cpu-green.svg" alt="maven badge"/></a>

```HTML
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.11-linux-x86_64-cpu</artifactId>
  <version>[1.5.0,)</version>
</dependency>
```

**macOS CPU**

<a href="https://mvnrepository.com/artifact/org.apache.mxnet/mxnet-full_2.11-osx-x86_64-cpu"><img src="https://img.shields.io/badge/org.apache.mxnet-macOS cpu-green.svg" alt="maven badge"/></a>
```HTML
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.11-osx-x86_64-cpu</artifactId>
  <version>[1.5.0,)</version>
</dependency>
```

**Note:** ```<version>[1.5.0,)<\version>``` indicates that we will fetch packages with version 1.5.0 or higher. This will always ensure that the pom.xml is able to fetch the latest and greatest jar files from Maven Snapshot repository.

Build From Source
------------

Checkout the [Installation Guide](http://mxnet.incubator.apache.org/install/index.html) contains instructions to install mxnet package and build it from source.
If you have built MXNet from source and are looking to setup Scala from that point, you may simply run the following from the MXNet source root:

```bash
make scalapkg
```

You can also run the unit tests and integration tests on the Scala Package by :

```bash
make scalaunittest
make scalaintegrationtest
```

Or run a subset of unit tests, for e.g.,

```bash
make SCALA_TEST_ARGS=-Dsuites=org.apache.mxnet.NDArraySuite scalaunittest
```

If everything goes well, you will find jars for `assembly`, `core` and `example` modules.
Also it produces the native library in `native/{your-architecture}/target`, which you can use to cooperate with the `core` module.

Examples & Usage
-------
- To set up the Scala Project using IntelliJ IDE on macOS follow the instructions [here](https://mxnet.incubator.apache.org/tutorials/scala/mxnet_scala_on_intellij.html).
- Several examples on using the Scala APIs are provided in the [Scala Examples Folder](https://github.com/apache/incubator-mxnet/tree/master/scala-package/examples/)

Scala Training APIs
-------
- Module API :
[The Module API](https://mxnet.incubator.apache.org/api/scala/module.html) provides an intermediate and high-level interface for performing computation with neural networks in MXNet. Modules provide high-level APIs for training, predicting, and evaluating.

- KVStore API : 
To run training over multiple GPUs and multiple hosts, one can use the [KVStore API](https://mxnet.incubator.apache.org/api/scala/kvstore.html).

- IO/Data Loading : 
MXNet Scala provides APIs for preparing data to feed as an input to models. Check out [Data Loading API](https://mxnet.incubator.apache.org/api/scala/io.html) for more info.
 
Other available Scala APIs for training can be found [here](https://mxnet.incubator.apache.org/api/scala/index.html).  
 

Scala Inference APIs
-------
The [Scala Inference APIs](https://mxnet.incubator.apache.org/api/scala/infer.html) provide an easy, out of the box solution to load a pre-trained MXNet model and run inference on it. The Inference APIs are present in the [Infer Package](https://github.com/apache/incubator-mxnet/tree/master/scala-package/infer) under the MXNet Scala Package repository, while the documentation for the Infer API is available [here](https://mxnet.incubator.apache.org/api/scala/docs/index.html#org.apache.mxnet.infer.package).  

Java Inference APIs
-------
The [Java Inference APIs](http://mxnet.incubator.apache.org/api/java/index.html) also provide an easy, out of the box solution to load a pre-trained MXNet model and run inference on it. The Inference APIs are present in the [Infer Package](https://github.com/apache/incubator-mxnet/tree/master/scala-package/infer/src/main/scala/org/apache/mxnet/infer/javaapi) under the MXNet Scala Package repository, while the documentation for the Infer API is available [here](https://mxnet.incubator.apache.org/api/java/docs/index.html#org.apache.mxnet.infer.package).
More APIs will be added to the Java Inference APIs soon.

JVM Memory Management
-------
The Scala/Java APIs also provide an automated resource management system, thus making it easy to manage the native memory footprint without any degradation in performance.
More details about JVM Memory Management are available [here](https://github.com/apache/incubator-mxnet/blob/master/scala-package/memory-management.md).

License
-------
MXNet Scala Package is licensed under [Apache-2](https://github.com/apache/incubator-mxnet/blob/master/scala-package/LICENSE) license.
