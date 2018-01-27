# Run MXNet Scala Examples Using the IntelliJ IDE

This tutorial guides you through setting up a Scala project in the IntelliJ IDE and shows how to use an MXNet package from your application.

## Prerequisites:
To use this tutorial, you need:

- [Maven 3](https://maven.apache.org/install.html).
- [Scala 2.11.8](https://www.scala-lang.org/download/2.11.8.html).
- MXNet. See the instructions for your operating system in [Setup and Installation](http://mxnet.io/install/index.html).
- The MXNet package for Scala. For installation instructions, see [this procedure](http://mxnet.io/get_started/osx_setup.html#install-the-mxnet-package-for-scala).
- [IntelliJ IDE](https://www.jetbrains.com/idea/).

## Set Up Your Project

- Install the plugin for IntelliJ IDE by following these steps:
 On **Menu**, choose **Preferences**, choose **Plugins**, type **Scala**, and then choose **Install**.

- Follow the instructions for [Scala plugin setup for IDE](https://www.jetbrains.com/help/idea/2016.3/scala.html).

- When you build the MXNet package with Scala, a JAR file called `mxnet-full_${scala.binary.version}-${platform}` is generated in `native/<your-architecture>/target` directory. You need this file to create an example package that has a dependency on MXNet.

- Specify project dependencies in pom.xml:

```HTML
    <dependencies>
      <dependency>
        <groupId>ml.dmlc.mxnet</groupId>
        <artifactId>mxnet-full_${scala.binary.version}-${platform}</artifactId>
        <version>0.1.1</version>
        <scope>system</scope>
        <systemPath>`MXNet-Scala-jar-path`</systemPath>
      </dependency>
      <dependency>
        <groupId>args4j</groupId>
        <artifactId>args4j</artifactId>
        <version>2.0.29</version>
      </dependency>
    </dependencies>
```

Be sure to change the system path of MXNet-Scala-jar, which is in the `native/<your-architecture>/target` directory.

- Choose the example project, choose Maven, and then reimport. These steps add all of the dependencies in pom.xml as external libraries in your project.

- To build the project, choose Menu, choose Build, and then choose Rebuild Project. If errors are reported in the IDE, address them.

- You can also compile the project by using the following command at the command line.

```bash
    cd mxnet-scala-example
    mvn clean package
```

- This also generates a file called mxnet-scala-example-0.1-SNAPSHOT.jar for your application.

## Next Steps
For more information about MXNet Scala resources, see the following:

* [Scala API](http://mxnet.io/api/scala/)
* [More Scala Examples](https://github.com/dmlc/mxnet/tree/master/scala-package/examples/)
* [MXNet tutorials index](http://mxnet.io/tutorials/index.html)
