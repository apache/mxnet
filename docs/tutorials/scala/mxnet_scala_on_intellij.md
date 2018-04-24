# Run MXNet Scala Examples Using the IntelliJ IDE

This tutorial guides you through setting up a Scala project in the IntelliJ IDE and shows how to use an MXNet package from your application.

## Prerequisites:
To use this tutorial you need the following items, however after this list, installation info for macOS is provided for your benefit:

- [Java 8 JDK](http://www.oracle.com/technetwork/java/javase/downloads/index.html)
- [Maven](https://maven.apache.org/install.html)
- [Scala](https://www.scala-lang.org/download/) - comes with IntelliJ, so you don't need to install it separately
- [MXNet Shared Library and Scala Package](#build-the-mxnet-shared-library-and-scala-package)
- [IntelliJ IDE](https://www.jetbrains.com/idea/)

## Mac Prerequisites Setup

For other operating systems, visit each Prerequisite's website and follow their installations instructions. For macOS, you're in luck:

**Step 1.** Install brew:
```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

**Step 2.** Install Java 8 JDK:
```
brew tap caskroom/versions
brew cask install java8
```

**Step 3.** Install maven:
```
brew update
brew install maven
```

## Build the MXNet Shared Library and Scala Package

This depends on your operating system. Instructions for macOS, Ubuntu, and Windows are provided:


| OS | Step 1 | Step 2 |
|---|---|---|
|macOS | [Shared Library for macOS](http://mxnet.incubator.apache.org/install/osx_setup.html#build-the-shared-library) | [Scala Package for macOS](http://mxnet.incubator.apache.org/install/osx_setup.html#install-the-mxnet-package-for-scala) |
| Ubuntu | [Shared Library for Ubuntu](http://mxnet.incubator.apache.org/install/ubuntu_setup.html#installing-mxnet-on-ubuntu) | [Scala Package for Ubuntu](http://mxnet.incubator.apache.org/install/ubuntu_setup.html#install-the-mxnet-package-for-scala) |
| Windows | [Shared Library for Windows](http://mxnet.incubator.apache.org/install/windows_setup.html#build-the-shared-library) | [Scala Package for Windows](http://mxnet.incubator.apache.org/install/windows_setup.html#installing-the-mxnet-package-for-scala) |


## Build Scala from an Existing MXNet Installation
If you have already built MXNet **from source** and are looking to setup Scala from that point, you may simply run the following from the MXNet source root:

```
make scalapkg
make scalainstall
```

## Set Up Your Project

Now that you've installed your prerequisites, you are ready to setup IntelliJ and your first MXNet-Scala project!

**Step 1.** Install and setup IntelliJ:
    - When prompted for what to features to enable during IntelliJ's first startup, make sure you select Scala.

    - Install the plugin for IntelliJ IDE by following these steps:
   On **Menu**, choose **Preferences**, choose **Plugins**, type **Scala**, and then choose **Install**. For further plugin help and instructions, refer to [Scala plugin setup for IDE](https://www.jetbrains.com/help/idea/scala.html).

**Step 2.** Create a new project:

![intellij welcome](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/scala/intellij-welcome.png)

From the IntelliJ welcome screen, select "Create New Project".

![maven project type](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/scala/intellij-project-type.png)

Choose the Maven project type.

![maven project type - archetype](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/scala/intellij-project-type-archetype-check.png)

Select the checkbox for `Create from archetype`.

![maven project type - archetype](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/scala/intellij-project-type-archetype-add.png)

Click the `Add Archetype` button, and add the following information to each field.

**GroupId**
```
net.alchim31.maven
```
**ArtifactId**
```
scala-archetype-simple
```
**Version**
```
1.6
```
**Repository**
```
https://mvnrepository.com/artifact/net.alchim31.maven/scala-archetype-simple
```

![maven project type - archetype](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/scala/intellij-project-type-archetype-add-confirm.png)

Click `Ok` to add the archetype, make sure it is selected from the list, and then click `Next`.

![project metadata](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/scala/intellij-project-metadata.png)

Set the project's metadata. For this tutorial, use the following:

**GroupId**
```
your-name
```
**ArtifactId**
```
ArtifactId: scalaMXNet
```
**Version**
```
1.0-SNAPSHOT
```

![project properties](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/scala/intellij-project-properties.png)

Review the project's properties. The settings can be left as their default.

![project location](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/scala/intellij-project-location.png)

Set the project's location. The rest of the settings can be left as their default.

![project 1](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/scala/intellij-project-1.png)

After clicking Finish, you will be presented with the project's first view.
The project's `pom.xml` will be open for editing.

**Step 3.** Setup project properties:
  - Specify project properties in `pom.xml` by pasting the following content in the `<properties>` tag. You will be overwriting the `<scala.version>` tag in the process, upgrading from `2.11.5` to `2.11.8`.

```xml
<properties>
  <scala.version>2.11.8</scala.version>
  <scala.binary.version>2.11</scala.binary.version>
</properties>
```

**Step 4.** Setup project profiles and platforms:

  - Specify project profiles and platforms in `pom.xml` by pasting the following content below the `</properties>` tag:

```xml
<profiles>
    <profile>
        <id>osx-x86_64-cpu</id>
        <properties>
            <platform>osx-x86_64-cpu</platform>
        </properties>
    </profile>
    <profile>
        <id>linux-x86_64-cpu</id>
        <properties>
            <platform>linux-x86_64-cpu</platform>
        </properties>
    </profile>
    <profile>
        <id>linux-x86_64-gpu</id>
        <properties>
            <platform>linux-x86_64-gpu</platform>
        </properties>
    </profile>
</profiles>
```

**Step 5.** Setup project dependencies:

  - Specify project dependencies in `pom.xml` adding the dependencies listed below. Place them inside the `<dependencies>` tag:

```xml
<dependencies>
  <!-- Begin deps for MXNet -->
  <dependency>
    <groupId>ml.dmlc.mxnet</groupId>
    <artifactId>mxnet-full_${scala.binary.version}-${platform}</artifactId>
    <version>1.2.0</version>
    <scope>system</scope>
    <systemPath>/Development/incubator-mxnet/scala-package/assembly/osx-x86_64-cpu/target/mxnet-full_2.11-osx-x86_64-cpu-1.2.0-SNAPSHOT.jar</systemPath>
  </dependency>
  <dependency>
    <groupId>args4j</groupId>
    <artifactId>args4j</artifactId>
    <version>2.0.29</version>
  </dependency>
  <dependency>
    <groupId>org.slf4j</groupId>
    <artifactId>slf4j-api</artifactId>
    <version>${slf4jVersion}</version>
  </dependency>
  <dependency>
    <groupId>org.slf4j</groupId>
    <artifactId>slf4j-log4j12</artifactId>
    <version>${slf4jVersion}</version>
  </dependency>
  <!-- End deps for MXNet -->
  <dependency>
    <groupId>org.scala-lang</groupId>
    <artifactId>scala-library</artifactId>
    <version>${scala.version}</version>
  </dependency>
  <!-- Test -->
  <dependency>
    <groupId>junit</groupId>
    <artifactId>junit</artifactId>
    <version>4.11</version>
    <scope>test</scope>
  </dependency>
  <dependency>
    <groupId>org.specs2</groupId>
    <artifactId>specs2-core_${scala.compat.version}</artifactId>
    <version>2.4.16</version>
    <scope>test</scope>
  </dependency>
  <dependency>
    <groupId>org.specs2</groupId>
    <artifactId>specs2-junit_${scala.compat.version}</artifactId>
    <version>2.4.16</version>
    <scope>test</scope>
  </dependency>
  <dependency>
    <groupId>org.scalatest</groupId>
    <artifactId>scalatest_${scala.compat.version}</artifactId>
    <version>2.2.4</version>
    <scope>test</scope>
  </dependency>
</dependencies>
```

![project 2](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/scala/intellij-project-2.png)

Note the `<systemPath>` tag and update it to match the file path to the jar file that was created when you built the MXNet-Scala package. It can be found in the `mxnet-incubator/scala-package/assembly/{platform}/target` directory, and is named with the pattern `mxnet-full_${scala.binary.version}-${platform}-{version-SNAPSHOT}.jar`.

**Step 6.** Import dependencies with Maven:

  - Note the prompt in the lower right corner that states "Maven projects need to be imported".

![project 3](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/scala/intellij-project-3.png)

Click "Import Changes" in this prompt.

**Step 7.** Build the project:
- To build the project, from the menu choose Build, and then choose Build Project.

**Note**: During the build you may experience `[ERROR] scalac error: bad option: '-make:transitive'`. You can fix this by deleting or commenting this out in your `pom.xml`. This line in question is: `<arg>-make:transitive</arg>`.

**Step 8.** Run the Hello World App:

![hello world app](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/scala/intellij-project-hello-world-app.png)

Navigate to the App included with the project.

![run hello world](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/scala/intellij-project-hello-world-run.png)

Run the App by clicking the green arrow, and verify the Hello World output

**Step 9.** Run Sample MXNet Code in the App:

![run hello mxnet](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/scala/intellij-project-hello-mxnet.png)

Paste the following code in the App, overwriting the original hello world code. Then click the green arrow to run it.

```scala
object App extends App {
  import ml.dmlc.mxnet._
  import org.apache.log4j.BasicConfigurator
  BasicConfigurator.configure()

  private val a = NDArray.ones(2, 3)
  println("Testing MXNet by generating an 2x3 NDArray...")
  println("Shape is: ")
  println(a.shape)
}
```

![run hello world](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/scala/intellij-project-hello-mxnet-output.png)

Your result should be similar to this output.

### Command Line Build Option

- You can also compile the project by using the following command at the command line. Change directories to this project's folder then run the following:

```bash
    mvn clean package -e -P osx-x86_64-cpu
```
The `-P <platform>` parameter tells the build which platform to target.
The `-e` will give you more details if the build fails. If it succeeds, you should see a lot of info and some warning messages, followed by:

```bash
[INFO] ------------------------------------------------------------------------
[INFO] BUILD SUCCESS
[INFO] ------------------------------------------------------------------------
[INFO] Total time: 1.186 s
[INFO] Finished at: 2018-03-06T15:17:36-08:00
[INFO] Final Memory: 11M/155M
[INFO] ------------------------------------------------------------------------
```
The build generates a new jar file in the `target` folder called `scalaInference-1.0-SNAPSHOT.jar`.

## Next Steps
For more information about MXNet Scala resources, see the following:

* [Scala API](http://mxnet.io/api/scala/)
* [More Scala Examples](https://github.com/incubator-mxnet/tree/master/scala-package/examples/)
* [MXNet tutorials index](http://mxnet.io/tutorials/index.html)
