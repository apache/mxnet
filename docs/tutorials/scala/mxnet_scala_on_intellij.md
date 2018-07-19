# Run MXNet Scala Examples Using the IntelliJ IDE (macOS)

This tutorial guides you through setting up a Scala project in the IntelliJ IDE on macOS, and shows how to use the MXNet package from your application.

## Prerequisites:
To use this tutorial you need the following software:

- [Java 8 JDK](http://www.oracle.com/technetwork/java/javase/downloads/index.html)
- [Maven](https://maven.apache.org/install.html)
- [Scala](https://www.scala-lang.org/download/) - comes with IntelliJ, so you don't need to install it separately
- [OpenCV](https://opencv.org/)
- [IntelliJ IDE](https://www.jetbrains.com/idea/)

The following instructions are provided for macOS and Ubuntu 16.04. Windows support is not yet available.

### Mac Prerequisites Setup

**Step 1.** Install brew:
```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

Or, if you already have brew, update it:
```
brew update
```

**Step 2.** Install Java 8 JDK:
```
brew tap caskroom/versions
brew cask install java8
```

**Step 3.** Install maven:
```
brew install maven
```

**Step 4.** Install OpenCV:
```
brew install opencv@2
```


### Ubuntu Prerequisites Setup

**Step 1.**: Download the MXNet source.

```bash
git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet
cd mxnet
```

**Step 2.**: Run the dependency installation scripts.

```bash
sudo ./ci/docker/install/ubuntu_core.sh
sudo ./ci/docker/install/ubuntu_scala.sh
```


## Access the Maven Package for MXNet

To view the latest Maven packages, you can check the Maven repository for the `org.apache.mxnet` packages.

*  [MXNet Maven Packages](https://search.maven.org/#search%7Cga%7C1%7Cg%3A%22org.apache.mxnet%22)

In the tutorial below, you will install the latest package at the time of this writing. You may find a different artifact for your specific operating system, or a newer one by reviewing what is currently available in the Maven repo.

The configuration you should update is in the pom file's dependency for MXNet:

```html
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.11-osx-x86_64-cpu</artifactId>
  <version>1.2.0</version>
</dependency>
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
mxnet
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

**Step 3.** Replace the pom file's content with the following code. Changes include:
  - Project properties: `scala.version`, upgrading from `2.11.5` to `2.11.8`
  - Project dependencies: adding the MXNet package from Maven and updating the dependency for JUnitRunner (specs2-junit_)
  - Build options: removing '-make:transitive'


```html
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/maven-v4_0_0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>mxnet</groupId>
  <artifactId>scalaMXNet</artifactId>
  <version>1.0-SNAPSHOT</version>
  <name>${project.artifactId}</name>
  <description>My wonderful scala app</description>
  <inceptionYear>2018</inceptionYear>
  <licenses>
    <license>
      <name>Apache2</name>
      <url>http://mxnet.apache.org</url>
      <distribution>repo</distribution>
    </license>
  </licenses>
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
  <properties>
    <maven.compiler.source>1.6</maven.compiler.source>
    <maven.compiler.target>1.6</maven.compiler.target>
    <encoding>UTF-8</encoding>
    <scala.version>2.11.8</scala.version>
    <scala.compat.version>2.11</scala.compat.version>
  </properties>

  <dependencies>
    <dependency>
      <groupId>org.apache.mxnet</groupId>
      <artifactId>mxnet-full_2.11-osx-x86_64-cpu</artifactId>
      <version>1.2.0</version>
    </dependency>
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

  <build>
    <sourceDirectory>src/main/scala</sourceDirectory>
    <testSourceDirectory>src/test/scala</testSourceDirectory>
    <plugins>
      <plugin>
        <!-- see http://davidb.github.com/scala-maven-plugin -->
        <groupId>net.alchim31.maven</groupId>
        <artifactId>scala-maven-plugin</artifactId>
        <version>3.2.0</version>
        <executions>
          <execution>
            <goals>
              <goal>compile</goal>
              <goal>testCompile</goal>
            </goals>
            <configuration>
              <args>
                <!--arg>-make:transitive</arg-->
                <arg>-dependencyfile</arg>
                <arg>${project.build.directory}/.scala_dependencies</arg>
              </args>
            </configuration>
          </execution>
        </executions>
      </plugin>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-surefire-plugin</artifactId>
        <version>2.18.1</version>
        <configuration>
          <useFile>false</useFile>
          <disableXmlReport>true</disableXmlReport>
          <!-- If you have classpath issue like NoDefClassError,... -->
          <!-- useManifestOnlyJar>false</useManifestOnlyJar -->
          <includes>
            <include>**/*Test.*</include>
            <include>**/*Suite.*</include>
          </includes>
        </configuration>
      </plugin>
    </plugins>
  </build>
</project>
```

**Step 4.** Import dependencies with Maven:

  - Note the prompt in the lower right corner that states "Maven projects need to be imported". If this is not visible, click on the little greed balloon that appears in the lower right corner.

![import dependencies](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/scala/intellij-project-import-changes.gif)

Click "Import Changes" in this prompt.

**Step 5.** Build the project:
- To build the project, from the menu choose Build, and then choose Build Project.


**Step 6.** Run the Hello World App:

![hello world app](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/scala/intellij-project-hello-world-app.png)

Navigate to the App included with the project.

![run hello world](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/scala/intellij-project-hello-world-run.png)

Run the App by clicking the green arrow, and verify the Hello World output

**Step 7.** Run Sample MXNet Code in the App:

![run hello mxnet](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/scala/intellij-project-hello-mxnet.png)

Paste the following code in the App, overwriting the original hello world code. Then click the green arrow to run it.

```scala
object App extends App {
  import org.apache.mxnet._

  private val a = NDArray.ones(2, 3)
  println("Testing MXNet by generating an 2x3 NDArray...")
  println("Shape is: ")
  println(a.shape)
}
```

The result should be this output:

```
Testing MXNet by generating an 2x3 NDArray...
Shape is:
(2,3)

Process finished with exit code 0
```


### Troubleshooting

If you get an error, check the dependencies at the beginning of this tutorial. For example, you might see the following in the middle of the error messages.

```
...
Library not loaded: /usr/local/opt/opencv@2/lib/libopencv_calib3d.2.4.dylib
...
```

This can be resolved be installing OpenCV2.



### Troubleshooting

If you get an error, check if it is like this one regarding OpenCV. For example, you might see the following in the middle of the error messages.

```
...
Library not loaded: /usr/local/opt/opencv@2/lib/libopencv_calib3d.2.4.dylib
...
```

This can be resolved be installing OpenCV2.

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
* [Scala Inference API](https://mxnet.incubator.apache.org/api/scala/infer.html)
* [Scala Examples](https://github.com/apache/incubator-mxnet/tree/master/scala-package/examples/)
* [MXNet Tutorials Index](http://mxnet.io/tutorials/index.html)
