# Install and run Java Examples

## Prerequisites:
Please follow the Step 1 in the [Scala configuration](http://mxnet.incubator.apache.org/install/scala_setup.html#setup-instructions)
These should help you install the correct Java version and all dependencies.

## Run the Java example project
We have provided a general MXNet Java template under `scala-package/mxnet-demo/java-demo` which contains complete instruction on running the Hello world and Object detection examples. Please copy the downloaded MXNet Java package jar file to the `java-demo` folder before build the package there.

## Import and run the Java package
For users using a desktop/laptop, we recommend using IntelliJ IDE as it is tested and supported to provide the necessary documentation for the Java API.

Alternatively, users can follow the second instruction to set up an empty Maven project for Java.

### IntelliJ instruction
If you are using a computer with Ubuntu16.04 or Mac, you can install IntelliJ to run the Java package. Please follow the instruction below:

1. Create a new Java project in IntelliJ. Fire up IntelliJ and click `Create New Project`.

2. Click `Next`, and in the `Create project from template` window, do not select anything and click `Next` again.

3. In the next window choose your `Project name` and the `Project location` and click on `Finish`.

4. Let's add the Java Inference API jars that we grabbed from Maven Central. At the top of the window, Go to the `File -> Project Structure`. In the popup window that opens up, click on `Libraries -> +` and select the path to the jar files downloaded. Click `Apply` and then click `OK`.

6. Create a new Java class under the folder `your-project-name/src`. Let's call this class `JavaSample.java`. Type in the following code snippet and run it. In this code snippet, we create an NDArray object in Java and print its shape.
```java
import org.apache.mxnet.javaapi.Context;
import org.apache.mxnet.javaapi.NDArray;

public class JavaSample {
public static void main(String[] args) {
  System.out.println("Hello");
  NDArray nd = NDArray.ones(Context.cpu(), new int[] {10, 20});

  System.out.println("Shape of NDarray is : "  + nd.shape());
}
}
```

7. If all went well, you should see an output like this : (Ignore the SLF4J warnings).
```
Hello
SLF4J: Failed to load class "org.slf4j.impl.StaticLoggerBinder".
SLF4J: Defaulting to no-operation (NOP) logger implementation
SLF4J: See http://www.slf4j.org/codes.html#StaticLoggerBinder for further details.
Shape of NDarray is : (10,20)
Process finished with exit code 0
```
This means you have successfully set it up on your machine

### Run the project manually in Maven
In this example, Maven is being used to create the project. This tutorial referred the [Maven in 5 min](https://maven.apache.org/guides/getting-started/maven-in-five-minutes.html) tutorial.

1. Create a new folder and run the following commands
```
mvn archetype:generate -DgroupId=com.mycompany.app -DartifactId=my-app -DarchetypeArtifactId=maven-archetype-quickstart -DinteractiveMode=false
```
You can specify the `groupId` and `artifactId` to your favourite names. You can also create a maven project using empty archetype.

2. then go to `pom.xml` file in your project folder and add the following content.

- Change the `osx-x86_64` to `linux-x86_64` if your platform is linux.
- Change `cpu` into `gpu` if you are using gpu
- Change the version of your package from `1.3.1-SNAPSHOT` to the matched jar version.
```xml
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.11-osx-x86_64-cpu</artifactId>
  <version>1.3.1-SNAPSHOT</version>
  <scope>system</scope>
  <systemPath>path-to-your-jar/jarName.jar</systemPath>
</dependency>
<dependency>
  <groupId>args4j</groupId>
  <artifactId>args4j</artifactId>
  <version>2.0.29</version>
  </dependency>
<dependency>
  <groupId>org.slf4j</groupId>
  <artifactId>slf4j-api</artifactId>
  <version>1.7.7</version>
</dependency>
<dependency>
  <groupId>org.slf4j</groupId>
  <artifactId>slf4j-log4j12</artifactId>
  <version>1.7.7</version>
</dependency>
```
3. Finally you can replace the code in `App.java`
```java
import org.apache.mxnet.javaapi.Context;
import org.apache.mxnet.javaapi.NDArray;

public class App {
public static void main(String[] args) {
  System.out.println("Hello");
  NDArray nd = NDArray.ones(Context.cpu(), new int[] {10, 20});

  System.out.println("Shape of NDarray is : "  + nd.shape());

}
}
```
make the package by
```
mvn package
```

and run it by
```
java -cp target/my-app-1.0-SNAPSHOT.jar:<full-path-to-jar>/<jarName>.jar com.mycompany.app.App
```
The result looks like this:
```
Hello
SLF4J: Failed to load class "org.slf4j.impl.StaticLoggerBinder".
SLF4J: Defaulting to no-operation (NOP) logger implementation
SLF4J: See http://www.slf4j.org/codes.html#StaticLoggerBinder for further details.
Shape of NDarray is : (10,20)
```