# MXNet Scala Package Test

This is an project created to run the test suite on a fully packaged mxnet jar. The test suite is found locally but mxnet is from the target jarfile.

## General Setup

To setup the packageTest, you must first build your tests. To build the tests, follow these steps from the mxnet main directory:

1. Build MXNet and the scala package from source following the directions [here](https://mxnet.incubator.apache.org/install/scala_setup.html#source)
2. Build the tests by running `mvn test-compile`.
3. Follow setup instructions below for your testing goal

## Running

There are three different modes of operation for testing based on the location of the jar and where it is coming from:

### Test Installed Jars

If you have a jar file, you can install it to your maven cache repository(`~/.m2/repository`). This might be useful if you acquire the .jar file from elsewhere. To install, it is easiest to use `mvn install:install-file -Dfile=<path-to-file> -DpomFile=<path-to-pomfile>`. If the pom file is not available, you can also run `mvn install:install-file -Dfile=<path-to-file> -DgroupId=<group-id> -DartifactId=<artifact-id> -Dversion=<version> -Dpackaging=<packaging>`. With the full mxnet jar, this might look like `mvn install:install-file -Dfile=<path-to-file> -DgroupId=org.apache.mxnet -DartifactId=mxnet-full_2.11-linux-x86_64-cpu -Dversion=1.3.0 -Dpackaging=jar`.

You can also run `mvn install` to install from a local build.

After installing, run `make testinstall` in the package test directory to run the tests.  Note that unless you also install an additional mxnetexamples jar, you can only run the unit tests.

### Test Local Deployment

To test the jars that would be produced by a deployment, you can run `mvn deploy` from the main mxnet directory. This produces a local snapshot located at `scala-package/deploy/target/repo`. To test this local snapshot, run `make testlocal`.  It also installs the component packages needed for testing the examples in `scala-package/*/target/repo`.

### Remote Repository Snapshot

This mode is to test a jar located in a remote repository. The default repository is the apache snapshot repisotory located at `https://repository.apache.org/content/repositories/snapshots`. Note that the actual jar in a repisotory should be located at `$repoUrl/org/apache/mxnet/mxnet-full_$scalaVersion-$osMode/$version/*.jar`.

Test the snapshot repo using `make testsnapshot` or a different repo using `make testsnapshot MXNET_REPO=$NEW_REPO_URL`.

### Options

You are able to run unit tests, integration tests, or both using this utility. To run the unit tests, add the flag `UNIT=1` to make (e.g. `make testsnapshot UNIT=1`). Use `INTEGRATION=1` for integration tests. The default behavior is to run both the unit and integration tests. However, the integration tests require that the mxnet examples be installed in addition to the full mxnet package (see test mode instructions above).

For running on GPU, add the flag `USE_CUDA=1`.

An additional option, you can specify the mxnet version with `MXNET_VERSION=1.3.1-SNAPSHOT`.

## Cleaning Up

You can clean temporary files and target artifacts by running `make clean`.

## Troubleshooting

### Missing Examples

If you fail with the following error
```
[ERROR] Failed to execute goal org.scalatest:scalatest-maven-plugin:1.0:test (test) on project mxnet-scala-packagetest-examples_2.11: There are test failures -> [Help 1]
[ERROR]
[ERROR] To see the full stack trace of the errors, re-run Maven with the -e switch.
[ERROR] Re-run Maven using the -X switch to enable full debug logging.
[ERROR]
[ERROR] For more information about the errors and possible solutions, please read the following articles:
[ERROR] [Help 1] http://cwiki.apache.org/confluence/display/MAVEN/MojoFailureException
[ERROR]
[ERROR] After correcting the problems, you can resume the build with the command
[ERROR]   mvn <goals> -rf :mxnet-scala-packagetest-examples_2.11
Makefile:57: recipe for target 'scalaintegrationtest' failed
make: *** [scalaintegrationtest] Error 1
```

and stacktrace begins with the following,

```
*** RUN ABORTED ***
  java.lang.NoClassDefFoundError: org/apache/mxnetexamples/Util$
```

you are missing the mxnetexamples package.  See your test mode installation section for details.
