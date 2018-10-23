# MXNet Scala Package Test

This is an project created to run the test suite on a fully packaged mxnet jar.

## Setup

### Install Package

To run the test suite, first install the package.  This can be done either by installing directly from a jar with `mvn install:install-file -Dfile=<path-to-file>` or by running `make scalainstall` in the main mxnet folder.  Note that if you use `mvn install:install-file`, you will be unable to run the example tests unless you also install the mxnetexamples jar. You can run all tests except for those examples with `make scalaintegrationtestwithoutexamples`.

### Build

Build the mxnet tests by running `make scalapkg` and then `make scalatestcompile` from the main mxnet directory.  This is needed for test discovery.

## Run

To run, ensure the versions are correct in the `Makefile`.  Then, just run `make scalaintegrationtest` to execute the test suite

## Clean

You can clean temporary files and target artifacts by running `make scalaclean`.

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

you are missing the mxnetexamples package.  See the "Install Package" section for details.
