# MXNet Publish Settings

This folder contains the configuration for restricted nodes on Jenkins for the publishing MXNet artifacts. It also contains a folder called `scala` that contains everything required for publishing to Maven. In this `README`, we provide a brief walkthrough of the Jenkins configuration as well as the usage of the Scala deployment files. Python publishing is TBD.

## Jenkins
Currently, Jenkins contains three build stages, namely `Build Packages`, `Test Packages` and `Deploy Packages`. During the `build package` stages, all dependencies are built and a Scala package are created. In the second stage, the package created from the previous stage moves to this stage to specifically run the tests. In the final stage, the packages that pass the tests are deployed by the instances.

The job is scheduled to be triggered every 24 hours on a [restricted instance](http://jenkins.mxnet-ci.amazon-ml.com/blue/organizations/jenkins/restricted-publish-artifacts).

Currently, we are supporting tests in the following systems:

- Ubuntu 16.04
- Ubuntu 18.04
- Cent OS 7

All packages are currently built in `Ubuntu 14.04`. All Dockerfile used for publishing are available in `ci/docker/` with prefix `Dockerfile.publish`.

Apart from that, the script used to create the environment and publish are available under `ci/docker/install`:

- `ubuntu_publish.sh` installs all required dependencies for Ubuntu 14.04 for publishing
- `ubuntu_base.sh` installs minimum dependencies required to run the published packages

## Scala publishing
Currently Scala publish on Linux is fully supported on Jenkins. The `scala/` folder contains all files needed for publishing. Here is a brief introduction of the files:

- `build.sh` Main executable files to build the backend as well as scala package
- `buildkey.py` Main file used to extract password from the system and configure the maven
- `deploy.sh` Script to deploy the package
- `fullDeploy.sh` Used by CI to make full publish
- `test.sh` Make Scala test on CI

## Python publishing
Python build support is TBD.
