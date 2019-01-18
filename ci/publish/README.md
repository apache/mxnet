# MXNet Publish Settings

This folder contains the configuration of restricted node on Jenkins for the publish. It also contains a folder called `scala` that contains everything required for scala publish. In this `README`, we would bring a brief walkthrough of the Jenkins configuration as well as the usages of the scala deployment files.

## Jenkins
Currently, Jenkins contains three build stages, namely `Build Packages`, `Test Packages` and `Deploy Packages`. During the `build package` stages, all dependencies will be built and a Scala package would be created. In the second stage, the package created from the previous stage would move to this stage to specifically run the tests. In the final stage, the packages passed the test would be deployed by the instances.

The job is scheduled to be triggered every 24 hours on a [restricted instance](http://jenkins.mxnet-ci.amazon-ml.com/blue/organizations/jenkins/restricted-publish-artifacts).

Currently, we are supporting tests in the following systems:

- Ubuntu 16.04
- Ubuntu 18.04
- Cent OS 7

All packages are currently built in `Ubuntu 14.04`. All Dockerfile used for publishing are available in `ci/docker/` with prefix `Dockerfile.publish`.

Apart from that, the script used to create the environment and publish are available under `ci/docker/install`:

- `ubuntu_publish.sh` install all required dependencies for Ubuntu 14.04 for publishing
- `ubuntu_base.sh` install minimum dependencies required to run the published packages

## Scala publish
Currently Scala publish on Linux is fully supported on jenkins. The `scala/` folder contains all files needed to do the publish. Here is a breif instroduction of the files:

- `build.sh` Main executable files to build the backend as well as scala package
- `buildkey.py` Main file used to extract password from the system and configure the maven
- `deploy.sh` Script to deploy the package
- `fullDeploy.sh` Used by CI to make full publish
- `test.sh` Make Scala test on CI

## Python
We plans to support Python build on Jenkins soon