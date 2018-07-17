# Nightly Tests for MXNet 

These are some longer running tests that are scheduled to run every night. 

### Description
There are two Jenkins pipelines that run these tests - 
1. Tests on the Source code: http://jenkins.mxnet-ci.amazon-ml.com/view/Nightly%20Tests/job/NightlyTests_onSource/
2. Tests on Built Binaries: http://jenkins.mxnet-ci.amazon-ml.com/view/Nightly%20Tests/job/NightlyTests_onBinaries/

### Adding a new Nightly Test
Add your test script to the mxnet repo (preferably)in the tests/nightly folder)
Make sure to describe in a readme or comments the purpose of the test. 

#### Setting up the Docker Container 
1. Your test must run on the CI slaves only within an official docker container available at ci/docker
2. Make sure your script itself does not install anything on the slaves. All dependencies must be added to the dockerfile.
3. For most tests you should be able to find an existing dockerfile that contains all the necessary dependencies - select based on the required os, cpu vs gpu, necessary dependencies etc
4. However, If there is no dockerfile which installs the required dependencies do the following - 
    a. Add your dependencies to a new install script in ci/docker/install
    b. Now either run this install script within an existing dockerfile or create a new dockerfile. 

#### Running the test from the Jenkinsfile
If the test runs on the MXNet source, modify tests/nightly/Jenkinsfile - 
1. Add a function to ci/docker/runtimefunctions.sh which calls your test script. 
2. In the Jenkinsfile, add a new test similar to existing tests under the `NightlyTests` stage. Make sure to call the right runtime function

If the test runs on MXNet binaries modify tests/nightly/JenkinsfileForBinaries -
1. Add a function to ci/docker/runtimefunctions.sh which calls your test script. 
2. In the Jenkinsfile, if needed, add a new build similar to existing tests under the `Build` stage. 
3. Add a new test similar to existing tests under the `NightlyTests` stage. Make sure to call the right runtime function

### Currently Running Tests

#### Tests on Source
1. Amalgamation Tests
2. Compilation Warnings
3. Installation Guide
4. MXNet Javascript Test
5. Apache RAT check

#### Tests on Binaries
1. Image Classification Test
2. Single Node KVStore 
