---
layout: page
title: Submit a Pull Request
subtitle: What to do to submit a pull request
action: Contribute
action_url: /community/index
permalink: /community/pull_request
---
<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

Submit a Pull Request
=====================

This is a quick guide to submit a pull request, please also refer to the
detailed guidelines.

-   Before submit, please rebase your code on the most recent version of
    master, you can do it by

    ```bash
    git remote add upstream git@github.com:apache/incubator-mxnet.git
    git fetch upstream
    git rebase upstream/master
    ```

-   Make sure code style check pass by typing the following command, and
    all the existing test-cases pass.

    ```bash
    # Reproduce the lint procedure in the CI.
    ci/build.py -R --docker-registry mxnetci --platform ubuntu_cpu --docker-build-retries 3 --shm-size 500m /work/runtime_functions.sh sanity
    ```

-   Add test-cases to cover the new features or bugfix the patch
    introduces.

-   Document the code you wrote, see more at [Write Document and Tutorials]({% link pages/community/document.md %}).

-   Send the pull request and fix the problems reported by automatic
    checks.

-   Request code reviews from other contributors and improves your patch
    according to feedbacks.

    -   To get your code reviewed quickly, we encourage you to help
        review others\' code so they can do the favor in return.
    -   Code review is a shepherding process that helps to improve
        contributor\'s code quality. We should treat it proactively, to
        improve the code as much as possible before the review. We
        highly value patches that can get in without extensive reviews.
    -   The detailed guidelines and summarizes useful lessons.

-   The patch can be merged after the reviewers approve the pull
    request.

CI Environment
--------------

We use docker containers to create stable CI environments that can be
deployed to multiple machines. Because we want a relatively stable CI
environment and make use of pre-cached image, all of the CI images are
built and maintained by committers.

Upgrade of CI base images are done automatically from the MXNet master branch
CI builds, tracked in [restricted-docker-cache-refresh](https://jenkins.mxnet-ci.amazon-ml.com/blue/organizations/jenkins/restricted-docker-cache-refresh/activity).
Sometimes this can be broken and needs fixes to accommodate
the new env. When this happens, send a PR to fix the build script in the repo.

Testing
-------

Even though we have hooks to run unit tests automatically for each pull
request, It\'s always recommended to run unit tests locally beforehand
to reduce reviewers\' burden and speedup review process.

### C++

C++ tests are maintained in [/tests/cpp](https://github.com/apache/incubator-mxnet/tree/master/tests/cpp) and requires [gtest](https://github.com/google/googletest) to build and run. Once you complete building the MXNet binary, tests are automatically built and generated in `/build/tests/mxnet_unit_tests`.

### Python

The dependencies for testing pipelines can be found in [/ci/docker/install/requirements](https://github.com/apache/incubator-mxnet/blob/master/ci/docker/install/requirements). To install these dependencies:

```bash
pip install --user -r ci/docker/install/requirements
```

<script defer src="https://use.fontawesome.com/releases/v5.0.12/js/all.js" integrity="sha384-Voup2lBiiyZYkRto2XWqbzxHXwzcm4A5RfdfG6466bu5LqjwwrjXCMBQBLMWh7qR" crossorigin="anonymous"></script>
<script async defer src="https://buttons.github.io/buttons.js"></script>
<script src="https://apis.google.com/js/platform.js"></script>
