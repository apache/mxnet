<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
  ~
-->

# PyPI CD Pipeline

The Jenkins pipelines for continuous delivery of the PyPI MXNet packages.
The pipelines for each variant are run, and fail, independently. Each depends
on a successful build of the statically linked libmxet library.

The pipeline relies on the scripts and resources located in [tools/pip](https://github.com/apache/mxnet/tree/master/tools/pip)
to build the PyPI packages.

## Credentials

The pipeline depends on the following environment variables in order to successfully
retrieve the credentials for the PyPI account:

* CD_PYPI_SECRET_NAME
* DOCKERHUB_SECRET_ENDPOINT_URL
* DOCKERHUB_SECRET_ENDPOINT_REGION

The credentials are stored in the Secrets Manager of the AWS account hosting Jenkins.
The [pypi_publish.py](pypi_publish.sh) script is in charge of retrieving the credentials.

## Mock publishing

Because of space limitations on PyPI, we don't want to push test packages from Jenkins Dev
everytime the pipeline is run. Therefore, the [pypi_publish.sh](pypi_publish.sh) 
script will fake publishing packages if the `username` is *skipPublish*.
