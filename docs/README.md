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

# Building and Updating MXNet Documentation

The website is hosted at http://mxnet.incubator.apache.org/.
http://mxnet.io redirects to this site and advised to use links with http://mxnet.incubator.apache.org/ instead of http://mxnet.io/.

MXNet Documentation Website is built with [Sphinx](http://www.sphinx-doc.org) and a variety of plugins including [pandoc](https://pandoc.org/), [recommonmark](https://github.com/rtfd/recommonmark), a custom Sphinx plugin ([mxdoc.py](https://github.com/apache/incubator-mxnet/blob/master/docs/mxdoc.py)).


## How to Build the MXNet Website for Development and QA

To be updated


## File Structure



## Production Website Deployment Process

[Apache Jenkins MXNet website building job](https://builds.apache.org/job/incubator-mxnet-build-site/) is used to build MXNet website.

The Jenkins docs build job will fetch MXNet repository, build MXNet website and push all static files to [host repository](https://github.com/apache/incubator-mxnet-site.git).
The host repo is hooked with [Apache gitbox](https://gitbox.apache.org/repos/asf?p=incubator-mxnet-site.git;a=summary) to host website.

### Process for Running the Docs Build Job

1. Login to [Jenkins](http://jenkins.mxnet-ci.amazon-ml.com/).
1. View the pipeline currently called `restricted website build`.
1. Click `Build with Parameters`.
1. Use the defaults, or change the domain to be your staging server's IP/DNS web address.
1. Wait about 20-30 minutes while it builds the full site.
1. On your staging server, clone the [mxnet site repo](https://github.com/apache/incubator-mxnet-site.git).
1. When you ran `restricted website build` it followed up with `restricted website publish` which pushed the changes to the incubator-mxnet-site repo.
1. Make sure you git pull if you had already cloned the site repo before this first run-through.
1. Copy the files to your webroot. For more info on this see the developer instructions for docs build.
1. Preview the site on your staging server. Note, use the domain default before you try to use this for production, but using your own is helpful for QA'ing the site.


## Build Versioning Website

To be updated

## Other Build Processes

* Perl API docs are maintained separately at [metacpan](https://metacpan.org/release/AI-MXNet).


## Troubleshooting

- If CSS or javascript are changed, clear the cache in the browser with a *forced refresh*.
- If search doesn't work, run `make clean` and then `make docs`.
