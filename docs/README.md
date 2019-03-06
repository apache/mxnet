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

Using `make docs` from the MXNet root is the quickest way to generate the MXNet API docs and the website, as long as you already have all of the dependencies installed. This method automatically generates each API, [except the Perl and R APIs](#other-build-processes).

**Easy docs setup for Ubuntu:** Run the following on Ubuntu 16.04 to install all MXNet and docs dependencies and to build MXNet from source. Then issue the `make docs` command from the source root to build the docs.

```bash
git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet
cd mxnet/docs/build_version_doc
./setup_docs_ubuntu.sh
cd ../../
make docs USE_OPENMP=1 SPHINXOPTS=-W
```

OpenMP speeds things up and will work on Ubuntu if you used the `setup_docs_ubuntu.sh` script.
The `-W` Sphinx option enforces "warnings as errors". This will help you debug your builds and get them through CI.
**CI will not let a PR through if it breaks the website.** Refer to the [MXNet Developer wiki's documentation guide](https://cwiki.apache.org/confluence/display/MXNET/Documentation+Guide) for troubleshooting tips.

For more information on each API's documentation dependencies, how to serve the docs, or how to build the full website with each legacy MXNet version, refer to the following links:

* [Dependencies](https://github.com/apache/incubator-mxnet/tree/master/docs/build_version_doc#dependencies) - required before you build the docs
* [Developer Build Instructions](https://github.com/apache/incubator-mxnet/tree/master/docs/build_version_doc#developer-instructions) - build your local branch
* [Full Site Build Instructions](https://github.com/apache/incubator-mxnet/tree/master/docs/build_version_doc#full-website-build) - build the latest commits to the official branches


## File Structure

* Static files such as **css**, **javascript** and **html** templates are under the `_static` folder:
  - Javascript files are under `_static/js` folder
  - Layout templates and landing page html file are under `_static/mxnet-theme` folder
  - `_static/mxnet.css` contains all MXNet website styles

* Page contents originate as markdown files. Sphinx converts markdown files to html through an `rst` intermediate format. Each content folder should contain an index file as landing page.

* There are some utility scripts to help building website, such as `mxdoc.py` and `build_version_doc/`. They are used to manipulate website contents during building. Refer to [Developer Build Instructions](https://github.com/apache/incubator-mxnet/tree/master/docs/build_version_doc#developer-instructions) for more information.


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

**IMPORTANT**: Refer to [Full Site Build Instructions](https://github.com/apache/incubator-mxnet/tree/master/docs/build_version_doc#full-website-build) for a working site build with the versions dropdown in the UI.


## Other Build Processes

* Perl API docs are maintained separately at [metacpan](https://metacpan.org/release/AI-MXNet).
* R API docs building must be triggered manually. The function for generating these automatically was disabled in the nightly builds. You may run the R docs build process in a local docs build by uncommenting the [function call in mxdoc.py](https://github.com/apache/incubator-mxnet/blob/master/docs/mxdoc.py#L378).


## Troubleshooting

- If C++ code has been changed, remove the previous results to trigger the rebuild for all pages. To do this, run `make clean_docs`.
- If C++ code fails to build, run `make clean`.
- If CSS or javascript are changed, clear the cache in the browser with a *forced refresh*.
- If search doesn't work, run `make clean` and then `make docs`.
