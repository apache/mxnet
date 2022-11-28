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

The website is hosted at https://mxnet.apache.org/.
https://mxnet.io redirects to this site and advised to use links with https://mxnet.apache.org/ instead of https://mxnet.io/.

## Website & Documentation Contributions

Detailed information on website development, continuous integration, and proposals for future projects can be found on the [MXNet Wiki](https://cwiki.apache.org/confluence/display/MXNET/Website).

The website is built using Jekyll. You may run your own version of the static website by following the instructions on the wiki.

Each language documentation is built in a modular way, so that if you are a contributor to Julia, for example, you only need Julia-related tools to build it. Each language API has a section on installation and building along with how to build the docs locally.

You can also use the project's CI tools to emulate any changes with Docker. You can use these tools to install dependencies and run the parts of the build you want to test.

Refer to the [MXNet Developer Wiki](https://cwiki.apache.org/confluence/pages/viewpage.action?pageId=125309983) for instructions on building the docs locally.

If you plan to contribute changes to the documentation or website, please submit a pull request. Contributions are welcome!

## Python Docs

MXNet's Python documentation is built with [Sphinx](https://www.sphinx-doc.org) and a variety of plugins including [pandoc](https://pandoc.org/), and [recommonmark](https://github.com/rtfd/recommonmark).

More information on the dependencies can be found in the [CI folder's installation scripts](https://github.com/apache/mxnet/tree/master/ci/docker/install/ubuntu_docs.sh).

You can run just the Python docs by following the instructions in the Python API guide.

## Other API Docs

The docs are hosted on the website in each language API's section. You can find installation and build instructions there.

## How to Build the MXNet Website for Development and QA

`conda` or `miniconda` is recommended.
* [Conda](https://www.anaconda.com/distribution/#download-section) (install to PATH)

If you only need to make changes to tutorials or other pages that are not generated from one of the API source code folders, then you can use a basic Python pip or conda installation. But if you want edit the API source and have the reference API docs update, you also need to build MXNet from source. Refer to the build from source instructions for this requirement.


### Ubuntu Setup

As this is maintained for CI, Ubuntu is recommended. Refer to [ubuntu_doc.sh](https://github.com/apache/mxnet/tree/master/ci/docker/install/ubuntu_docs.sh) for the latest install script.

### Caveat for Rendering Outputs

Note that without a GPU you will not be able to generate the docs with the outputs in the tutorials.

### GPU setup
To run the full build, including tests of all tutorials,
**you will need at least two GPUs**.
Distributed training is a key feature of MXNet,
so multiple GPUs are required for running through every tutorial.
* [CUDA 9.2](https://developer.nvidia.com/cuda-downloads)

### CPU-only setup
In the `environment.yml` file:
* Change `mxnet-cu92` to `mxnet`.

### macOS setup
In the `environment.yml` file:
* Change `mxnet-cu92` to `mxnet`. (There is no CUDA package for mac anyway.)

### Windows Setup
If you have a GPU and have installed CUDA 9.2 you can leave the MXNet dependency alone.
Otherwise, in the `environment.yml` file:
* Change `mxnet-cu92` to `mxnet`.

Install recommended software:
* [git bash](https://gitforwindows.org/)
* Be sure to install `Conda` in `PATH`
* Install `make` from a `git bash` terminal with Admin rights
    - [Install chocolatey](https://chocolatey.org/install)
    - Use `choco to install make`
* Restart terminals after installations to make sure PATH is set.
    - The `choco`, `make`, and `conda` commands should work in `git bash`.

### Conda environment setup
Run the following commands from the project root (`new-docs`) to setup the environment.

```bash
conda env create -f environment.yml
source activate mxnet-docs
```

## Build the docs

* Change directories to `new-docs/python`.

To build without GPUs and without testing the notebooks (faster):

```bash
make EVAL=0
```

To build with testing the notebooks (requires GPU):

```bash
make
```

The build docs will be available at `build/_build/html`.

Each build may take a few minutes even without evaluation. To accelerate it, we can use one of the following ways:

1. open `build/conf.py`, add the folders you want to skip into `exclude_patterns`, such as `exclude_patterns = ['templates', 'api', 'develop', 'blog']`.
2. move the files into a different folder, such as `mv api /tmp/`, and then `make clean`.

## Check results

To run a server to see the website:

1. Start a http server: `cd build/_build/html; python -m http.server`
2. For viewing a remote machine, ssh to your machine with port forwarding: `ssh -L8000:localhost:8000 your_machine`
3. Open http://localhost:8000 in your local machine

## Run tutorials

In addition to view the built html pages, you can run the Jupyter notebook from a remote machine.
1. Install `notedown` plugin: `pip install https://github.com/mli/notedown/tarball/master` in remote server
2. Start Jupyter notebook `jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'` in remote server
3. ssh to your machine with port forwarding: `ssh -L8888:localhost:8888 your_machine`
4. Open http://localhost:8888 in your local machine and run the md files directly

Optionally, one can run the following to launch the notedown plugin automatically when starting jupyter notebook.
1. Generate the jupyter configure file `~/.jupyter/jupyter_notebook_config.py` if it
is not existing by run `jupyter notebook --generate-config`
2. Add `c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'` to `~/.jupyter/jupyter_notebook_config.py`
3. Simply run `jupyter notebook`

## Troubleshooting
Dependencies and the setup steps for this website are changing often. Here are some troubleshooting tips.

* You might need to update the environment for the latest modules.
```bash
conda env update -f environment.yml
```

The `-W` Sphinx option enforces "warnings as errors". This will help you debug your builds and get them through CI.
**CI will not let a PR through if it breaks the website.** Refer to the [MXNet Developer wiki's documentation guide](https://cwiki.apache.org/confluence/display/MXNET/Documentation+Guide) for troubleshooting tips.


## Production Website Deployment Process

[Apache Jenkins MXNet website building job](https://builds.apache.org/job/mxnet-build-site/) is used to build MXNet website.

The Jenkins docs build job will fetch MXNet repository, build MXNet website and push all static files to [host repository](https://github.com/apache/mxnet-site.git).

The host repo is hooked with [Apache gitbox](https://gitbox.apache.org/repos/asf?p=mxnet-site.git;a=summary) to host website.

### Processes for Running the Docs Build Jobs

This information is maintained on the [MXNet Wiki](https://cwiki.apache.org/confluence/display/MXNET/Website).


## Other Docs Build Processes

* Perl API docs are maintained separately at [metacpan](https://metacpan.org/release/AI-MXNet).


## Troubleshooting

- If C++ code has been changed, remove the previous results to trigger the rebuild for all pages. To do this, run `make clean_docs`.
- If C++ code fails to build, run `make clean`.
- If CSS or javascript are changed, clear the cache in the browser with a *forced refresh*.
- If search doesn't work, run `make clean` and then `make docs`.
