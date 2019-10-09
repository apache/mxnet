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

Refer to the [MXNet Developer Wiki](https://cwiki.apache.org/confluence/display/MXNET/Building+the+New+Website) for instructions on building the docs locally.

If you plan to contribute changes to the documentation or website, please submit a pull request. Contributions are welcome!

## Python Docs

MXNet's Python documentation is built with [Sphinx](https://www.sphinx-doc.org) and a variety of plugins including [pandoc](https://pandoc.org/), and [recommonmark](https://github.com/rtfd/recommonmark).

More information on the dependencies can be found in the [CI folder's installation scripts](https://github.com/apache/incubator-mxnet/tree/master/ci/docker/install/ubuntu_docs.sh).

You can run just the Python docs by following the instructions in the Python API guide.

## Other API Docs

The docs are hosted on the website in each language API's section. You can find installation and build instructions there.

## How to Build the MXNet Website for Development and QA

`conda` or `miniconda` is recommended.
* [Conda](https://www.anaconda.com/distribution/#download-section) (install to PATH)

If you only need to make changes to tutorials or other pages that are not generated from one of the API source code folders, then you can use a basic Python pip or conda installation. But if you want edit the API source and have the reference API docs update, you also need to build MXNet from source. Refer to the build from source instructions for this requirement.


### Ubuntu Setup

As this is maintained for CI, Ubuntu is recommended. Refer to [ubuntu_doc.sh](https://github.com/apache/incubator-mxnet/tree/master/ci/docker/install/ubuntu_docs.sh) for the latest install script.

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

[Apache Jenkins MXNet website building job](https://builds.apache.org/job/incubator-mxnet-build-site/) is used to build MXNet website.

The Jenkins docs build job will fetch MXNet repository, build MXNet website and push all static files to [host repository](https://github.com/apache/incubator-mxnet-site.git).

The host repo is hooked with [Apache gitbox](https://gitbox.apache.org/repos/asf?p=incubator-mxnet-site.git;a=summary) to host website.

### Processes for Running the Docs Build Jobs

This information is maintained on the [MXNet Wiki](https://cwiki.apache.org/confluence/display/MXNET/Website).


## Other Docs Build Processes

* Perl API docs are maintained separately at [metacpan](https://metacpan.org/release/AI-MXNet).


## Troubleshooting

- If C++ code has been changed, remove the previous results to trigger the rebuild for all pages. To do this, run `make clean_docs`.
- If C++ code fails to build, run `make clean`.
- If CSS or javascript are changed, clear the cache in the browser with a *forced refresh*.
- If search doesn't work, run `make clean` and then `make docs`.



Apache License
                           Version 2.0, January 2004
                        https://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright 2019 Rolando Gopez Lacuata

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       https://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

