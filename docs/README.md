# MXNet Documentation

The website is hosted at http://mxnet.incubator.apache.org/.
http://mxnet.io redirects to this site and advised to use links with http://mxnet.incubator.apache.org/ instead of http://mxnet.io/.

MXNet Documentation Website is built with [Sphinx](http://www.sphinx-doc.org) and a variety of plugins including [pandoc](https://pandoc.org/), [recommonmark](https://github.com/rtfd/recommonmark), a custom Sphinx plugin ([mxdoc.py](https://github.com/apache/incubator-mxnet/blob/master/docs/mxdoc.py)).


## How to Build the MXNet Website for Development and QA

* [Dependencies](build_doc_version/README.md#dependencies)
* [Developer Build Instructions](build_doc_version/README.md#developer-instructions)
* [Full Site Build Instructions](build_doc_version/README.md#full-website-build)


## File Structure

* Static files such as **css**, **javascript** and **html** templates are under the `_static` folder:
  - Javascript files are under `_static/js` folder
  - Layout templates and landing page html file are under `_static/mxnet-theme` folder
  - `_static/mxnet.css` contains all MXNet website styles

* Page contents originate as markdown files. Sphinx converts markdown files to html through an `rst` intermediate format. Each content folder should contain an index file as landing page.

* There are some utility scripts to help building website, such as `mxdoc.py` and `build_version_doc/`. They are used to manipulate website contents during building. Refer to [Developer Build Instructions](build_doc_version/README.md#developer-instructions) for more information.


## Production Website Building Process

**IMPORTANT**: this is currently offline.

[Apache Jenkins MXNet website building job](https://builds.apache.org/job/incubator-mxnet-build-site/) is used to build MXNet website.
There are two ways to trigger this job.
First is nightly build for master branch.
Second is manually trigger job when a new version is released. This will build for new version.

The job will fetch mxnet repository, build MXNet website and push all static files to [host repository](https://github.com/apache/incubator-mxnet-site.git).
The host repo is hooked with [Apache gitbox](https://gitbox.apache.org/repos/asf?p=incubator-mxnet-site.git;a=summary) to host website.

## Build Versioning Website

**IMPORTANT**: Refer to [Full Site Build Instructions](build_doc_version/README.md#full-website-build) for a working site build with the versions dropdown in the UI.



`build_doc.sh` is used by Apache Jenkins MXNet website building job to incremental adding of versions. We don't need it for local website development. **Currently offline due to site rendering problems.**


## Troubleshooting

- If C++ code has been changed, remove the previous results to trigger the rebuild for all pages. To do this, run `make clean_docs`.
- If C++ code fails to build, run `make clean`.
- If CSS or javascript are changed, clear the cache in the browser with a *forced refresh*.
- If search doesn't work, run `make clean` and then `make docs`.
