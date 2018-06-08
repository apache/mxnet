# MXNet Documentation

The website is hosted at http://mxnet.incubator.apache.org/.
http://mxnet.io redirects to this site and advised to use links with http://mxnet.incubator.apache.org/ instead of http://mxnet.io/.

MXNet Documentation Website is built with [Sphinx](http://www.sphinx-doc.org) and a variety of plugins including [pandoc](https://pandoc.org/), [recommonmark](https://github.com/rtfd/recommonmark), a custom Sphinx plugin ([mxdoc.py](https://github.com/apache/incubator-mxnet/blob/master/docs/mxdoc.py)).


## How to Build the MXNet Website for Development and QA

* [Dependencies](https://github.com/apache/incubator-mxnet/tree/master/docs/build_version_doc#dependencies)
* [Developer Build Instructions](https://github.com/apache/incubator-mxnet/tree/master/docs/build_version_doc#developer-instructions)
* [Full Site Build Instructions](https://github.com/apache/incubator-mxnet/tree/master/docs/build_version_doc#full-website-build)


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
1. View the pipeline currently called `website build pipeline`.
1. Click `Build with Parameters`.
1. Use the defaults, or change the domain to be your staging server's IP/DNS web address.
1. Wait about 20-30 minutes while it builds the full site.
1. On your staging server, clone the [mxnet site repo](https://github.com/apache/incubator-mxnet-site.git).
1. When you ran `website build pipeline` it followed up with website build - test publish which pushed the changes to the incubator-mxnet-site repo.
1. Make sure you git pull if you had already cloned the site repo before this first run-through.
1. Copy the files to your webroot. For more info on this see the developer instructions for docs build.
1. Preview the site on your staging server. Note, use the domain default before you try to use this for production, but using your own is helpful for QA'ing the site.


## Build Versioning Website

**IMPORTANT**: Refer to [Full Site Build Instructions](https://github.com/apache/incubator-mxnet/tree/master/docs/build_version_doc#full-website-build) for a working site build with the versions dropdown in the UI.


## Troubleshooting

- If C++ code has been changed, remove the previous results to trigger the rebuild for all pages. To do this, run `make clean_docs`.
- If C++ code fails to build, run `make clean`.
- If CSS or javascript are changed, clear the cache in the browser with a *forced refresh*.
- If search doesn't work, run `make clean` and then `make docs`.
