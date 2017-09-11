# MXNet documentation

## How to build MXNet website

MXNet Documentation Website is built with [sphinx 1.5.1](http://www.sphinx-doc.org/en/1.5.1/intro.html).

A built version of document is available at http://mxnet.io

To build the documents locally, we need to first install [docker](https://docker.com).
Then use the following commands to clone and
build the documents.

```bash
git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet
cd mxnet && make docs
```

In case docker method is not available, there is an alternate method:
```bash
sudo pip install sphinx==1.5.1 CommonMark==0.5.4 breathe mock==1.0.1 recommonmark pypandoc
cd mxnet/docs && make html USE_OPENMP=0
```

The results will be available at `docs/_build/html/`.

Note:

- If C++ codes have been changed, we suggest to remove the previous results to
  trigger the rebuild for all pages, namely run `make clean_docs`.
- If C++ code fails to build, run `make clean`
- If CSS or javascript are changed, we often need to do a *force refresh* in the
  browser to clear the cache.
- If search doesn't work, we need to `make clean` and rebuild.
  
## File structure

1. Static files such as css, javascript and html templates are under `_static` folder:
- Javascript files are under `_static/js` folder.
- Layout templates and landing page html file are under `_static/mxnet-theme` folder.
- `_static/mxnet.css` contains all MXNet website styles.

2. Sphinx converts markdowns files to html. Page contents are markdown files. Each content folder 
contains an index file as landing page.

3. There are some utility scripts to help building website, such as `mxdoc.py` and `build_version_doc/`.
They are used to manipulate website contents during building.

## Production website building process

[Apache Jenkins MXNet website building job](https://builds.apache.org/job/incubator-mxnet-build-site/) is used to build MXNet website. 
There are two ways to trigger this job. 
First is nightly build for master branch. 
Second is manually trigger job when a new version is released. This will build for new version.

The job will fetch mxnet repository, build MXNet website and push all static files to [host repository](https://github.com/apache/incubator-mxnet-site.git). 
The host repo is hooked with [Apache gitbox](https://gitbox.apache.org/repos/asf?p=incubator-mxnet-site.git;a=summary) to host website.

## Build versioning website

`make docs` doesn't add any version information. Version information is added by [Apache Jenkins MXNet website building job](https://builds.apache.org/job/incubator-mxnet-build-site/).
Landing page will point to the latest released version. Older versions and master version are placed under versions folder. 
After completing website update and testing it locally, we also need to build and test versioning website.

Python Beautifulsoup is the dependency:

```bash
sudo pip install beautifulsoup4
```

The essenitial part of adding version is to use `AddPackageLink.py` to add Apache source packages and 
`AddVersion.py` to update all version related information on website. These two scripts are used in `build_doc.sh` and `build_all_version`. 

`build_doc.sh` is used by Apache Jenkins MXNet webiste building job to incremental adding version. We don't need it 
for local website development. 

`build_all_version.sh` is to rebuild versioning website locally and is useful to verify versioning website locally. 
We need to specify which versions to be built. This can be set in `tag_list` variable at the beginning of the script. 
Version order should be from latest to oldest and placing master at the end. We may also want to modify `mxnet_url` 
variable to our own repository for local testing. Another use case is to completely rebuild website with specific versions. 
Although this will not happen often, we can use it to rebuld whole website and push to [host repo](https://github.com/apache/incubator-mxnet-site.git).

```bash
./build_all_version.sh
```

## Develop notes

1. `AddVersion.py` depends on Beautiful library, which requires target html files to have close tags. Although open tag html can still be rendered by browser, it will be problematic for Beautifulsoup. 

2. `AddVersion.py` and `AddPackageLink.py` manipulates contents for website. If there are layout changes, it may break these two scripts. We need to change scripts respectively.

