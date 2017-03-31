# MXNet documentation

MXNet's documents can be built by running `make html` in this folder.

A built version of document is available at http://mxnet.io

To build the documents locally, the easiest way is by using `docker`. First make
sure [docker](docker.com) is installed. Then use the following commands to clone and
build MXNet's documents (not including jupyter notebooks and API documents
execept for Python):

```bash
git clone --recursive https://github.com/dmlc/mxnet
cd mxnet
tests/ci_build/ci_build.sh doc DEV=1 make -C docs/ html
```

The built documents will be available at `docs/_build/html/`.

Note:

- If C++ codes have been changed, we suggest to remove the previous results before
  building, namely run `rm -rf docs/_build/html/`.

- If CSS or javascript are changed, we often need to do a *force refresh* in the
  browser to clear the cache.
