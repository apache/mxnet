# MXNet documentation

A built version of document is available at http://mxnet.io

To build the documents locally, we need to first install [docker](https://docker.com).
Then use the following commands to clone and
build the documents.

```bash
git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet
cd mxnet && make docs
```

The results will be available at `docs/_build/html/`.

Note:

- If C++ codes have been changed, we suggest to remove the previous results to
  trigger the rebuild for all pages, namely run `make clean_docs`.
- If C++ code fails to build, run `make clean`
- If CSS or javascript are changed, we often need to do a *force refresh* in the
  browser to clear the cache.
