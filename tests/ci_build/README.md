# CI Build

This folder contains Dockerfiles and scripts for CI build.

Assume the we are on the root directory of mxnet, and the following env
variables are set.

```bash
export WORKSPACE=`pwd`
``

## Lint

```bash
docker build tests/ci_build/ -f tests/ci_build/Dockerfile.lint -t mxnet/lint && \
docker run --rm -v ${WORKSPACE}:/mxnet -w /mxnet mxnet/lint bash -c "make lint"
```
