# Nightly build for mxnet

This fold contains scripts to test some heavy jobs, often training with multiple
GPUs, to ensure everything is right. Normally it runs everyday.
The current build server is equipped with Intel i7-4790 and 4 Nvidia GTX
970 Tis. The build status is available at [ci.dmlc.ml](http://ci.dmlc.ml). We
thank [Dave Andersen](http://www.cs.cmu.edu/~dga) for providing the build machine.

## How to use

### Run locallly

Run `tests/nightly/test_all.sh 4` if there are 4 GPUs.

### Run on Jenkins

First merge codes into the master branch, then go to
http://ci.dmlc.ml/job/mxnet/, click **Build Now**.
