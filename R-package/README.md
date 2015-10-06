MXNet R-Package
===============
This is an on-going effort to support mxnet in R, stay tuned.

Bleeding edge Installation
- First build ```../lib/libmxnet.so``` by following [Build Instruction](doc/build.md)
- Set the path to ```lib/libmxnet.so``` in ```LD_LIBRARY_PATH```, you can do it by modify ```~/.bashrc```
```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/lib/libmxnet.so
```
- Type ```R CMD INSTALL R-package``` in the root folder.
