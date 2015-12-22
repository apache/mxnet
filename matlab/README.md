# MATLAB binding for mxnet

### How to use

The only requirment is build mxnet to get `lib/libmxnet.so`. Then run `demo` in
matlab.

### FAQ

1. You may get the error `GLIBCXX_x.x.xx` is not found. Such as on Ubuntu 14.04:

```
> In loadlibrary (line 359)
Error using loadlibrary (line 447)
There was an error loading the library "/home/muli/work/mxnet/lib/libmxnet.so"
/usr/local/MATLAB/R2015a/bin/glnxa64/../../sys/os/glnxa64/libstdc++.so.6:
version `GLIBCXX_3.4.18' not found (required by
/home/muli/work/mxnet/lib/libmxnet.so)

Caused by:
    Error using loaddefinedlibrary
    /usr/local/MATLAB/R2015a/bin/glnxa64/../../sys/os/glnxa64/libstdc++.so.6:
    version `GLIBCXX_3.4.18' not found (required by
    /home/muli/work/mxnet/lib/libmxnet.so)
```

   One way to fix it is to link `MATLAB_ROOT/sys/os/glnxa64/libstdc++.so.6` to
   your system's `libstdc++`. For example

```bash
muli@ghc:/usr/local/MATLAB/R2015a/sys/os/glnxa64$ sudo rm -r libstdc++.so.6
muli@ghc:/usr/local/MATLAB/R2015a/sys/os/glnxa64$ sudo ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.19 libstdc++.so.6
```
