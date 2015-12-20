

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

```bash
muli@ghc:/usr/local/MATLAB/R2015a/sys/os/glnxa64$ sudo rm -r libstdc++.so.6
[sudo] password for muli:
muli@ghc:/usr/local/MATLAB/R2015a/sys/os/glnxa64$ sudo ln -s libstdc++.so.6 /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.19
ln: failed to create symbolic link ‘/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.19’: File exists
muli@ghc:/usr/local/MATLAB/R2015a/sys/os/glnxa64$ sudo ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.19 libstdc++.so.6
muli@ghc:/usr/local/MATLAB/R2015a/sys/os/glnxa64$
```
