# MATLAB binding for mxnet

### How to use

The only requirment is build mxnet to get `lib/libmxnet.so`. Sample usage

- Load model and data:

  ```matlab
  img = single(imresize(imread('cat.png'), [224 224])) - 120;
  model = mxnet.model;
  model.load('model/Inception_BN', 39);
  ```

- Get prediction:

  ```matlab
  pred = model.forward(img);
  ```

- Do feature extraction on GPU 0:

  ```matlab
  feas = model.forward(img, 'gpu', 0, {'max_pool_5b_pool', 'global_pool', 'fc'});
  ```

- See [demo.m](demo.m) for more examples

### Note on Implementation

We use `loadlibrary` to load mxnet library directly into Matlab and `calllib` to
call MXNet functions. Note that Matlab uses the column-major to store N-dim
arraies while and MXNet uses the row-major. So assume we create an array in
matlab with

```matlab
X = zeros([2,3,4,5]);
```

If we pass the memory of `X` into MXNet, then the correct shape will be
`[5,4,3,2]` in MXNet. When processing images, MXNet assumes the data layout is

```c++
example x channel x width x height
```

while in matlab we often store images by

```matlab
width x height x channel x example
```

So we should permuate the dimensions by `X = permute(X, [2, 1, 3, 4])` before
passing `X` into MXNet.

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
