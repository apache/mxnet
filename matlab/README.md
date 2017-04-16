# MATLAB binding for MXNet

### How to use

MXNet needs to be built so that the `lib/libmxnet.so` is available, which can be done by:

```bash
cd ..
make
```
The pre-trained `Inception-BN` should be downloaded to obtain the symbol and network parameters.

```bash
./get_inception_model.sh
```

This data will be saved in the `./data` folder:

```bash
./data/
├── cat.png
├── Inception-BN-0126.params
├── Inception-BN-symbol.json
└── synset.txt
```

####Sample usage

Run the demo script from the command-line without invoking Matlab GUI:

```bash
matlab -nodisplay -nojvm -nosplash -nodesktop -r "run('./demo.m'), exit(0);"
```
or the script may be run from the Matlab GUI as usual.

The script has the following components:

- Load model
  
  ```matlab
  model = mxnet.model;
  model.load('data/Inception-BN', 126);
  ```

- Load data and normalise.  Here we assume a fixed value of 120 as 'mean image':

  ```matlab
  img = single(imresize(imread('./data/cat.png'), [224 224])) - 120;
  ```

- Get prediction:

  ```matlab
  pred = model.forward(img);
  ```

- Do feature extraction on CPU or GPU 0:

  ```matlab
  feas = model.forward(img, {'max_pool_5b_pool', 'global_pool', 'fc1'});           % CPU mode
  feas = model.forward(img, 'gpu', 0, {'max_pool_5b_pool', 'global_pool', 'fc1'}); % GPU mode
  ```

- See [demo.m](demo.m) for more details

### Note on Implementation

We use `loadlibrary` to load mxnet library directly into Matlab and `calllib` to
call MXNet functions. Note that Matlab uses the column-major to store N-dim
arrays while and MXNet uses the row-major. So assume we create an array in
Matlab with

```matlab
X = zeros([2,3,4,5]);
```

If we pass the memory of `X` into MXNet, then the correct shape will be
`[5,4,3,2]` in MXNet. When processing images, MXNet assumes the data layout is

```
batchSize x channel x width x height
```

while in Matlab we often store images in

```
width x height x channel x batchSize
```

So we should permute the dimensions by `X = permute(X, [2, 1, 3, 4])` before
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
    muli@ghc:/usr/local/MATLAB/R2015a/sys/os/glnxa64$ sudo ln -s /usr/lib/x86_64-linux-gnu/    libstdc++.so.6.0.19 libstdc++.so.6
    ```


2. Matlab binding has been tested with the following version:

    `R2016b (9.1.0.441655) 64-bit (glnxa64)`
