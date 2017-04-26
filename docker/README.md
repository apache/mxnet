# Docker images for MXNET

## How to use

First make sure [docker](https://docs.docker.com/engine/installation/) is
installed. The docker plugin
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker) is required to run on
Nvidia GPUs.

Pre-built docker containers are available at https://hub.docker.com/r/mxnet/

For example, the following command launches a container with the Python package
installed. It will pull the docker images from docker hub if it does not exist
locally.

```bash
docker run -ti --rm mxnet/python
```

Then you can run MXNet in python, e.g.:

```bash
# python -c 'import mxnet as mx; a = mx.nd.ones((2,3)); print((a*2).asnumpy())'
[[ 2.  2.  2.]
 [ 2.  2.  2.]]
```

If the host machine has at least one GPU installed and `nvidia-docker` is installed, namely
`nvidia-docker run --rm nvidia/cuda nvidia-smi` runs successfully, then you can
run a container with GPU supports

```bash
nvidia-docker run -ti --rm mxnet/python:gpu
```

Now you can run the above example in `GPU 0`:

```bash
# python -c 'import mxnet as mx; a = mx.nd.ones((2,3), mx.gpu(0)); print((a*2).asnumpy())'
[[ 2.  2.  2.]
 [ 2.  2.  2.]]
```

## Hosted containers

All images are based on Ubuntu 14.04. The `gpu` tag is built with CUDA 8.0 and
cuDNN 5.

### Python

Hosted at https://hub.docker.com/r/mxnet/python/

Python versions: 2.7.12 and 3.5.2.

Available tags:

- mxnet/python
- mxnet/python:gpu

Jupyter is included in python image and it is possilbe to use browser on the host to run jupyter on the docker image.
## Instructions

bind port 9999 or any other port you would like from host to docker using -p switch. 
```bash
docker run -it -p 9999:8888 mxnet/python
```


The previous line will open an interactive session on your docker. enter below line onto the console of your docker.

```bash
jupyter notebook --no-browser --allow-root --ip='*' &
```
the outcome is similar to:

```bash
  Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://localhost:8888/?token=fdded3fa0be0cf582f67a86d21fc4bcd7bd15176adf2d795
```
copy the link and paste it in your host machine's browser. Do not forget to change the port from 8888 to 9999 or whatever port your have used.

### R

Hosted at https://hub.docker.com/r/mxnet/r-lang/

R version: 3.3.3

Available tags:

- mxnet/r-lang
- mxnet/r-lang:gpu


### Julia

Hosted at https://hub.docker.com/r/mxnet/julia/

Julia version: 0.5.1

Available tags:

- mxnet/julia
- mxnet/julia:gpu

#### Scala

Hosted at https://hub.docker.com/r/mxnet/scala/

Scala version: 2.11.8

Available tags:

- mxnet/scala

## How to build

The following command build the default Python package

```bash
./tool.sh build python cpu
```

Run `./tool.sh` for more details. Use


Tips: The following commands stop all docker containers and delete all docker images.

```bash
docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)
```

```bash
docker rmi $(docker images -a -q)
```
