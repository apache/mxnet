# 在 Docker上运行 MXNet 

通过 [Docker](http://docker.com/) 你可以编译独立版本的 Linux 操作系统，它可以单独运行在你的电脑上。在这个独立版本的 Linux 上你可以运行 MXNet 以及其他软件，并且和你电脑上的软件不会产生相互影响。

MXNet 为 Docker 提供了两个镜像:

1. MXNet Docker (CPU) - [https://hub.docker.com/r/kaixhin/mxnet/](https://hub.docker.com/r/kaixhin/mxnet/)

2. MXNet Docker (GPU) - [https://hub.docker.com/r/kaixhin/cuda-mxnet/](https://hub.docker.com/r/kaixhin/cuda-mxnet/)

这些镜像会每周更新到 MXNet 的最新版。如果要支持 CUDA，你需要 [NVIDIA Docker镜像](https://github.com/NVIDIA/nvidia-docker).

在 Docker 上运行 MXNet:

1. 在电脑上安装 Docker，更详细的信息参考 [Docker documentation](https://docs.docker.com/engine/installation/).
2. 用命令来启动 MXNet Docker 容器.

	对于 CPU 版的容器，使用这个命令:

	```bash
		sudo docker run -it kaixhin/mxnet
	```

	对于 GPU 版的容器，使用这个命令:

	```bash
		sudo nvidia-docker run -it kaixhin/cuda-mxnet:7.0
	```

在 Docker 如何使用 MXNet 更详细的文档可以参考 [MXNet Docker GitHub documentation](https://github.com/Kaixhin/dockerfiles).

# 下一步
* [教程](http://mxnet.io/tutorials/index.html)
* [如何使用](http://mxnet.io/how_to/index.html)
* [架构设计](http://mxnet.io/architecture/index.html)
