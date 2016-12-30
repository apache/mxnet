# Run MXNet on Docker

[Docker](http://docker.com/) is a system that lets you build self-contained versions of a Linux operating system that can run in isolation on your computer. On the self-contained version of Linux, you can run MXNet and other software packages without them interacting with the packages on your computer.

MXNet provides two Docker images for running MXNet:

1. MXNet Docker (CPU) - [https://hub.docker.com/r/kaixhin/mxnet/](https://hub.docker.com/r/kaixhin/mxnet/)

2. MXNet Docker (GPU) - [https://hub.docker.com/r/kaixhin/cuda-mxnet/](https://hub.docker.com/r/kaixhin/cuda-mxnet/)

These Docker images are updated weekly with the latest builds of MXNet.
For CUDA support, you need the [NVIDIA Docker image](https://github.com/NVIDIA/nvidia-docker).

To run MXNet on Docker:

1. Install Docker on your computer. For more information, see the [Docker documentation](https://docs.docker.com/engine/installation/).
2. Run the command to start the MXNet Docker container.

	For CPU containers, run this command:

	```bash
		sudo docker run -it kaixhin/mxnet
	```

	For GPU containers, run this command:

	```bash
		sudo nvidia-docker run -it kaixhin/cuda-mxnet:7.0
	```

For more details on how to use the MXNet Docker images, see the [MXNet Docker GitHub documentation](https://github.com/Kaixhin/dockerfiles).

# Next Steps

* [Tutorials](http://mxnet.io/tutorials/index.html)
* [How To](http://mxnet.io/how_to/index.html)
* [Architecture](http://mxnet.io/architecture/index.html)
