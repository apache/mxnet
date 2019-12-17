Docker images with *MXNet* are available at [DockerHub](https://hub.docker.com/r/mxnet/).

**Step 1** Install Docker on your machine by following the [docker installation
instructions](https://docs.docker.com/engine/installation/linux/ubuntu/#install-using-the-repository).

*Note* - You can install Community Edition (CE) to get started with *MXNet*.

**Step 2** [Optional] Post installation steps to manage Docker as a non-root user.

Follow the four steps in this [docker
documentation](https://docs.docker.com/engine/installation/linux/linux-postinstall/#manage-docker-as-a-non-root-user)
to allow managing docker containers without *sudo*.

If you skip this step, you need to use *sudo* each time you invoke Docker.

**Step 3** Install *nvidia-docker-plugin* following the [installation
instructions](https://github.com/NVIDIA/nvidia-docker/wiki). *nvidia-docker-plugin*
is required to
enable the usage of GPUs from the docker containers.

**Step 4** Pull the MXNet docker image.

{% highlight bash %}
$ docker pull mxnet/python:gpu # Use sudo if you skip Step 2
{% endhighlight %}

You can list docker images to see if mxnet/python docker image pull was successful.

{% highlight bash %}
$ docker images # Use sudo if you skip Step 2

REPOSITORY TAG IMAGE ID CREATED SIZE
mxnet/python gpu 493b2683c269 3 weeks ago 4.77 GB
{% endhighlight %}

Using the latest MXNet with [Intel MKL-DNN](https://github.com/intel/mkl-dnn) is
recommended for the
fastest inference speeds with MXNet.

{% highlight bash %}
$ docker pull mxnet/python:1.3.0_cpu_mkl # Use sudo if you skip Step 2
$ docker images # Use sudo if you skip Step 2

REPOSITORY TAG IMAGE ID CREATED SIZE
mxnet/python 1.3.0_gpu_cu92_mkl adcb3ab19f50 4 days ago 4.23 GB
{% endhighlight %}

**Step 5** <a href="get_started/validate_mxnet">Validate the installation</a>.