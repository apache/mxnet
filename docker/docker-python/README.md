# Release Python Docker Images for MXNet

The `docker-python` directory can be used to release mxnet python docker images to dockerhub after any mxnet release.  
It uses the appropriate pip binaries to build different docker images as -
* cpu
* cpu_mkl
* latest (same as cpu)
* gpu_cu90 
* gpu_cu90_mkl
* gpu (same as gpu_cu90)
* gpu_cu80 
* gpu_cu80_mkl
* gpu_cu92 
* gpu_cu92_mkl


** Note: If you want to use a different pip binary (specific mxnet or cuda version, etc), you can edit the last line of the cpu or gpu dockerfile as required. 

Refer: https://pypi.org/project/mxnet/

### Usage
`./build_python_dockerfile.sh <mxnet_version> <path_to_cloned_mxnet_repo>`

For example: 
`./build_python_dockerfile.sh 1.3.0 ~/build-docker/incubator-mxnet`

** Note: The build script picks up the latest pip binaries. This means it uses the latest released mxnet version. The version specified as a parameter to the script is only used to tag the built image correctly.  

### Tests run
* [test_conv.py](https://github.com/apache/incubator-mxnet/blob/master/tests/python/train/test_conv.py)
* [train_mnist.py](https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/train_mnist.py)
* [test_mxnet.py](https://github.com/apache/incubator-mxnet/blob/master/docker/docker-python/test_mxnet.py): This script is used to make sure that the docker image builds the expected mxnet version. That is, the version picked by pip is the same as as the version passed as a parameter. 

### Dockerhub Credentials
Dockerhub credentials will be required to push images at the end of this script. 
Credentials can be provided in the following ways:
* **Interactive Login:** Run the script as is and it will ask you for credentials interactively.
* **Be Already Logged in:** Login to the mxnet dockerhub account before you run the build script and the script will complete build, test and push.
* **Set Environment Variables:** Set the following environment variables which the script will pick up to login to dockerhub at runtime -
    * $MXNET_DOCKERHUB_PASSWORD
    * $MXNET_DOCKERHUB_USERNAME
