MXNet is available on several cloud providers with GPU support. You can also find
GPU/CPU-hybrid support for
use cases like scalable inference, or even fractional GPU support with AWS Elastic
Inference.

* **Alibaba**
- [NVIDIA
VM](https://docs.nvidia.com/ngc/ngc-alibaba-setup-guide/launching-nv-cloud-vm-console.html#launching-nv-cloud-vm-console)
* **Amazon Web Services**
- [Amazon SageMaker](https://aws.amazon.com/sagemaker/) - Managed training and deployment of
MXNet models
- [AWS Deep Learning AMI](https://aws.amazon.com/machine-learning/amis/) - Preinstalled
Conda environments
for Python 2 or 3 with MXNet, CUDA, cuDNN, MKL-DNN, and AWS Elastic Inference
- [Dynamic Training on
AWS](https://github.com/awslabs/dynamic-training-with-apache-mxnet-on-aws) -
experimental manual EC2 setup or semi-automated CloudFormation setup
- [NVIDIA VM](https://aws.amazon.com/marketplace/pp/B076K31M1S)
* **Google Cloud Platform**
- [NVIDIA
VM](https://console.cloud.google.com/marketplace/details/nvidia-ngc-public/nvidia_gpu_cloud_image)
* **Microsoft Azure**
- [NVIDIA
VM](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/nvidia.ngc_azure_17_11?tab=Overview)
* **Oracle Cloud**
- [NVIDIA VM](https://docs.cloud.oracle.com/iaas/Content/Compute/References/ngcimage.htm)

All NVIDIA VMs use the [NVIDIA MXNet Docker
container](https://ngc.nvidia.com/catalog/containers/nvidia:mxnet).
Follow the [container usage
instructions](https://ngc.nvidia.com/catalog/containers/nvidia:mxnet) found in
[NVIDIA's container repository](https://ngc.nvidia.com/).