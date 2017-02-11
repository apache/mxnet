# 在云服务器上安装 MXNet

你想在云服务器上安装使用 MXnet，参考下面的详细介绍。

## 使用 Amazon Machine Images(AMIs) 安装 MXNet

这里有一个链接，链接到AWS博客，在博客里有如何安装 Amazon Machine Image(AMI) 的详细图解，AMI 可以同时支持MXNet和其他流行深度学习框架 。

* [P2 and Deep Learning Blog](https://aws.amazon.com/blogs/aws/new-p2-instance-type-for-amazon-ec2-up-to-16-gpus/)
* [Deep Learning AMI](https://aws.amazon.com/marketplace/pp/B01M0AXXQB)

或者你可以使用 Bitfusion's MXNet AMI，它预装了多种深度学习和数据科学库的框架，以及 Jupyter 笔记实例源码
* [Bitfusion MXNet AMI](https://aws.amazon.com/marketplace/pp/B01NBF5O1N/ref=_ptnr_docs_mxnet)

## 在 AWS 上多实例使用 MXNet

可以使用 CloudFormation 模板来扩充AWS GPU 实例，参考下面链接的介绍.
* [CloudFormation Template AWS Blog](https://aws.amazon.com/blogs/compute/distributed-deep-learning-made-easy/)

# 下一步
* [教程](http://mxnet.io/tutorials/index.html)
* [如何使用](http://mxnet.io/how_to/index.html)
* [架构设计](http://mxnet.io/architecture/index.html)
