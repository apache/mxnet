# Model Server for Apache MXNet (incubating)

Model Server for Apache MXNet (incubating), otherwise known as MXNet Model Server (MMS), is an open source project aimed at providing a simple yet scalable solution for model inference. It is a set of command line tools for packaging model archives and serving them. The tools are written in Python, and have been extended to support containers for easy deployment and scaling. MMS also supports basic logging and advanced logging with AWS CloudWatch integration.

While the name implies that MMS is just for MXNet, it is in fact much more flexible, as it can support models in the [ONNX](#) format. This means that models created and trained in PyTorch, Caffe2, or other ONNX-supporting frameworks can be served with MMS.

To install MMS with ONNX support, make sure you have Python installed, then for Ubuntu run:

```bash
sudo apt-get install protobuf-compiler libprotoc-dev
pip install mxnet-model-server
```

Or for Mac run:

```bash
conda install -c conda-forge protobuf
pip install mxnet-model-server
```


## Serving a Model

To serve a model you must first create or download a model archive. Visit the model zoo to browse the free models. MMS options can be explored as follows:

```bash
mxnet-model-server --help
```

Here is an easy example for serving an object classification model. You can use any URI and the model will be downloaded first, then served from that location:

```bash
mxnet-model-server \
  --models squeezenet=https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model
```


## Test Inference on a Model

Assuming you have run the previous `mxnet-model-server` command to start serving the object classification model, you can now upload an image to its `predict` REST API endpoint. The following will download a picture of a kitten, then upload it to the prediction endpoint.

```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl -X POST http://127.0.0.1:8080/squeezenet/predict -F "data=@kitten.jpg"
```

For more examples of serving models visit the following resources:

## Create a Model Archive

Creating a model archive involves rounding up the required model artifacts, then using the `mxnet-model-export` command line interface. As the process for creating archives is likely to evolve as the project adds features it is recommended that you review the following resources to get the latest instructions:

* [Quickstart: Export a Model](#)
* [Model Artifacts](#)
* [Saving a Model from Gluon](#)
* [Creating a MMS Model Archive from an ONNX Model](#)


## Using Containers

Using Docker or other container services with MMS is a great way to scale your inference applications. It is recommended that you review the following resources for more information:

* [Docker Quickstart](#)
* [MMS on Fargate](#)
* [Optimized Configurations](#)


## Community & Contributions

The MMS project is open to contributions from the community. If you like the idea of a simple serving solution for your models and would like to provide feedback, suggest features, or even jump in and contribute code or examples, please visit the [project page on GitHub](). You can create an issue there, or join the discussion on the forum.

* [MXNet Forum - MMS Discussions](#)


## Further Reading

* [GitHub](#)
* [MMS Docs]()
