## Import ONNX model into Mxnet

The code demonstration in this document assumes that the following packages are installed and imported.

```python
from PIL import Image
import numpy as np
import mxnet as mx
import mxnet.contrib.onnx as onnx_mxnet
```

The PIL package is a Python Image Processing package and is required for input preprocessing. It can be installed as follows

```python
pip install Pillow
```

We will now try to import a [Super_Resolution model](http://pytorch.org/tutorials/advanced/super_resolution_with_caffe2.html), trained with PyTorch, and run inference in MXNet. PyTorch provides a way to export models in ONNX protobuf format. Using this functionality, we have exported the model into ONNX format.

You can download the converted ONNX model from
[here](https://s3.amazonaws.com/onnx-mxnet/examples/super_resolution.onnx).

A pre-trained model in MXNet contains two elements: a symbolic graph, containing the model's network definition, and a binary file containing the model weights. You can import the ONNX model and get the symbol and parameters objects using "import_model" API as shown below:

```python
sym, params = onnx_mxnet.import_model('super_resolution.onnx')
```

To run inference on the imported mxnet model, you need to use MXNet's [Module API](https://mxnet.incubator.apache.org/api/python/module.html), following these steps:

- Input image preprocessing

For the input image pre-process step, we will download and transform the image into an input tensor:

```python
img_url = 'https://s3.amazonaws.com/onnx-mxnet/examples/super_res_input.jpg'
download(img_url, 'super_res_input.jpg')
img = Image.open('super_res_input.jpg').resize((224, 224))
img_ycbcr = img.convert("YCbCr")
img_y, img_cb, img_cr = img_ycbcr.split()
test_image = np.array(img_y)[np.newaxis, np.newaxis, :, :]
```
- We'll be using MXNet's Module API to create the module, bind it and assign the loaded weights.

```
# By default, 'input_0' is an input of the imported model.
mod = mx.mod.Module(symbol=sym, data_names=['input_0'], context=mx.cpu(), label_names=None)
mod.bind(for_training=False, data_shapes=[('input_0',test_image.shape)], label_shapes=None)
mod.set_params(arg_params=params, aux_params=params, allow_missing=True, allow_extra=True)
```

- Run inference
```
# Forward method needs Batch of data as input
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

# forward on the provided data batch
mod.forward(Batch([mx.nd.array(test_image)]))
```

- To get the output of previous forward computation, use "module.get_outputs()" method.
It returns ndarray that we convert to numpy array, create and save the super resolution image:
```
output = mod.get_outputs()[0][0][0]
img_out_y = Image.fromarray(np.uint8((output.asnumpy().clip(0, 255)), mode='L'))
result_img = Image.merge(
"YCbCr", [
        	img_out_y,
        	img_cb.resize(img_out_y.size, Image.BICUBIC),
        	img_cr.resize(img_out_y.size, Image.BICUBIC)
]).convert("RGB")
result_img.save("super_res_output.jpg")

```

Here's the input image and the resulting output images compared. As you can see, the model was able to increase the spatial resolution from 256x256 to 672x672.

| Input Image | Output Image |
| ----------- | ------------ |
| ![input](https://s3.amazonaws.com/onnx-mxnet/examples/super_res_input.jpg) | ![output](https://s3.amazonaws.com/onnx-mxnet/examples/super_res_expected_output.jpg) |

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->