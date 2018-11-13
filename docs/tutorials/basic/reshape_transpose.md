
## Difference between reshape and transpose operators
Modyfing the shape of tensors is a very common operation in Deep Learning. For instance, when using pretrained neural networks it is often required to adjust input data dimensions to correspond to what the network has been trained on, e.g. tensors of shape `[batch_size, channels, width, height]`.  This notebook discusses briefly the difference between the operators `Reshape` and `Transpose`. Both allow to change the shape, however they are not the same and are commonly mistaken.


```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

```


```python
img_array = mpimg.imread('https://raw.githubusercontent.com/NRauschmayr/web-data/tutorial_transpose_reshape/mxnet/doc/tutorials/basic/transpose_reshape/cat.png')
plt.imshow(img_array)
plt.axis("off")
print (img_array.shape)
```

(210, 200, 3) <!--notebook-skip-line-->

![png](https://raw.githubusercontent.com/NRauschmayr/web-data/tutorial_transpose_reshape/mxnet/doc/tutorials/basic/transpose_reshape/cat.png) <!--notebook-skip-line-->


The color image has the following properties:
* width: 210 pixels
* height: 200 pixels
* colors: 3 (RGB)

Now lets reshape the image in order to exchange width and height dimension.


```python
reshaped = img_array.reshape((200,210,3))
print (reshaped.shape)
plt.imshow(reshaped)
plt.axis("off")
```
(200, 210, 3)<!--notebook-skip-line-->

![png](https://raw.githubusercontent.com/NRauschmayr/web-data/tutorial_transpose_reshape/mxnet/doc/tutorials/basic/transpose_reshape/reshaped_image.png) <!--notebook-skip-line-->


As we can see the first and second dimensions have changed. However the image can't be identified as cat anylonger. In order to understand what happened, let's have a look at the image below.

<img src="https://raw.githubusercontent.com/NRauschmayr/web-data/tutorial_transpose_reshape/mxnet/doc/tutorials/basic/transpose_reshape/reshape.png" style="width:700px;height:300px;">

While the number of rows and columns changed, the layout of the underlying data did not. The pixel values that have been in one row are still in one row. This means for instance that pixel 10 in the upper right corder ends up in the middle of the image instead of the lower left corner. Consequently contextual information gets lost, because the relative position of pixel values is not the same anymore. As one can imagine a neural network would not be able to classify such an image as cat. 

`Transpose` instead changes the layout of the underlying data.


```python
transposed = img_array.transpose((1,0,2))
plt.imshow(transposed)
plt.axis("off")
```

![png](https://raw.githubusercontent.com/NRauschmayr/web-data/tutorial_transpose_reshape/mxnet/doc/tutorials/basic/transpose_reshape/transposed_image.png) <!--notebook-skip-line-->


As we can see width and height changed, by rotating pixel values by 90 degrees. Transpose does the following:

<img src="https://raw.githubusercontent.com/NRauschmayr/web-data/tutorial_transpose_reshape/mxnet/doc/tutorials/basic/transpose_reshape/transpose.png" style="width:700px;height:300px;">

As shown in the diagram, the axis have been flipped: pixel values that have been in the first row are now in the first column.
## When to transpose/reshape with MXNet
In this chapter we discuss when transpose and reshape is used in MXNet. 
#### Channel first for images
Images are usually stored in the format height, wight, channel. When working with [convolutional](https://mxnet.incubator.apache.org/api/python/gluon/nn.html#mxnet.gluon.nn.Conv1D) layers, MXNet expects the layout to be `NCHW` (batch, channel, height, width). MXNet uses this layout because of performance reasons on the GPU. Consequently, images need to be transposed to have the right format. For instance, you may have a function like the following:
``` 
def transform(data, label): 
     return data.astype(np.float32).transpose((2,0,1))/255.0, np.float32(label)
```
Images may also be stored as 1 dimensional vector for example in byte packed datasets. For instance, instead of `[28,28,1]` you may have `[784,1]`. In this situation you need to perform a reshape e.g. `ndarray.reshape((1,28,28))`


#### TNC layout for RNN
When working with [LSTM](https://mxnet.incubator.apache.org/api/python/gluon/rnn.html#mxnet.gluon.rnn.LSTM) or [GRU](https://mxnet.incubator.apache.org/api/python/gluon/rnn.html#mxnet.gluon.rnn.GRU) layers, the default layout for input and ouput tensors has to be `TNC` (sequence length, batch size, and feature dimensions). For instance in the following network the input goes into a 1 dimensional convolution layer and whose output goes into a GRU cell. Here the tensors would mismatch, because `Conv1D` takes data as `NCT`, but GRU  expects it to be `NTC`. To ensure that the forward pass does not crash, we need to do a tensor transpose. We can do this by defining a ```HybridLambda```.
```
network = gluon.nn.HybridSequential()
with network.name_scope():
       network.add(gluon.nn.Conv1D(196, kernel_size=2, strides=1))
       network.add(gluon.nn.HybridLambda(lambda F, x: F.transpose(x, (0, 2, 1))))
       network.add(gluon.rnn.GRU(128))

network.hybridize()
network.initialize(mx.init.Xavier(), ctx=mx.cpu())
a = mx.random.uniform(shape=(1,100,1000))
network(a)
```
#### Advanced reshaping with MXNet ndarrays
It is sometimes useful to automatically infer the shape of tensors. Especially when you deal with very deep neural networks, it may not always be clear what the shape of a tensor is after a specific layer. For instance you may want the tensor to be two-dimensional where one dimension is the known batch_size. With ```mx.nd.array(-1, batch_size)``` the first dimension will be automatically inferred. Here a simplified example:
```
batch_size = 100
input_data = mx.random.uniform(shape=(20,100,batch_size))
reshaped = input_data.reshape(-1,batch_size)
print (input_data.shape, reshaped.shape) 
```
(20L, 100L, 100L), (2000L, 100L) <!--notebook-skip-line-->

The reshape function of [MXNet's NDArray API](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html?highlight=reshape#mxnet.ndarray.NDArray.reshape) allows even more advanced transformations: For instance: with -2 you copy all/remainder of the input dimensions to the output shape. With -3 reshape will use the product of two consecutive dimensions of the input shape as the output dim.    

#### Check out the MXNet documentation for more details
http://mxnet.incubator.apache.org/test/api/python/ndarray.html#mxnet.ndarray.NDArray.reshape
http://mxnet.incubator.apache.org/test/api/python/ndarray.html#mxnet.ndarray.transpose
<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
