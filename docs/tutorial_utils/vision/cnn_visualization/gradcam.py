# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import print_function

import mxnet as mx
import mxnet.ndarray as nd

from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import nn

import numpy as np
import cv2

class ReluOp(mx.operator.CustomOp):
    """Modified ReLU as described in section 3.4 in https://arxiv.org/abs/1412.6806.
    This is used for guided backpropagation to get gradients of the image w.r.t activations.
    This Operator will do a regular backpropagation if `guided_backprop` is set to False
    and a guided packpropagation if `guided_backprop` is set to True. Check gradcam_demo.py
    for an example usage."""

    guided_backprop = False

    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        y = nd.maximum(x, nd.zeros_like(x))
        self.assign(out_data[0], req[0], y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        if ReluOp.guided_backprop:
            # Get output and gradients of output
            y = out_data[0]
            dy = out_grad[0]
            # Zero out the negatives in the gradients of the output
            dy_positives = nd.maximum(dy, nd.zeros_like(dy))
            # What output values were greater than 0?
            y_ones = y.__gt__(0)
            # Mask out the values for which at least one of dy or y is negative
            dx = dy_positives * y_ones
            self.assign(in_grad[0], req[0], dx)
        else:
            # Regular backward for ReLU
            x = in_data[0]
            x_gt_zero = x.__gt__(0)
            dx = out_grad[0] * x_gt_zero
            self.assign(in_grad[0], req[0], dx)

def set_guided_backprop(mode=True):
    ReluOp.guided_backprop = mode

@mx.operator.register("relu")
class ReluProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(ReluProp, self).__init__(True)

    def infer_shape(self, in_shapes):
        data_shape = in_shapes[0]
        output_shape = data_shape
        return (data_shape,), (output_shape,), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return ReluOp()  

class Activation(mx.gluon.HybridBlock):
    @staticmethod
    def set_guided_backprop(mode=False):
        ReluOp.guided_backprop = mode

    def __init__(self, act_type, **kwargs):
        assert act_type == 'relu'
        super(Activation, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.Custom(x, op_type='relu')

class Conv2D(mx.gluon.HybridBlock):
    """Wrapper on top of gluon.nn.Conv2D to capture the output and gradients of output of a Conv2D
    layer in a network. Use `set_capture_layer_name` to select the layer
    whose outputs and gradients of outputs need to be captured. After the backward pass,
    `conv_output` will contain the output and `conv_output.grad` will contain the
    output's gradients. Check gradcam_demo.py for example usage."""

    conv_output = None
    capture_layer_name = None

    def __init__(self, channels, kernel_size, strides=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, layout='NCHW',
                 activation=None, use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.conv = nn.Conv2D(channels, kernel_size, strides=strides, padding=padding,
                             dilation=dilation, groups=groups, layout=layout,
                             activation=activation, use_bias=use_bias, weight_initializer=weight_initializer,
                             bias_initializer=bias_initializer, in_channels=in_channels)

    def hybrid_forward(self, F, x):
        out = self.conv(x)
        name = self._prefix[:-1]
        if name == Conv2D.capture_layer_name:
            out.attach_grad()
            Conv2D.conv_output = out
        return out

def set_capture_layer_name(name):
    Conv2D.capture_layer_name = name

def _get_grad(net, image, class_id=None, conv_layer_name=None, image_grad=False):
    """This is an internal helper function that can be used for either of these
    but not both at the same time:
    1. Record the output and gradient of output of an intermediate convolutional layer.
    2. Record the gradients of the image.

    Parameters
    ----------
    image : NDArray
        Image to visuaize. This is an NDArray with the preprocessed image.
    class_id : int
        Category ID this image belongs to. If not provided,
        network's prediction will be used.
    conv_layer_name: str
        Name of the convolutional layer whose output and output's gradients need to be acptured.
    image_grad: bool
        Whether to capture gradients of the image."""

    if image_grad:
        image.attach_grad()
        Conv2D.capture_layer_name = None
        Activation.set_guided_backprop(True)
    else:
        # Tell convviz.Conv2D which layer's output and gradient needs to be recorded
        Conv2D.capture_layer_name = conv_layer_name
        Activation.set_guided_backprop(False)
    
    # Run the network
    with autograd.record(train_mode=False):
        out = net(image)
    
    # If user didn't provide a class id, we'll use the class that the network predicted
    if class_id == None:
        model_output = out.asnumpy()
        class_id = np.argmax(model_output)

    # Create a one-hot target with class_id and backprop with the created target
    one_hot_target = mx.nd.one_hot(mx.nd.array([class_id]), 1000)
    out.backward(one_hot_target, train_mode=False)

    if image_grad:
        return image.grad[0].asnumpy()
    else:
        # Return the recorded convolution output and gradient
        conv_out = Conv2D.conv_output
        return conv_out[0].asnumpy(), conv_out.grad[0].asnumpy()

def get_conv_out_grad(net, image, class_id=None, conv_layer_name=None):
    """Get the output and gradients of output of a convolutional layer.

    Parameters:
    ----------
    net: Block
        Network to use for visualization.
    image: NDArray
        Preprocessed image to use for visualization.
    class_id: int
        Category ID this image belongs to. If not provided,
        network's prediction will be used.
    conv_layer_name: str
        Name of the convolutional layer whose output and output's gradients need to be acptured."""
    return _get_grad(net, image, class_id, conv_layer_name, image_grad=False)

def get_image_grad(net, image, class_id=None):
    """Get the gradients of the image.

    Parameters:
    ----------
    net: Block
        Network to use for visualization.
    image: NDArray
        Preprocessed image to use for visualization.
    class_id: int
        Category ID this image belongs to. If not provided,
        network's prediction will be used."""
    return _get_grad(net, image, class_id, image_grad=True)

def grad_to_image(gradient):
    """Convert gradients of image obtained using `get_image_grad`
    into image. This shows parts of the image that is most strongly activating
    the output neurons."""
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    gradient = np.uint8(gradient * 255).transpose(1, 2, 0)
    gradient = gradient[..., ::-1]
    return gradient

def get_cam(imggrad, conv_out):
    """Compute CAM. Refer section 3 of https://arxiv.org/abs/1610.02391 for details"""
    weights = np.mean(imggrad, axis=(1, 2))
    cam = np.ones(conv_out.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_out[i, :, :]
    cam = cv2.resize(cam, (imggrad.shape[1], imggrad.shape[2]))
    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam)) 
    cam = np.uint8(cam * 255)
    return cam

def get_guided_grad_cam(cam, imggrad):
    """Compute Guided Grad-CAM. Refer section 3 of https://arxiv.org/abs/1610.02391 for details"""
    return np.multiply(cam, imggrad)

def get_img_heatmap(orig_img, activation_map):
    """Draw a heatmap on top of the original image using intensities from activation_map"""
    heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_COOL)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_heatmap = np.float32(heatmap) + np.float32(orig_img)
    img_heatmap = img_heatmap / np.max(img_heatmap)
    img_heatmap *= 255
    return img_heatmap.astype(int)

def to_grayscale(cv2im):
    """Convert gradients to grayscale. This gives a saliency map."""
    # How strongly does each position activate the output
    grayscale_im = np.sum(np.abs(cv2im), axis=0)

    # Normalize between min and 99th percentile
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1)

    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im

def visualize(net, preprocessed_img, orig_img, conv_layer_name):
    # Returns grad-cam heatmap, guided grad-cam, guided grad-cam saliency
    imggrad = get_image_grad(net, preprocessed_img)
    conv_out, conv_out_grad = get_conv_out_grad(net, preprocessed_img, conv_layer_name=conv_layer_name)

    cam = get_cam(conv_out_grad, conv_out)
    cam = cv2.resize(cam, (imggrad.shape[1], imggrad.shape[2]))
    ggcam = get_guided_grad_cam(cam, imggrad)
    img_ggcam = grad_to_image(ggcam)
    
    img_heatmap = get_img_heatmap(orig_img, cam)
    
    ggcam_gray = to_grayscale(ggcam)
    img_ggcam_gray = np.squeeze(grad_to_image(ggcam_gray))
    
    return img_heatmap, img_ggcam, img_ggcam_gray

