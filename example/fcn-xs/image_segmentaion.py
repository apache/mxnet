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

"""
This module encapsulates running image segmentation model for inference.

Example usage:
    $ python image_segmentaion.py --input <your JPG image path>
"""

import argparse
import os
import numpy as np
import mxnet as mx
from PIL import Image

def make_file_extension_assertion(extension):
    """Function factory for file extension argparse assertion
        Args:
            extension (string): the file extension to assert

        Returns:
            string: the supplied extension, if assertion is successful.

    """
    def file_extension_assertion(file_path):
        base, ext = os.path.splitext(file_path)
        if ext.lower() != extension:
            raise argparse.ArgumentTypeError('File must have ' + extension + ' extension')
        return file_path
    return file_extension_assertion

def get_palette(num_colors=256):
    """generates the colormap for visualizing the segmentation mask
            Args:
                num_colors (int): the number of colors to generate in the output palette

            Returns:
                string: the supplied extension, if assertion is successful.

    """
    pallete = [0]*(num_colors*3)
    for j in range(0, num_colors):
        lab = j
        pallete[j*3+0] = 0
        pallete[j*3+1] = 0
        pallete[j*3+2] = 0
        i = 0
        while (lab > 0):
            pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
            pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
            pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3
    return pallete

def get_data(img_path):
    """get the (1, 3, h, w) np.array data for the supplied image
                Args:
                    img_path (string): the input image path

                Returns:
                    np.array: image data in a (1, 3, h, w) shape

    """
    mean = np.array([123.68, 116.779, 103.939])  # (R,G,B)
    img = Image.open(img_path)
    img = np.array(img, dtype=np.float32)
    reshaped_mean = mean.reshape(1, 1, 3)
    img = img - reshaped_mean
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = np.expand_dims(img, axis=0)
    return img

def main():
    """Module main execution"""
    # Initialization variables - update to change your model and execution context
    model_prefix = "FCN8s_VGG16"
    epoch = 19

    # By default, MXNet will run on the CPU. Change to ctx = mx.gpu() to run on GPU.
    ctx = mx.cpu()

    fcnxs, fcnxs_args, fcnxs_auxs = mx.model.load_checkpoint(model_prefix, epoch)
    fcnxs_args["data"] = mx.nd.array(get_data(args.input), ctx)
    data_shape = fcnxs_args["data"].shape
    label_shape = (1, data_shape[2]*data_shape[3])
    fcnxs_args["softmax_label"] = mx.nd.empty(label_shape, ctx)
    exector = fcnxs.bind(ctx, fcnxs_args, args_grad=None, grad_req="null", aux_states=fcnxs_args)
    exector.forward(is_train=False)
    output = exector.outputs[0]
    out_img = np.uint8(np.squeeze(output.asnumpy().argmax(axis=1)))
    out_img = Image.fromarray(out_img)
    out_img.putpalette(get_palette())
    out_img.save(args.output)

if __name__ == "__main__":
    # Handle command line arguments
    parser = argparse.ArgumentParser(description='Run VGG16-FCN-8s to segment an input image')
    parser.add_argument('--input',
                        required=True,
                        type=make_file_extension_assertion('.jpg'),
                        help='The segmentation input JPG image')
    parser.add_argument('--output',
                        default='segmented.png',
                        type=make_file_extension_assertion('.png'),
                        help='The segmentation putput PNG image')
    args = parser.parse_args()
    main()
