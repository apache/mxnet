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

# This example is inspired by https://github.com/jason71995/Keras-GAN-Library,
# https://github.com/kazizzad/DCGAN-Gluon-MxNet/blob/master/MxnetDCGAN.ipynb
# https://github.com/apache/incubator-mxnet/blob/master/example/gluon/dc_gan/dcgan.py

import math

import numpy as np
import imageio

def save_image(data, epoch, image_size, batch_size, output_dir, padding=2):
    """ save image """
    data = data.asnumpy().transpose((0, 2, 3, 1))
    datanp = np.clip(
        (data - np.min(data))*(255.0/(np.max(data) - np.min(data))), 0, 255).astype(np.uint8)
    x_dim = min(8, batch_size)
    y_dim = int(math.ceil(float(batch_size) / x_dim))
    height, width = int(image_size + padding), int(image_size + padding)
    grid = np.zeros((height * y_dim + 1 + padding // 2, width *
                     x_dim + 1 + padding // 2, 3), dtype=np.uint8)
    k = 0
    for y in range(y_dim):
        for x in range(x_dim):
            if k >= batch_size:
                break
            start_y = y * height + 1 + padding // 2
            end_y = start_y + height - padding
            start_x = x * width + 1 + padding // 2
            end_x = start_x + width - padding
            np.copyto(grid[start_y:end_y, start_x:end_x, :], datanp[k])
            k += 1
    imageio.imwrite(
        '{}/fake_samples_epoch_{}.png'.format(output_dir, epoch), grid)
