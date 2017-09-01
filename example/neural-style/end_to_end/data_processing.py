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

import numpy as np
from skimage import io, transform
from skimage.restoration import denoise_tv_chambolle
import logging
import random
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)

def PreprocessContentImage(path, short_edge, dshape=None):
    img = io.imread(path)
    #logging.info("load the content image, size = %s", img.shape[:2])
    factor = float(short_edge) / min(img.shape[:2])
    new_size = (int(img.shape[0] * factor), int(img.shape[1] * factor))
    resized_img = transform.resize(img, new_size)
    sample = np.asarray(resized_img) * 256
    if dshape is not None:
        # random crop
        xx = int((sample.shape[0] - dshape[2]))
        yy = int((sample.shape[1] - dshape[3]))
        xstart = random.randint(0, xx)
        ystart = random.randint(0, yy)
        xend = xstart + dshape[2]
        yend = ystart + dshape[3]
        sample = sample[xstart:xend, ystart:yend, :]

    # swap axes to make image from (224, 224, 3) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # sub mean
    sample[0, :] -= 123.68
    sample[1, :] -= 116.779
    sample[2, :] -= 103.939
    #logging.info("resize the content image to %s", sample.shape)
    return np.resize(sample, (1, 3, sample.shape[1], sample.shape[2]))

def PreprocessStyleImage(path, shape):
    img = io.imread(path)
    resized_img = transform.resize(img, (shape[2], shape[3]))
    sample = np.asarray(resized_img) * 256
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)

    sample[0, :] -= 123.68
    sample[1, :] -= 116.779
    sample[2, :] -= 103.939
    return np.resize(sample, (1, 3, sample.shape[1], sample.shape[2]))

def PostprocessImage(img):
    img = np.resize(img, (3, img.shape[2], img.shape[3]))
    img[0, :] += 123.68
    img[1, :] += 116.779
    img[2, :] += 103.939
    img = np.swapaxes(img, 1, 2)
    img = np.swapaxes(img, 0, 2)
    img = np.clip(img, 0, 255)
    return img.astype('uint8')

def SaveImage(img, filename, remove_noise=0.02):
    logging.info('save output to %s', filename)
    out = PostprocessImage(img)
    if remove_noise != 0.0:
        out = denoise_tv_chambolle(out, weight=remove_noise, multichannel=True)
    io.imsave(filename, out)




