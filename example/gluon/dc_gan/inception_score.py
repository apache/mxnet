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

from mxnet.gluon.model_zoo import vision as models
import mxnet as mx
from mxnet import nd
import numpy as np
import math
import sys

import cv2


inception_model = None


def get_inception_score(images, splits=10):
    """
    Inception_score function.
        The images will be divided into 'splits' parts, and calculate each inception_score separately,
        then return the mean and std of inception_scores of these parts.
    :param images: Images(num x c x w x h) that needs to calculate inception_score.
    :param splits:
    :return: mean and std of inception_score
    """
    assert (images.shape[1] == 3)

    # load inception model
    if inception_model is None:
        _init_inception()

    # resize images to adapt inception model(inceptionV3)
    if images.shape[2] != 299:
        images = resize(images, 299, 299)

    preds = []
    bs = 4
    n_batches = int(math.ceil(float(images.shape[0])/float(bs)))

    # to get the predictions/picture of inception model
    for i in range(n_batches):
        sys.stdout.write(".")
        sys.stdout.flush()
        inps = images[(i * bs):min((i + 1) * bs, len(images))]
        # inps size. bs x 3 x 299 x 299
        pred = nd.softmax(inception_model(inps))
        # pred size. bs x 1000
        preds.append(pred.asnumpy())

    # list to array
    preds = np.concatenate(preds, 0)
    scores = []

    # to calculate the inception_score each split.
    for i in range(splits):
        # extract per split image pred
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))

    return np.mean(scores), np.std(scores)


def _init_inception():
    global inception_model
    inception_model = models.inception_v3(pretrained=True)
    print("success import inception model, and the model is inception_v3!")


def resize(images, w, h):
    nums = images.shape[0]
    res = nd.random.uniform(0, 255, (nums, 3, w, h))
    for i in range(nums):
        img = images[i, :, :, :]
        img = mx.nd.transpose(img, (1, 2, 0))
        # Replace 'mx.image.imresize()' with 'cv2.resize()' because : Operator _cvimresize is not implemented for GPU.
        # img = mx.image.imresize(img, w, h)
        img = cv2.resize(img.asnumpy(), (299, 299))
        img = nd.array(img)
        img = mx.nd.transpose(img, (2, 0, 1))
        res[i, :, :, :] = img

    return res


if __name__ == '__main__':
    if inception_model is None:
        _init_inception()
    # dummy data
    images = nd.random.uniform(0, 255, (64, 3, 64, 64))
    print(images.shape[0])
    # resize(images,299,299)

    score = get_inception_score(images)
    print(score)
