from mxnet.gluon.model_zoo import vision as models
from mxnet import gluon
import mxnet as mx
from mxnet import nd
import numpy as np
import math
import sys


inception_model = None


def get_inception_score(images, splits=10):

    assert (images.shape[1] == 3)
    # assert (nd.max(images[0]) > 10)
    # assert (nd.min(images[0]) >= 0.0)

    # load inception model
    if inception_model is None:
        _init_inception()

    if images.shape[2] != 299:
        images = resize(images, 299, 299)

    preds = []
    bs = 4
    # scores = []
    n_batches = int(math.ceil(float(images.shape[0])/float(bs)))

    # to get the predictions/picture of inception model
    for i in range(n_batches):
        sys.stdout.write(".")
        sys.stdout.flush()
        inps = images[(i * bs):min((i + 1) * bs, len(images))]
        pred = nd.softmax(inception_model(inps))
        preds.append(pred.asnumpy())

    # list to array
    preds = np.concatenate(preds, 0)
    scores = []

    # to calculate the inception_score each split of 10 splits
    for i in range(splits):
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
        img = mx.image.imresize(img, 299, 299)
        img = mx.nd.transpose(img, (2, 0, 1))
        res[i, :, :, :] = img

    return res


if __name__ == '__main__':
    if inception_model is None:
        _init_inception()
    # dummy data
    images = nd.random.uniform(0, 255, (15, 3, 64, 64))
    print(images.shape[0])
    # resize(images,299,299)

    score = get_inception_score(images)
    print(score)
