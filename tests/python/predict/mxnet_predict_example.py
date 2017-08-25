import sys, os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append("../../../amalgamation/python/")

from mxnet_predict import Predictor, load_ndarray_file
import logging
import numpy as np
from skimage import io, transform

# Load the pre-trained model
prefix = "resnet/resnet-18"
num_round = 0
symbol_file = "%s-symbol.json" % prefix
param_file = "%s-0000.params" % prefix
predictor = Predictor(open(symbol_file, "r").read(),
                      open(param_file, "rb").read(),
                      {'data':(1, 3, 224, 224)})

synset = [l.strip() for l in open('resnet/synset.txt').readlines()]

def PreprocessImage(path, show_img=False):
    # load image
    img = io.imread(path)
    print("Original Image Shape: ", img.shape)
    # we crop image from center
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
    # resize to 224, 224
    resized_img = transform.resize(crop_img, (224, 224))
    # convert to numpy.ndarray
    sample = np.asarray(resized_img) * 255
    # swap axes to make image from (224, 224, 3) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)

    # sub mean
    return sample

# Get preprocessed batch (single image batch)
batch = PreprocessImage('./download.jpg', True)

predictor.forward(data=batch)
prob = predictor.get_output(0)[0]

pred = np.argsort(prob)[::-1]
# Get top1 label
top1 = synset[pred[0]]
print("Top1: ", top1)
# Get top5 label
top5 = [synset[pred[i]] for i in range(5)]
print("Top5: ", top5)

