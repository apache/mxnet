import numpy as np
from easydict import EasyDict as edict

config = edict()

config.EPS = 1e-14
config.PIXEL_MEANS = np.array([[[123.68, 116.779, 103.939]]])

config.TRAIN = edict()

config.TRAIN.SCALES = (600, )
config.TRAIN.MAX_SIZE = 1000

config.TRAIN.BATCH_IMAGES = 2
config.TRAIN.BATCH_SIZE = 128
config.TRAIN.FG_FRACTION = 0.25
config.TRAIN.FG_THRESH = 0.5
config.TRAIN.BG_THRESH_HI = 0.5
config.TRAIN.BG_THRESH_LO = 0.1

config.TRAIN.BBOX_REGRESSION_THRESH = 0.5
config.TRAIN.BBOX_INSIDE_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0])

config.TEST = edict()

config.TEST.SCALES = (600, )
config.TEST.NMS = 0.3
config.TEST.DEDUP_BOXES = 1. / 16.
