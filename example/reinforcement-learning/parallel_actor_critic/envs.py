import numpy as np


class Atari8080Preprocessor(object):
    def __init__(self):
        self.prev = None
        self.obs_size = 80*80

    def reset(self):
        self.prev = None

    def preprocess(self, img):
        """
        Preprocess a 210x160x3 uint8 frame into a 6400 (80x80) (1 x input_size)
        float vector.
        """
        # Crop, down-sample, erase background and set foreground to 1.
        # See https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
        img = img[35:195]
        img = img[::2, ::2, 0]
        img[img == 144] = 0
        img[img == 109] = 0
        img[img != 0] = 1
        curr = np.expand_dims(img.astype(np.float).ravel(), axis=0)
        # Subtract the last preprocessed image.
        diff = (curr - self.prev if self.prev is not None
                else np.zeros((1, curr.shape[1])))
        self.prev = curr
        return diff


class IdentityPreprocessor(object):
    def __init__(self, obs_size):
        self.obs_size = obs_size

    def reset(self):
        pass

    def preprocess(self, x):
        return x
