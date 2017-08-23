import os
from PIL import Image

import numpy as np
import mxnet as mx
import mxnet.ndarray as F

def tensor_load_rgbimage(filename, ctx, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1).astype(float)
    img = F.expand_dims(mx.nd.array(img, ctx=ctx), 0)
    return img


def tensor_save_rgbimage(img, filename, cuda=False):
    img = F.clip(img, 0, 255).asnumpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = F.split(tensor, num_outputs=3, axis=0)
    tensor = F.concat(r, g, b, dim=0)
    tensor_save_rgbimage(tensor, filename, cuda)


def subtract_imagenet_mean_batch(batch):
    """Subtract ImageNet mean pixel-wise from a BGR image."""
    batch = F.swapaxes(batch,0, 1)
    (r, g, b) = F.split(batch, num_outputs=3, axis=0)
    r = r - 123.680
    g = g - 116.779
    b = b - 103.939
    batch = F.concat(r, g, b, dim=0)
    batch = F.swapaxes(batch,0, 1)
    return batch


def subtract_imagenet_mean_preprocess_batch(batch):
    """Subtract ImageNet mean pixel-wise from a BGR image."""
    batch = F.swapaxes(batch,0, 1)
    (r, g, b) = F.split(batch, num_outputs=3, axis=0)
    r = r - 123.680
    g = g - 116.779
    b = b - 103.939
    batch = F.concat(b, g, r, dim=0)
    batch = F.swapaxes(batch,0, 1)
    return batch


def add_imagenet_mean_batch(batch):
    batch = F.swapaxes(batch,0, 1)
    (b, g, r) = F.split(batch, num_outputs=3, axis=0)
    r = r + 123.680
    g = g + 116.779
    b = b + 103.939
    batch = F.concat(b, g, r, dim=0)
    batch = F.swapaxes(batch,0, 1)
    """
    batch = denormalizer(batch)
    """
    return batch


def imagenet_clamp_batch(batch, low, high):
    """ Not necessary in practice """
    F.clip(batch[:,0,:,:],low-123.680, high-123.680)
    F.clip(batch[:,1,:,:],low-116.779, high-116.779)
    F.clip(batch[:,2,:,:],low-103.939, high-103.939)


def preprocess_batch(batch):
    batch = F.swapaxes(batch, 0, 1)
    (r, g, b) = F.split(batch, num_outputs=3, axis=0)
    batch = F.concat(b, g, r, dim=0)
    batch = F.swapaxes(batch, 0, 1)
    return batch


class ToTensor(object):
    def __init__(self, ctx):
        self.ctx = ctx

    def __call__(self, img):
        img = mx.nd.array(np.array(img).transpose(2, 0, 1).astype('float32'), ctx=self.ctx)
        return img


class StyleLoader():
    def __init__(self, style_folder, style_size, ctx):
        self.folder = style_folder
        self.style_size = style_size
        self.files = os.listdir(style_folder)
        assert(len(self.files) > 0)
        self.ctx = ctx
    
    def get(self, i):
        idx = i%len(self.files)
        filepath = os.path.join(self.folder, self.files[idx])
        style = tensor_load_rgbimage(filepath, self.ctx, self.style_size)
        return style

    def size(self):
        return len(self.files)

def init_vgg_params(vgg, model_folder, ctx):
    if not os.path.exists(os.path.join(model_folder, 'mxvgg.params')):
        os.system('wget https://www.dropbox.com/s/7c92s0guekwrwzf/mxvgg.params?dl=1 -O' + os.path.join(model_folder, 'mxvgg.params'))
    vgg.collect_params().load(os.path.join(model_folder, 'mxvgg.params'), ctx=ctx)
