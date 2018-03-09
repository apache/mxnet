import numpy as np
import os
import logging

try:
    from PIL import Image
except ImportError:
    Image = None
from ...ndarray import NDArray
from ...ndarray import ndarray as nd
from ...ndarray import op
from ...ndarray import random


def _make_numpy_array(x):
    # if already numpy, return
    if isinstance(x, np.ndarray):
        return x
    elif np.isscalar(x):
        return np.array([x])
    elif isinstance(x, NDArray):
        return x.asnumpy()
    else:
        raise TypeError('_make_numpy_array only accepts input types of numpy.ndarray, scalar,'
                        ' and MXNet NDArray, while received type {}'.format(str(type(x))))


def _make_grid_v1(img, ncols=8):
    """This will be deprecated once make_grid is stable."""
    assert isinstance(img, np.ndarray), 'plugin error, should pass numpy array here'
    assert img.ndim == 4 and img.shape[1] == 3
    nimg = img.shape[0]
    h = img.shape[2]
    w = img.shape[3]
    ncols = min(nimg, ncols)
    nrows = int(np.ceil(float(nimg) / ncols))
    canvas = np.zeros((3, h * nrows, w * ncols))
    i = 0
    for y in range(nrows):
        for x in range(ncols):
            if i >= nimg:
                break
            canvas[:, y * h:(y + 1) * h, x * w:(x + 1) * w] = img[i]
            i = i + 1
    return canvas


# TODO(junwu): Add support for MXNet NDArray, currently only supports np.ndarray
# the change should be simple since most of ops in MXNet have the same names as in NumPy
def make_grid_v1(tensor, nrow=8, padding=2, normalize=False,
                 norm_range=None, scale_each=False, pad_value=0):
    """Make a grid of images. This is a NumPy version of torchvision.utils.make_grid
    Ref: https://github.com/pytorch/vision/blob/master/torchvision/utils.py#L6

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (N x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        norm_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.

    Example:
        See this notebook
        `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`

    """
    if not isinstance(tensor, np.ndarray) \
            or not (isinstance(tensor, np.ndarray)
                    and all(isinstance(t, np.ndarray) for t in tensor)):
        raise TypeError('numpy.ndarray or list of numpy.ndarrays expected,'
                        ' got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = np.stack(tensor, axis=0)

    if tensor.ndim == 2:  # single image H x W
        tensor = tensor.reshape(((1,) + tensor.shape))
    if tensor.ndim == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = np.concatenate((tensor, tensor, tensor), axis=0)
        tensor = tensor.reshape((1,) + tensor.shape)
    if tensor.ndim == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = np.concatenate((tensor, tensor, tensor), axis=1)

    if normalize is True:
        tensor = tensor.copy()  # avoid modifying tensor in-place
        if norm_range is not None:
            assert isinstance(norm_range, tuple) and len(norm_range) == 2, \
                "norm_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            np.clip(a=img, a_min=min, a_max=max, out=img)
            img -= min
            img /= (max - min)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, t.min(), t.max())

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, norm_range)
        else:
            norm_range(tensor, norm_range)

    # if single image, just return
    if tensor.shape[0] == 1:
        return tensor.squeeze()

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(np.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    grid = np.empty(shape=(3, height * ymaps + padding, width * xmaps + padding), dtype=tensor.dtype)
    grid[:] = pad_value
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            start1 = y * height + padding
            end1 = start1 + height - padding
            start2 = x * width + padding
            end2 = start2 + width - padding
            grid[:, start1:end1, start2:end2] = tensor[k]
            k = k + 1
    return grid


def make_grid(tensor, nrow=8, padding=2, normalize=False, norm_range=None, scale_each=False, pad_value=0):
    """Make a grid of images. This is an MXNet version of torchvision.utils.make_grid
    Ref: https://github.com/pytorch/vision/blob/master/torchvision/utils.py

    Parameters
    ----------
        tensor : `NDArray` or list of `NDArray`s
            Input image(s) in the format of HW, CHW, or NCHW.
        nrow : int
            Number of images displayed in each row of the grid. The Final grid size is (batch_size / `nrow`, `nrow`).
        padding : int
            Padding value for each image in the grid.
        normalize : bool
            If True, shift the image to the range (0, 1), by subtracting the
            minimum and dividing by the maximum pixel value.
        norm_range : tuple
            Tuple of (min, max) where min and max are numbers. These numbers are used to normalize the image.
            By default, `min` and `max` are computed from the `tensor`.
        scale_each : bool
            If True, scale each image in the batch of images separately rather than the `(min, max)` over all images.
        pad_value : float
            Value for the padded pixels.

    Returns
    -------
    NDArray
        A image grid made of the input images.
    """
    if not isinstance(tensor, NDArray) or not (isinstance(tensor, NDArray) and
                                               all(isinstance(t, NDArray) for t in tensor)):
        raise TypeError('MXNet NDArray or list of NDArrays expected, got {}'.format(str(type(tensor))))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = op.stack(tensor, axis=0)

    if tensor.ndim == 2:  # single image H x W
        tensor = tensor.reshape(((1,) + tensor.shape))
    if tensor.ndim == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = op.concat(*(tensor, tensor, tensor), dim=0)
        tensor = tensor.reshape((1,) + tensor.shape)
    if tensor.ndim == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = op.concat(*(tensor, tensor, tensor), dim=1)

    if normalize is True:
        tensor = tensor.copy()  # avoid modifying tensor in-place
        if norm_range is not None:
            assert isinstance(norm_range, tuple) and len(norm_range) == 2, \
                "norm_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            op.clip(img, a_min=min, a_max=max, out=img)
            img -= min
            img /= (max - min)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, t.min(), t.max())

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, norm_range)
        else:
            norm_range(tensor, norm_range)

    # if single image, just return
    if tensor.shape[0] == 1:
        return tensor.squeeze()

    # make the batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(np.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    grid = nd.empty(shape=(3, height * ymaps + padding, width * xmaps + padding), dtype=tensor.dtype,
                    ctx=tensor.context)
    grid[:] = pad_value
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            start1 = y * height + padding
            end1 = start1 + height - padding
            start2 = x * width + padding
            end2 = start2 + width - padding
            grid[:, start1:end1, start2:end2] = tensor[k]
            k = k + 1
    return grid


def save_image_v1(tensor, filename, nrow=8, padding=2,
                  normalize=False, norm_range=None,
                  scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    if isinstance(tensor, NDArray):
        tensor = tensor.asnumpy()
    elif not isinstance(tensor, np.ndarray):
        raise TypeError('expected numpy.ndarray or mx.nd.NDArray if MXNet is installed'
                        ', while received type=%s' % str(type(tensor)))
    grid = make_grid_v1(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                        normalize=normalize, norm_range=norm_range, scale_each=scale_each)
    ndarr = grid * 255
    np.clip(a=ndarr, a_min=0, a_max=255, out=ndarr)
    ndarr = ndarr.astype(np.uint8).transpose((1, 2, 0))
    if Image is None:
        raise ImportError('saving image failed because PIL is not found')
    im = Image.fromarray(ndarr)
    im.save(filename)


def save_image(tensor, filename, nrow=8, padding=2, normalize=False, norm_range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file. If the input tensor contains multiple images,
    a grid of images will be saved.

    Parameters
    ----------
        tensor : `NDArray` or list of `NDArray`s
            Input image(s) in the format of HW, CHW, or NCHW.
        filename : str
            Filename of the saved image(s).
        nrow : int
            Number of images displayed in each row of the grid. The Final grid size is (batch_size / `nrow`, `nrow`).
        padding : int
            Padding value for each image in the grid.
        normalize : bool
            If True, shift the image to the range (0, 1), by subtracting the
            minimum and dividing by the maximum pixel value.
        norm_range : tuple
            Tuple of (min, max) where min and max are numbers. These numbers are used to normalize the image.
            By default, `min` and `max` are computed from the `tensor`.
        scale_each : bool
            If True, scale each image in the batch of images separately rather than the `(min, max)` over all images.
        pad_value : float
            Value for the padded pixels.
    """
    if not isinstance(tensor, NDArray) or not (isinstance(tensor, NDArray) and
                                               all(isinstance(t, NDArray) for t in tensor)):
        raise TypeError('MXNet NDArray or list of NDArrays expected, got {}'.format(str(type(tensor))))
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, norm_range=norm_range, scale_each=scale_each)
    ndarr = grid * 255
    op.clip(ndarr, a_min=0, a_max=255, out=ndarr)
    ndarr = ndarr.astype(np.uint8).transpose((1, 2, 0))
    if Image is None:
        raise ImportError('saving image failed because PIL is not found')
    im = Image.fromarray(ndarr.asnumpy())
    im.save(filename)


def _prepare_image_v1(img):
    assert isinstance(img, np.ndarray), 'plugin error, should pass numpy array here'
    assert img.ndim == 2 or img.ndim == 3 or img.ndim == 4
    if img.ndim == 4:  # NCHW
        if img.shape[1] == 1:  # N1HW
            img = np.concatenate((img, img, img), 1)  # N3HW
        assert img.shape[1] == 3
        img = _make_grid_v1(img)  # 3xHxW
    if img.ndim == 3 and img.shape[0] == 1:  # 1xHxW
        img = np.concatenate((img, img, img), 0)  # 3xHxW
    if img.ndim == 2:  # HxW
        img = np.expand_dims(img, 0)  # 1xHxW
        img = np.concatenate((img, img, img), 0)  # 3xHxW
    img = img.transpose((1, 2, 0))

    return img


def _prepare_image(img):
    """Given an image of format HW, CHW, or NCHW, returns a image of format HWC. If the input is a batch
    of images, a grid of images is made by stitching them together."""
    assert img.ndim == 2 or img.ndim == 3 or img.ndim == 4
    if isinstance(img, NDArray):
        return make_grid(img).transpose((1, 2, 0))
    elif isinstance(img, np.ndarray):
        return make_grid_v1(img).transpose((1, 2, 0))
    else:
        raise TypeError('expected NDArray of np.ndarray, while received type {}'.format(str(type(img))))


def _make_metadata_tsv(metadata, save_path):
    """Given an `NDArray` or a `numpy.ndarray` as metadata e.g. labels, save the flattened array into
    the file metadata.tsv under the path provided by the user. Made to satisfy the requirement in the following link:
    https://www.tensorflow.org/programmers_guide/embedding#metadata"""
    if isinstance(metadata, NDArray):
        metadata = metadata.asnumpy().flatten()
    elif isinstance(metadata, np.ndarray):
        metadata = metadata.flatten()
    else:
        raise TypeError('expected NDArray of np.ndarray, while received type {}'.format(str(type(metadata))))
    metadata = [str(x) for x in metadata]
    with open(os.path.join(save_path, 'metadata.tsv'), 'w') as f:
        for x in metadata:
            f.write(x + '\n')


def _make_sprite_image_v1(images, save_path):
    # this ensures the sprite image has correct dimension as described in
    # https://www.tensorflow.org/get_started/embedding_viz
    shape = images.shape
    nrow = int(np.ceil(np.sqrt(shape[0])))

    images = _make_numpy_array(images)
    # augment images so that #images equals nrow*nrow
    images = np.concatenate((images,
                             np.random.normal(loc=0, scale=1, size=((nrow * nrow - shape[0],) + shape[1:])) * 255),
                            axis=0)
    save_image(images, os.path.join(save_path, 'sprite.png'), nrow=nrow, padding=0)


def _make_sprite_image(images, save_path):
    """Given an NDArray as a batch images, make a sprite image out of it following the rule defined in
    https://www.tensorflow.org/programmers_guide/embedding and save it in sprite.png under the path provided
    by the user"""
    assert isinstance(images, NDArray)
    shape = images.shape
    nrow = int(np.ceil(np.sqrt(shape[0])))
    # augment images so that #images equals nrow * nrow
    aug = random.normal(loc=0, scale=1, shape=((nrow * nrow - shape[0],) + shape[1:]), ctx=images.context) * 255
    images = op.concat(*(images, aug), dim=0)
    save_image(images, os.path.join(save_path, 'sprite.png'), nrow=nrow, padding=0)


def _add_embedding_config(file_path, global_step, metadata=None, label_img_shape=None, tag='default'):
    """Creates a config file used by the embedding projector.
    Adapted from the TensorFlow function `visualize_embeddings()` at
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/tensorboard/plugins/projector/__init__.py"""
    with open(os.path.join(file_path, 'projector_config.pbtxt'), 'a') as f:
        s = 'embeddings {\n'
        s += 'tensor_name: "{}:{}"\n'.format(tag, global_step)
        s += 'tensor_path: "{}"\n'.format(os.path.join(global_step, 'tensors.tsv'))
        if metadata is not None:
            s += 'metadata_path: "{}"\n'.format(os.path.join(global_step, 'metadata.tsv'))
        if label_img_shape is not None:
            if len(label_img_shape) != 4:
                logging.warn('expected 4D sprite image in the format NCHW, while received image ndim={},'
                             ' skipping saving sprite image info'.format(len(label_img_shape)))
            else:
                s += 'sprite {\n'
                s += 'image_path: "{}"\n'.format(os.path.join(global_step, 'sprite.png'))
                s += 'single_image_dim: {}\n'.format(label_img_shape[3])
                s += 'single_image_dim: {}\n'.format(label_img_shape[2])
                s += '}\n'
        s += '}\n'
        f.write(s)


def _save_ndarray_tsv(data, file_path):
    """Given an `NDarray` or a `numpy.ndarray`, save it in tensors.tsv under the path provided by the user."""
    if isinstance(data, np.ndarray):
        data_list = data.tolist()
    elif isinstance(data, NDArray):
        data_list = data.asnumpy().tolist()
    else:
        raise TypeError('expected NDArray of np.ndarray, while received type {}'.format(str(type(data))))
    with open(os.path.join(file_path, 'tensors.tsv'), 'w') as f:
        for x in data_list:
            x = [str(i) for i in x]
            f.write('\t'.join(x) + '\n')
