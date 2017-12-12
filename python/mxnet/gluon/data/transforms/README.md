# Transoforms

## Generic

### `Compose`
Function that composes multiple transformations into one.

```python
from mxbox import transforms
transform = transforms.Compose([
    transforms.Scale(256), 
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.mx.ToNdArray(),
    transforms.mx.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                            std  = [ 0.229, 0.224, 0.225 ]),
])
```

### `Lambda`
Given a Python lambda, applies it to the input img and returns it. For example:

```python
transforms.Lambda(lambda x: x.add(10))
```

## Transformation on PIL.Image

Note: This part is almost same as transformations provided in [torchvision](https://github.com/pytorch/vision#transforms-on-pilimage).

### `Scale(size, interpolation=Image.BILINEAR)`

Rescales the input PIL.Image to the given 'size'.

If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.

If 'size' is a number, it will indicate the size of the smaller edge. For example, if height > width, then image will be rescaled to (size * height / width, size) - size: size of the smaller edge - interpolation: Default: PIL.Image.BILINEAR

### `CenterCrop(size)` - center-crops the image to the given size

Crops the given PIL.Image at the center to have a region of the given size. size can be a tuple (target_height, target_width) or an integer, in which case the target will be of a square shape (size, size)
RandomCrop(size, padding=0)

Crops the given PIL.Image at a random location to have a region of the given size. size can be a tuple (target_height, target_width) or an integer, in which case the target will be of a square shape (size, size) If padding is non-zero, then the image is first zero-padded on each side with padding pixels.

### `RandomHorizontalFlip()`

Randomly horizontally flips the given PIL.Image with a probability of 0.5

### `RandomSizedCrop(size, interpolation=Image.BILINEAR)`

Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio

This is popularly used to train the Inception networks - size: size of the smaller edge - interpolation: Default: PIL.Image.BILINEAR

### `Pad(padding, fill=0)`

Pads the given image on each side with padding number of pixels, and the padding pixels are filled with pixel value fill. If a 5x5 image is padded with padding=1 then it becomes 7x7


## Transformation on mx.ndarray
Under namespace `mxbox.transforms.mx`, e.g, `mxbox.transforms.mx.stack()`,

### `stack(sequence, axis=0)`

Stack a sequences of `mx.ndarray` along with a specified new dimension.

```python
seq = [mx.nd.array(np.zeros([3, 32, 32])) for i in range(10)]

stack(seq, axis=0)  # results in a [10x3x32x32] ndarray
stack(seq, axis=1)  # results in a [3x10x32x32] ndarray

# sometimes appear in classification labels
seq = [i for i in range(10)]
stack(seq) # results in a [10] ndarray
```

### `ToNdArray(dtype=np.float32)`
Convert `PIL.Image` or `numpy` to `mx.ndarray`. Default dtype is `np.float32`, which should be compatible with popular graphic cards. If you want to try higher precision, or your card does not support `float32`, you can set it by yourself.

Note: `ToNdArray()` will automatically transpose channels from `NxHxW` to `WxHxN` to fit mxnet preference.


```python
img # [3x32x32]
ToNdArray()(img) # [32x3x3]
``` 
### `Normalize(mean, std=[1, 1, 1])`

Given mean: (R, G, B) and std: (R, G, B), will normalize each channel of the `mx.ndarray`, i.e. channel = (channel - mean) / std.

