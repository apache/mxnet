#Image Caption Generation

This is a simple implementaion of paper Neural Image Talk[1] based on mxnet.
I borrowed many ideas from NeuralTalk2[2] by Karparthy.

##Requirement
1. h5py

##Usage
1. Prepare input data using `preprocess.py`. The raw json file looks like
>  raw.json
> 
>  {'captions': [u'A man with a red helmet on a small moped on a dirt road. ',
>  u'Man riding a motor bike on a dirt road on the countryside.',
>  u'A man riding on the back of a motorcycle.',
>  u'A dirt path with a young person on a motor bike rests to the foreground
>  of a verdant area with a bridge and a background of cloud-wreathed mountains. ',
>  u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'],
>  'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

2. train it. `python train.py -h` to know details.

## Some problems
1. Compared to Torch implementation, speed is slow.
2. HDF5 is used for saving image and data iterator is ugly.

---
#Reference
1. Vinyals O, Toshev A, Bengio S, et al. Show and tell: A neural image caption generator[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015: 3156-3164.
2. https://github.com/karpathy/neuraltalk2
