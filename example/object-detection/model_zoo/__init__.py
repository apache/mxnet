"""Module for pretrained model for object-detection package.
"""
from mxnet.gluon.model_zoo import vision
from model_zoo.ssd import *

def get_detection_model(name, **kwargs):
    """Return a pre-defined model by name

    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : int
        Whether to load the pretrained weights for model.
    classes : int
        Number of classes for the output layer.

    Returns
    -------
    Block
        The model.
    """
    models = {'ssd_512_resnet18_v1': ssd_512_resnet18_v1,
              'ssd_512_resnet50_v1': ssd_512_resnet50_v1,
             }
    name = name.lower()
    if name not in models:
        raise ValueError(
            'Model %s is not supported. Available options are\n\t%s'%(
                name, '\n\t'.join(sorted(models.keys()))))
    return models[name](**kwargs)
