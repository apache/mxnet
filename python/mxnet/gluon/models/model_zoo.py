# coding: utf-8
"""Model zoo for pre-trained models."""
from __future__ import print_function
__all__ = ['get_model_file']
import hashlib
import os
import zipfile

from ...test_utils import download

_model_sha1 = {name: checksum for checksum, name in [
    ('374727bb60dfaa27a8b6d3edd7060970cd53a1b0', 'alexnet'),
    ('9789fb109175d8d91ac92880169b1dc33eaedcd6', 'densenet121'),
    ('c35e8edff0ac7978b05c441adfb8c5c34d034e7b', 'densenet161'),
    ('12c2757f26a2bc9cb9911ca13d2f87822c926a48', 'densenet169'),
    ('e997f871a0a126efa4d252de6677c07d8f37258b', 'densenet201'),
    ('382cd1c5c5f2153feaac77aba7bf4f44568d671a', 'inceptionv3'),
    ('2b54423eccae747026dea3d092c4938d55d7dc6e', 'resnet101_v1'),
    ('98b4908c1417a003453c64583029c9fa8fd189a6', 'resnet152_v1'),
    ('de4170ddac5a3124e4ee8407e17fdac6e638bb83', 'resnet18_v1'),
    ('137e986b2db597245954ee2f7c19399d1a74a9f4', 'resnet34_v1'),
    ('2d2c53abbb7ffd913a6a724a45c5a1f3ca7dcf29', 'resnet50_v1'),
    ('1d896a3420e60477a21411851446caf974560acf', 'squeezenet1.0'),
    ('96d8b168050be3b7addab38e48cd226b04a4f69b', 'squeezenet1.1'),
    ('a01d1ba90b230dae62b4a4abda9d3b2e6e123e0c', 'vgg11'),
    ('5e3b0398046fa0dca90c2f9b1ea1ec34f368148b', 'vgg11_bn'),
    ('a0f433b865847938b7ca6586deb016d779e0af7e', 'vgg13'),
    ('97dc8506037456243d1cf1263853b100d188c033', 'vgg13_bn'),
    ('4b664b92522d95c4b893b1351d88489bf60a9e4b', 'vgg16'),
    ('d315c48b9ba628f7f19db5d8f69105fd280ef938', 'vgg16_bn'),
    ('8ba5ac028ff1baad1f1f69d2d0395398c1d30f44', 'vgg19'),
    ('497948c20de0fdfb92f9bcfa5b222a9656046f1e', 'vgg19_bn')]}

_url_format = 'https://{bucket}.s3.amazonaws.com/gluon/models/{file_name}.zip'
bucket = 'apache-mxnet'

def short_hash(name):
    if name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha1[name][:8]

def verified(file_path, name):
    sha1 = hashlib.sha1()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == _model_sha1[name]

def get_model_file(name, local_dir=os.path.expanduser('~/.mxnet/models/')):
    r"""Return location for the pretrained on local file system.

    This function will download from online model zoo when model cannot be found or has mismatch.

    Parameters
    ----------
    name : str
        Name of the model.
    local_dir : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    file_name = '{name}-{short_hash}'.format(name=name,
                                             short_hash=short_hash(name))
    file_path = os.path.join(local_dir, file_name+'.params')
    if os.path.exists(file_path):
        if verified(file_path, name):
            return file_path
        else:
            print('Mismatch in the content of model file detected. Downloading again.')
    else:
        print('Model file is not found. Downloading.')

    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    download(_url_format.format(bucket=bucket,
                                file_name=file_name),
             fname=file_name+'.zip',
             dirname=local_dir,
             overwrite=True)
    zip_file_path = os.path.join(local_dir, file_name+'.zip')
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(local_dir)
    os.remove(zip_file_path)

    if verified(file_path, name):
        return file_path
    else:
        raise ValueError('Downloaded file has different hash. Please try again.')
