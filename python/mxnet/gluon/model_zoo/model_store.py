# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
"""Model zoo for pre-trained models."""
__all__ = ['get_model_file', 'purge']
import os
import zipfile
import logging
import tempfile
import uuid
import shutil

from ..utils import download, check_sha1, replace_file
from ... import base

_model_sha1 = {name: checksum for checksum, name in [
    ('44335d1f0046b328243b32a26a4fbd62d9057b45', 'alexnet'),
    ('f27dbf2dbd5ce9a80b102d89c7483342cd33cb31', 'densenet121'),
    ('b6c8a95717e3e761bd88d145f4d0a214aaa515dc', 'densenet161'),
    ('2603f878403c6aa5a71a124c4a3307143d6820e9', 'densenet169'),
    ('1cdbc116bc3a1b65832b18cf53e1cb8e7da017eb', 'densenet201'),
    ('ed47ec45a937b656fcc94dabde85495bbef5ba1f', 'inceptionv3'),
    ('9f83e440996887baf91a6aff1cccc1c903a64274', 'mobilenet0.25'),
    ('8e9d539cc66aa5efa71c4b6af983b936ab8701c3', 'mobilenet0.5'),
    ('529b2c7f4934e6cb851155b22c96c9ab0a7c4dc2', 'mobilenet0.75'),
    ('6b8c5106c730e8750bcd82ceb75220a3351157cd', 'mobilenet1.0'),
    ('36da4ff1867abccd32b29592d79fc753bca5a215', 'mobilenetv2_1.0'),
    ('e2be7b72a79fe4a750d1dd415afedf01c3ea818d', 'mobilenetv2_0.75'),
    ('aabd26cd335379fcb72ae6c8fac45a70eab11785', 'mobilenetv2_0.5'),
    ('ae8f9392789b04822cbb1d98c27283fc5f8aa0a7', 'mobilenetv2_0.25'),
    ('a0666292f0a30ff61f857b0b66efc0228eb6a54b', 'resnet18_v1'),
    ('48216ba99a8b1005d75c0f3a0c422301a0473233', 'resnet34_v1'),
    ('0aee57f96768c0a2d5b23a6ec91eb08dfb0a45ce', 'resnet50_v1'),
    ('d988c13d6159779e907140a638c56f229634cb02', 'resnet101_v1'),
    ('671c637a14387ab9e2654eafd0d493d86b1c8579', 'resnet152_v1'),
    ('a81db45fd7b7a2d12ab97cd88ef0a5ac48b8f657', 'resnet18_v2'),
    ('9d6b80bbc35169de6b6edecffdd6047c56fdd322', 'resnet34_v2'),
    ('ecdde35339c1aadbec4f547857078e734a76fb49', 'resnet50_v2'),
    ('18e93e4f48947e002547f50eabbcc9c83e516aa6', 'resnet101_v2'),
    ('f2695542de38cf7e71ed58f02893d82bb409415e', 'resnet152_v2'),
    ('264ba4970a0cc87a4f15c96e25246a1307caf523', 'squeezenet1.0'),
    ('33ba0f93753c83d86e1eb397f38a667eaf2e9376', 'squeezenet1.1'),
    ('dd221b160977f36a53f464cb54648d227c707a05', 'vgg11'),
    ('ee79a8098a91fbe05b7a973fed2017a6117723a8', 'vgg11_bn'),
    ('6bc5de58a05a5e2e7f493e2d75a580d83efde38c', 'vgg13'),
    ('7d97a06c3c7a1aecc88b6e7385c2b373a249e95e', 'vgg13_bn'),
    ('e660d4569ccb679ec68f1fd3cce07a387252a90a', 'vgg16'),
    ('7f01cf050d357127a73826045c245041b0df7363', 'vgg16_bn'),
    ('ad2f660d101905472b83590b59708b71ea22b2e5', 'vgg19'),
    ('f360b758e856f1074a85abd5fd873ed1d98297c3', 'vgg19_bn')]}

apache_repo_url = 'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/'
_url_format = '{repo_url}gluon/models/{file_name}.zip'

def short_hash(name):
    if name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha1[name][:8]

def get_model_file(name, root=os.path.join(base.data_dir(), 'models')):
    r"""Return location for the pretrained on local file system.

    This function will download from online model zoo when model cannot be found or has mismatch.
    The root directory will be created if it doesn't exist.

    Parameters
    ----------
    name : str
        Name of the model.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    file_name = '{name}-{short_hash}'.format(name=name,
                                             short_hash=short_hash(name))
    root = os.path.expanduser(root)
    file_path = os.path.join(root, file_name+'.params')
    sha1_hash = _model_sha1[name]
    if os.path.exists(file_path):
        if check_sha1(file_path, sha1_hash):
            return file_path
        else:
            logging.warning('Mismatch in the content of model file detected. Downloading again.')
    else:
        logging.info('Model file not found. Downloading to %s.', file_path)

    os.makedirs(root, exist_ok=True)

    repo_url = os.environ.get('MXNET_GLUON_REPO', apache_repo_url)
    if repo_url[-1] != '/':
        repo_url = repo_url + '/'

    random_uuid = str(uuid.uuid4())
    temp_zip_file_path = os.path.join(root, file_name+'.zip'+random_uuid)
    download(_url_format.format(repo_url=repo_url, file_name=file_name),
             path=temp_zip_file_path, overwrite=True)
    with zipfile.ZipFile(temp_zip_file_path) as zf:
        temp_dir = tempfile.mkdtemp(dir=root)
        zf.extractall(temp_dir)
        temp_file_path = os.path.join(temp_dir, file_name+'.params')
        replace_file(temp_file_path, file_path)
        shutil.rmtree(temp_dir)
    os.remove(temp_zip_file_path)

    if check_sha1(file_path, sha1_hash):
        return file_path
    else:
        raise ValueError('Downloaded file has different hash. Please try again.')

def purge(root=os.path.join(base.data_dir(), 'models')):
    r"""Purge all pretrained model files in local file store.

    Parameters
    ----------
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    root = os.path.expanduser(root)
    files = os.listdir(root)
    for f in files:
        if f.endswith(".params"):
            os.remove(os.path.join(root, f))
