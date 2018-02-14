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
from __future__ import print_function
__all__ = ['get_model_file', 'purge']
import os
import zipfile

from ..utils import download, check_sha1

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
    ('38d6d423c22828718ec3397924b8e116a03e6ac0', 'resnet18_v1'),
    ('4dc2c2390a7c7990e0ca1e53aeebb1d1a08592d1', 'resnet34_v1'),
    ('c940b1a062b32e3a5762f397c9d1e178b5abd007', 'resnet50_v1'),
    ('d992389084bc5475c370e9b52c3561706e755799', 'resnet101_v1'),
    ('48ce7775d375987d019ec9aa96bc43b98165dfcb', 'resnet152_v1'),
    ('8aacf80ff4014c1efa2362a963ac5ec82cf92d5b', 'resnet18_v2'),
    ('0ed3cd06da41932c03dea1de7bc2506ef3fb97b3', 'resnet34_v2'),
    ('81a4e66af7859a5aa904e2b4051aa0d3bc472b2f', 'resnet50_v2'),
    ('7eb2b3cde097883c11941b927048a705ed334294', 'resnet101_v2'),
    ('64c75ac8c292f6ac54f873f9ef62e0531105878b', 'resnet152_v2'),
    ('264ba4970a0cc87a4f15c96e25246a1307caf523', 'squeezenet1.0'),
    ('33ba0f93753c83d86e1eb397f38a667eaf2e9376', 'squeezenet1.1'),
    ('dd221b160977f36a53f464cb54648d227c707a05', 'vgg11'),
    ('ee79a8098a91fbe05b7a973fed2017a6117723a8', 'vgg11_bn'),
    ('6bc5de58a05a5e2e7f493e2d75a580d83efde38c', 'vgg13'),
    ('7d97a06c3c7a1aecc88b6e7385c2b373a249e95e', 'vgg13_bn'),
    ('649467530119c0f78c4859999e264e7bf14471a9', 'vgg16'),
    ('6b9dbe6194e5bfed30fd7a7c9a71f7e5a276cb14', 'vgg16_bn'),
    ('f713436691eee9a20d70a145ce0d53ed24bf7399', 'vgg19'),
    ('9730961c9cea43fd7eeefb00d792e386c45847d6', 'vgg19_bn')]}

apache_repo_url = 'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/'
_url_format = '{repo_url}gluon/models/{file_name}.zip'

def short_hash(name):
    if name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha1[name][:8]

def get_model_file(name, root=os.path.join('~', '.mxnet', 'models')):
    r"""Return location for the pretrained on local file system.

    This function will download from online model zoo when model cannot be found or has mismatch.
    The root directory will be created if it doesn't exist.

    Parameters
    ----------
    name : str
        Name of the model.
    root : str, default '~/.mxnet/models'
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
            print('Mismatch in the content of model file detected. Downloading again.')
    else:
        print('Model file is not found. Downloading.')

    if not os.path.exists(root):
        os.makedirs(root)

    zip_file_path = os.path.join(root, file_name+'.zip')
    repo_url = os.environ.get('MXNET_GLUON_REPO', apache_repo_url)
    if repo_url[-1] != '/':
        repo_url = repo_url + '/'
    download(_url_format.format(repo_url=repo_url, file_name=file_name),
             path=zip_file_path,
             overwrite=True)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(root)
    os.remove(zip_file_path)

    if check_sha1(file_path, sha1_hash):
        return file_path
    else:
        raise ValueError('Downloaded file has different hash. Please try again.')

def purge(root=os.path.join('~', '.mxnet', 'models')):
    r"""Purge all pretrained model files in local file store.

    Parameters
    ----------
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    root = os.path.expanduser(root)
    files = os.listdir(root)
    for f in files:
        if f.endswith(".params"):
            os.remove(os.path.join(root, f))
