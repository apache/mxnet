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
# pylint: disable=redefined-builtin

"""Utility functions."""

import os

from ...text import Vocabulary

def flatten_samples(samples):
    """Flatten list of list of tokens into a single flattened list of tokens.

    Parameters
    ----------
    samples : list of list of object
        List of samples, each of which is a list of tokens.

    Returns
    -------
    Flattened list of tokens.
    """
    return [token for sample in samples for token in sample if token]

def collate(flat_sample, seq_len, overlap=0):
    """Collate a flat list of tokens into list of list of tokens, with each
    inner list's length equal to the specified `seq_len`.

    Parameters
    ----------
    flat_sample : list of object
        A flat list of tokens.
    seq_len : int
        The length of each of the samples.
    overlap : int, default 0
        The extra number of items in current sample that should overlap with the
        next sample.

    Returns
    -------
    List of samples, each of which has length equal to `seq_len`.
    """
    num_samples = (len(flat_sample)-seq_len) // (seq_len-overlap) + 1
    return [flat_sample[i*(seq_len-overlap):((i+1)*seq_len-i*overlap)] for i in range(num_samples)]

def collate_pad_length(num_items, seq_len, overlap=0):
    """Calculate the padding length needed for collated samples in order not to discard data.

    Parameters
    ----------
    num_items : int
        Number of items in dataset before collating.
    seq_len : int
        The length of each of the samples.
    overlap : int, default 0
        The extra number of items in current sample that should overlap with the
        next sample.

    Returns
    -------
    Length of paddings.
    """
    step = seq_len-overlap
    span = num_items-seq_len
    return (span // step + 1) * step - span

_vocab_sha1 = {}

def _load_pretrained_vocab(name, root=os.path.join('~', '.mxnet', 'models')):
    """Load the accompanying vocabulary object for pretrained model.

    Parameters
    ----------
    name : str
        Name of the model.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested vocabulary object file of pretrained model.
    """
    file_name = '{name}-{short_hash}'.format(name=name,
                                             short_hash=short_hash(name))
    root = os.path.expanduser(root)
    file_path = os.path.join(root, file_name+'.vocab')
    sha1_hash = _vocab_sha1[name]
    if os.path.exists(file_path):
        if check_sha1(file_path, sha1_hash):
            return file_path
        else:
            print('Detected mismatch in the content of model vocab file. Downloading again.')
    else:
        print('Vocab file is not found. Downloading.')

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
        return Vocabulary.json_deserialize(open(file_path, "rb").read())
    else:
        raise ValueError('Downloaded file has different hash. Please try again.')
