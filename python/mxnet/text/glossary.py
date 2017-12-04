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

"""Read text files and load embeddings."""
from __future__ import absolute_import
from __future__ import print_function

import os
import random
import logging
import json
import numpy as np

from ..base import numeric_types
from ..test_utils import download
from .. import ndarray as nd
from .. import io



class Glossary(object):
    ## keys of counter cannot contain symbols.
    ## counter cannot be None.
    ## symbols first, then token are sort by frequency/entropy, # then alphabetically
    def __init__(self, counter, top_k_freq=None, min_freq=1,
                 symbols=['<unk>'], embeds=None):
        # Sanity checks.
        assert min_freq > 0, '`min_freq` must be set to a positive value.'
        assert len(symbols) > 0, '`symbols` must be an non-empty list whose ' \
                                 'first element is the string representation ' \
                                 'for unknown tokens, such as "<unk>".'

        # Initialize attributes.
        self._init_attrs(counter, symbols)

        # Set self._idx_to_token and self._token_to_idx according to specified
        # frequency thresholds.
        self._set_idx_and_token(counter, symbols, top_k_freq, min_freq)

        if embeds is not None:
            self.set_idx_to_embed(embeds)

    def _init_attrs(self, counter, symbols):
        self._counter = counter.copy()
        self._token_to_idx = {token: idx for idx, token in enumerate(symbols)}
        self._idx_to_token = symbols.copy()
        self._idx_to_embed = None
        self._embed_dim = 0

    def _set_idx_and_token(self, counter, symbols, top_k_freq, min_freq):
        # Update _counter to include special symbols, such as '<unk>'.
        self._counter.update({token: 0 for token in symbols})
        assert len(self._counter) == len(counter) + len(symbols), 'symbols ' \
            'cannot contain any token from keys of counter.'

        token_freqs = sorted(self._counter.items(), key=lambda x: x[0])
        token_freqs.sort(key=lambda x: x[1], reverse=True)

        token_cap = len(self._counter) if top_k_freq is None \
            else len(symbols) + top_k_freq

        for token, freq in token_freqs:
            if freq < min_freq or len(self._idx_to_token) == token_cap:
                break
            self._idx_to_token.append(token)
            self._token_to_idx[token] = len(self._idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    @property
    def counter(self):
        return self._counter

    @property
    def token_to_idx(self):
        return self._token_to_idx

    @property
    def idx_to_token(self):
        return self._idx_to_token

    @property
    def idx_to_embed(self):
        return self._idx_to_embed

    @property
    def embed_dim(self):
        return self._embed_dim

    def set_idx_to_embed(self, embeds):
        # Sanity check.
        if isinstance(embeds, list):
            for loaded_embed in embeds:
                assert isinstance(loaded_embed, Embedding)
        else:
            assert isinstance(embeds, Embedding)
            embeds = [embeds]

        self._embed_dim = sum(embed.dim for embed in embeds)
        self._idx_to_embed = nd.zeros(shape=(len(self), self._embed_dim))

        for idx, token in enumerate(self.idx_to_token):
            col_start = 0
            # Concatenate all the embedding vectors in embeds.
            for embed in embeds:
                col_end = col_start + embed.dim
                self._idx_to_embed[idx][col_start:col_end] = embed[token]
                col_start = col_end


class Embedding(object):

    # Key-value pairs for embedding name in lower case and embedding class.
    embed_registry = {}

    ## When url is None, pretrain_file must be a path to a pretrain_file given by user
    ##
    ## pretrain format: word embed_vec_elem0 embed_vec_elem1 ...
    ##
    def __init__(self, pretrain_file, url=None, embed_name='my_embed',
                 embed_root='~/.mxnet/embeddings/', unk_embed=nd.zeros_like):

        embed_root = os.path.expanduser(embed_root)






        self._unk_embed = unk_embed

        # Sanity check.
        if os.path.isfile(pretrain_file) and url is not None:
            raise ValueError('When pretrain_file is a path to a user-provided '
                             'pretrained embedding file, url must be set to '
                             'None; when url to pretrained embedding file(s) '
                             'is specified, pretrain_file must be the name '
                             'rather than the path of the pretrained embedding '
                             'file. This is to avoid confusion of the source '
                             'pretrained embedding file to be loaded.')

        # User specifies pretrained embedding file at the path of pretrain_file.
        if os.path.isfile(pretrain_file):
            # The path to the pretrained embedding file to be loaded.
            pretrain_path = pretrain_file
            # The path to the serialized embedding object.
            serialized_path = os.path.join(embed_root, embed_name,
                                           os.path.basename(pretrain_file)
                                           + '.mx')
        # User specifies pretrained embedding file to be downloaded from url if
        # url is not None; if url is None, user may want to load a serialized
        # embedding object at embed_root/embed_name/pretrain_file.mx.
        else:
            pretrain_path = os.path.join(embed_root, embed_name, pretrain_file)
            serialized_path = pretrain_path + '.mx'

        # Load serialized embedding object if it is available.
        if os.path.isfile(serialized_path):
            logging.info('Loading embedding from %s' % serialized_path)
            (self._dim, self._token_to_idx, self._idx_to_token,
             self._idx_to_embed) = nd.load(serialized_path)
        else:
            # The pretrained embedding file is to be downloaded from url if it
            # has not been downloaded yet.
            if url is not None:
                embed_dir_name = os.path.join(embed_root, embed_name)
                download_file_path = os.path.join(embed_dir_name,
                                                  os.path.basename(url))

                # Donwload if necessary.
                if not os.path.isfile(download_file_path):
                    logging.info('Downloading pretrained embedding files from '
                                 '%s' % url)
                    download(url, dirname=embed_dir_name)

                logging.info('Extracting embedding from %s' % )
                ext = os.path.splitext(download_file_path)[1]
                if ext == '.zip':






        #self._dim = 0
        #self._token_to_idx = None
        #self._idx_to_token = None
        #self._idx_to_embed = None

    def __getitem__(self, token):
        if token in self.token_to_idx:
            return self.idx_to_embed[self.token_to_idx[token]]
        else:
            return self.unk_embed(nd.zeros(shape=self.dim))

    @property
    def dim(self):
        return self._dim

    @property
    def unk_embed(self):
        return self._unk_embed

    @property
    def token_to_idx(self):
        return self._token_to_idx

    @property
    def idx_to_token(self):
        return self._idx_to_token

    @property
    def idx_to_embed(self):
        return self._idx_to_embed

    @staticmethod
    def register(embed_cls):
        assert(isinstance(embed_cls, type))
        embed_name = embed_cls.__name__.lower()
        if embed_name in Embedding.embed_registry:
            logging.warning('WARNING: New embedding %s.%s is overriding '
                            'existing embedding %s.%s',
                            embed_cls.__module__, embed_cls.__name__,
                            Embedding.embed_registry[embed_name].__module__,
                            Embedding.embed_registry[embed_name].__name__)
        Embedding.embed_registry[embed_name] = embed_cls
        return embed_cls

    @staticmethod
    def create_embedding(embed_name, **kwargs):
        if embed_name.lower() in Embedding.embed_registry:
            return Embedding.embed_registry[embed_name.lower()](**kwargs)
        else:
            raise ValueError('Cannot find embedding %s. Valid embedding '
                             'names: %s' % (embed_name, ', '.join(
                                Embedding.embed_registry.keys())))

    @staticmethod
    def check_pretrain_names(pretrain_name, embed_name):
        embed_cls = Embedding.embed_registry[embed_name]
        if pretrain_name not in embed_cls.pretrain_names:
            raise KeyError('Cannot find pretrain name %s for embedding %s. '
                           'Valid pretrain names for embedding %s: %s' %
                           (pretrain_name, embed_name, embed_name,
                            ', '.join(embed_cls.pretrain_names)))

    @staticmethod
    def show_embed_and_pretrain_names():
        for embed_name, embed_cls in Embedding.embed_registry.items():
            print('embed_name: %s' % embed_name)
            print('pretrain_names: %s\n' % ', '.join(embed_cls.pretrain_names))


@Embedding.register
class GloVe(Embedding):
    pretrain_names = {
        '42B.300d', '6B.50d', '6B.100d', '6B.200d', '6B.300d', '840B.300d',
        'twitter.27B.25d', 'twitter.27B.50d', 'twitter.27B.100d',
        'twitter.27B.200d'
    }
    url_prefix = 'http://nlp.stanford.edu/data/'

    def __init__(self, pretrain_name='840B.300d', **kwargs):
        Embedding.check_pretrain_names(pretrain_name, GloVe.__name__.lower())
        url = self._get_url(pretrain_name)
        pretrain_file = 'glove.%s.txt' % pretrain_name
        super(GloVe, self).__init__(pretrain_file=pretrain_file, url=url,
                                    embed_name=GloVe.__name__.lower(), **kwargs)

    def _get_url(self, pretrain_name):
        zip_base = 'glove.%s.zip'
        zip_file = zip_base % pretrain_name

        first_token = pretrain_name.split('.')[0]
        if first_token == '6B':
            zip_file = zip_base % '6B'
        elif first_token == 'twitter':
            zip_file = zip_base % 'twitter.27B'

        return GloVe.url_prefix + zip_file


@Embedding.register
class FastText(Embedding):
    pretrain_names = {'wiki.en', 'wiki.simple', 'wiki.zh'}
    url_prefix = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/'

    def __init__(self, pretrain_name='wiki.en', **kwargs):
        Embedding.check_pretrain_names(pretrain_name, FastText.__name__.lower())
        pretrain_file = pretrain_name + '.vec'
        url = FastText.url_prefix + pretrain_file
        super(FastText, self).__init__(pretrain_file=pretrain_file, url=url,
                                       embed_name=FastText.__name__.lower(),
                                       **kwargs)
