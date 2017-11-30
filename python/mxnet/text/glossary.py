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
from .. import ndarray as nd
from .. import io


class Glossary(object):
    ## keys of counter cannot contain symbols.
    def __init__(self, counter=None, top_k_freq=None, min_freq=1,
                 symbols=['<unk>'], loaded_embeds=None):
        # Sanity check.
        if min_freq < 1:
            raise ValueError('`min_freq` must be set to a value that is '
                             'greater than or equal to 1.')
        if len(symbols) == 0:
            raise ValueError('`symbols` must be an non-empty list whose first '
                             'element is the string representation for unknown '
                             'tokens, such as "<unk>".')

        # Initialize attributes.
        self._counter = counter.copy()
        self._token_to_idx = {token: idx for idx, token in enumerate(symbols)}
        self._idx_to_token = symbols.copy()
        self._idx_to_embed = None

        # Update _counter to include special symbols, such as '<unk>'.
        self._counter.update({token: 0 for token in symbols})

        # Update _token_to_idx and _idx_to_token according to specified
        # frequency thresholds.
        token_freqs = sorted(self._counter.items(), key=lambda x: x[0])
        token_freqs.sort(key=lambda x: x[1], reverse=True)

        token_cap = len(self._counter) if top_k_freq is None \
            else len(symbols) + top_k_freq

        for token, freq in token_freqs:
            if freq < min_freq or len(self._idx_to_token) == token_cap:
                break
            self._idx_to_token.append(token)
            self._token_to_idx[token] = len(self._idx_to_token) - 1

        if loaded_embeds is not None:
            self.load_embeds(loaded_embeds)


    # TODO: do we need to compare all of the four members?
    def __eq__(self, other):
        if self.counter != other.counter:
            return False
        if self.token_to_idx != other.token_to_idx:
            return False
        if self.idx_to_token != other.idx_to_token:
            return False
        if self.idx_to_embed != other.idx_to_embed:
            return False
        return True

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

    def load_embeds(self, loaded_embeds):
        # set_idx_to_embed()
        return 0


class Embedding(object):
    embed_registry = {}

    def __init__(self, pretrain_file, url=None, embed_dir='.embeddings'):
        print(Embedding.embed_registry)
        print(list(Embedding.embed_registry.keys()))
        print(embed_dir)
        print(pretrain_file)

    @staticmethod
    def register(embed):
        assert (isinstance(embed, type))
        name = embed.__name__.lower()
        if name in Embedding.embed_registry:
            logging.warning('WARNING: New embedding %s.%s is overriding '
                            'existing embedding %s.%s',
                            embed.__module__, embed.__name__,
                            Embedding.embed_registry[name].__module__,
                            Embedding.embed_registry[name].__name__)
        Embedding.embed_registry[name] = embed
        return embed

    @staticmethod
    def create_embedding(name, **kwargs):
        if name.lower() in Embedding.embed_registry:
            return Embedding.embed_registry[name.lower()](**kwargs)
        else:
            raise ValueError('Cannot find embedding %s. Valid embedding '
                             'names: %s' %
                             (name, tuple(Embedding.embed_registry.keys())))

    @staticmethod
    def fetch_url(src, embed_name):
        embed = Embedding.embed_registry[embed_name]
        if src in embed.src_to_url:
            return embed.src_to_url[src]
        else:
            raise KeyError('Cannot find source name %s for embedding %s. '
                           'Valid source names for embedding %s: %s' %
                           (src, embed.__name__.lower(),
                            embed.__name__.lower(),
                            tuple(embed.src_to_url.keys())))


@Embedding.register
class GloVe(Embedding):
    src_to_url = {
        '42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
        '6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
        '840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
        'twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip'
    }

    def __init__(self, src='840B', dim=300, **kwargs):
        url = Embedding.fetch_url(src, GloVe.__name__.lower())
        pretrain_file = 'glove.%s.%sd.txt' % (src, str(dim))
        super(GloVe, self).__init__(pretrain_file=pretrain_file, url=url,
                                    **kwargs)


@Embedding.register
class FastText(Embedding):
    # There may be growing sources for pre-trained FastText embeddings.
    src_to_url = {
        'wiki': 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.'
    }

    def __init__(self, src='wiki', lang="en", **kwargs):
        url = Embedding.fetch_url(src, FastText.__name__.lower())
        url += '%s.vec' % lang
        pretrain_file = os.path.basename(url)
        super(FastText, self).__init__(pretrain_file=pretrain_file, url=url,
                                       **kwargs)
