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
# pylint: disable=not-callable, invalid-encoded-data, dangerous-default-value
# pylint: disable=logging-not-lazy, consider-iterating-dictionary
# pylint: disable=non-parent-init-called, super-init-not-called

"""Read text files and load embeddings."""
from __future__ import absolute_import
from __future__ import print_function

from .. import ndarray as nd
from .embedding import TextEmbed
from .embedding import TextIndexer


class Glossary(TextEmbed):
    """Indexing and embedding for text and reserved tokens in a glossary.

    For each indexed text or reserved token (e.g., an unknown_token token) in a
    glossary, an embedding vector will be associated with the token. Such
    embedding vectors can be loaded from externally pre-trained embeddings,
    such as via mxnet.text.glossary.TextEmbed instances.


    Parameters
    ----------
    counter : collections.Counter
        Counts text token frequencies in the text data.
    most_freq_count : None or int, default None
        The number of most frequent tokens in the keys of `counter` that will be
        indexed. If None or larger than the cardinality of the keys of
        `counter`, all the tokens in the keys of `counter` will be indexed.
    min_freq : int, default 1
        The minimum frequency required for a token in the keys of `counter` to
        be indexed.
    unknown_token : str, default '<unk>'
        The string representation for any unknown_token token. It is a reserved token
        to be indexed.
    reserveds : list of strs or None, default None
        A list of other reserved tokens to be indexed.  It cannot contain any
        token from the keys of `counter`. If None, there is no reserved token
        other than the reserved unknown_token token `unknown_token`.
    embeds : an mxnet.text.glossary.TextEmbed instance, a list of
        mxnet.text.glossary.TextEmbed instances, or None, default None
        Pre-trained embeddings to load. If None, there is nothing to load.
    """
    def __init__(self, counter, embeds, most_freq_count=None, min_freq=1,
                 unknown_token='<unk>', reserved_tokens=None):

        if not isinstance(embeds, list):
            embeds = [embeds]

        # Sanity checks.
        for embed in embeds:
            assert isinstance(embed, TextEmbed), \
                'The parameter `embeds` must be a ' \
                'mxnet.text.embedding.TextEmbed instance or a list of ' \
                'mxnet.text.embedding.TextEmbed instances whose embedding ' \
                'vectors will be loaded or concatenated then loaded to map ' \
                'to the indexed tokens from keys of `counter`.'

        # Index tokens from keys of `counter` and reserved tokens.
        TextIndexer.__init__(self, counter=counter, most_freq_count=most_freq_count,
                             min_freq=min_freq, unknown_token=unknown_token,
                             reserved_tokens=reserved_tokens)

        # Set idx_to_vec so that indices of tokens from keys of `counter` are
        # associated with text embedding vectors from `embeds`.
        self.set_idx_to_vec_by_embeds(embeds)

    def set_idx_to_vec_by_embeds(self, embeds):
        """Sets the mapping between token indices and token embedding vectors.


        Parameters
        ----------
        embeds : mxnet.text.glossary.TextEmbed or list of
            mxnet.text.glossary.TextEmbed instances. If it is a list of
            mxnet.text.glossary.TextEmbed instances, their embedding vectors
            are concatenated for each token.
        """

        if not isinstance(embeds, list):
            embeds = [embeds]

        # Sanity check.
        for embed in embeds:
            assert isinstance(embed, TextEmbed), \
                'The parameter `embeds` must be a ' \
                'mxnet.text.glossary.TextEmbed instance or a list of ' \
                'mxnet.text.glossary.TextEmbed instances whose embedding ' \
                'vectors will be loaded or concatenated then loaded to map ' \
                'to the indexed tokens from keys of `counter`.'

            assert self.unknown_token == embed.unknown_token, \
                'The `unknown_token` of the mxnet.text.glossary.TextEmbed ' \
                'instance %s must be the same'

        self._vec_len = sum(embed.vec_len for embed in embeds)
        self._idx_to_vec = nd.zeros(shape=(len(self), self.vec_len))

        col_start = 0
        # Concatenate all the embedding vectors in embeds.
        for embed in embeds:
            col_end = col_start + embed.vec_len
            self._idx_to_vec[:, col_start:col_end] = embed[self.idx_to_token]
            col_start = col_end
