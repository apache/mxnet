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
# pylint: disable=super-init-not-called

"""Index text tokens and load their embeddings."""
from __future__ import absolute_import
from __future__ import print_function

from . import embedding
from . import indexer
from ... import ndarray as nd


class Glossary(embedding.TokenEmbedding):
    """Indexing and embedding for text tokens in a glossary.


    For each indexed token in a glossary, an embedding vector will be associated with it. Such
    embedding vectors can be loaded from externally hosted or custom pre-trained token embedding
    files, such as via instances of :class:`~mxnet.contrib.text.embedding.TokenEmbedding`.


    Parameters
    ----------
    token_indexer : :class:`~mxnet.contrib.text.indexer.TokenIndexer`
        It contains the indexed tokens to load, where each token is associated with an index.
    token_embeddings : instance or list of :class:`~TokenEmbedding`
        One or multiple pre-trained token embeddings to load. If it is a list of multiple
        embeddings, these embedding vectors will be concatenated for each token.


    Properties
    ----------
    token_to_idx : dict mapping str to int
        A dict mapping each token to its index integer.
    idx_to_token : list of strs
        A list of indexed tokens where the list indices and the token indices are aligned.
    unknown_token : hashable object
        The representation for any unknown token. In other words, any unknown token will be indexed
        as the same representation.
    reserved_tokens : list of strs or None
        A list of reserved tokens that will always be indexed.
    vec_len : int
        The length of the embedding vector for each token.
    idx_to_vec : mxnet.ndarray.NDArray
        For all the indexed tokens in this embedding, this NDArray maps each token's index to an
        embedding vector. The largest valid index maps to the initialized embedding vector for every
        reserved token, such as an unknown_token token and a padding token.
    """
    def __init__(self, token_indexer, token_embeddings):

        # Sanity checks.
        assert isinstance(token_indexer, indexer.TokenIndexer), \
            'The argument `token_indexer` must be an instance of ' \
            'mxnet.contrib.text.indexer.TokenIndexer.'

        if not isinstance(token_embeddings, list):
            token_embeddings = [token_embeddings]

        for embed in token_embeddings:
            assert isinstance(embed, embedding.TokenEmbedding), \
                'The argument `token_embeddings` must be an instance or a list of instances ' \
                'of `mxnet.contrib.text.embedding.TextEmbedding` whose embedding vectors will be' \
                'loaded or concatenated-then-loaded to map to the indexed tokens.'

        # Index tokens.
        self._token_to_idx = token_indexer.token_to_idx.copy() \
            if token_indexer.token_to_idx is not None else None
        self._idx_to_token = token_indexer.idx_to_token[:] \
            if token_indexer.idx_to_token is not None else None
        self._unknown_token = token_indexer.unknown_token
        self._reserved_tokens = token_indexer.reserved_tokens[:] \
            if token_indexer.reserved_tokens is not None else None

        # Set _idx_to_vec so that indices of tokens from keys of `counter` are
        # associated with token embedding vectors from `token_embeddings`.
        self._set_idx_to_vec_by_embeds(token_embeddings)

    def _set_idx_to_vec_by_embeds(self, token_embeddings):
        """Sets the mapping between token indices and token embedding vectors.


        Parameters
        ----------
        token_embeddings : an instance or a list of instances of
            :class:`~mxnet.contrib.text.embedding.TokenEmbedding`
            One or multiple pre-trained token embeddings to load. If it is a list of multiple
            embeddings, these embedding vectors will be concatenated for each token.
        """

        self._vec_len = sum(embed.vec_len for embed in token_embeddings)
        self._idx_to_vec = nd.zeros(shape=(len(self), self.vec_len))

        col_start = 0
        # Concatenate all the embedding vectors in token_embeddings.
        for embed in token_embeddings:
            col_end = col_start + embed.vec_len
            # Cancatenate vectors of the unknown token.
            self._idx_to_vec[0, col_start:col_end] = embed.idx_to_vec[0]
            self._idx_to_vec[1:, col_start:col_end] = embed.get_vecs_by_tokens(
                self.idx_to_token[1:])
            col_start = col_end
