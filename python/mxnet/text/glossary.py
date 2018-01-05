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

"""Index text tokens and load their embeddings."""
from __future__ import absolute_import
from __future__ import print_function

from .. import ndarray as nd
from .embedding import TokenEmbedding


class Glossary(TokenEmbedding):
    """Indexing and embedding for text tokens in a glossary.

    For each indexed token in a glossary, an embedding vector will be associated
    with it. Such embedding vectors can be loaded from externally hosted or
    custom pre-trained text embedding files, such as via instances of
    :func:`~mxnet.text.embedding.TokenEmbedding`.


    Parameters
    ----------
    counter : collections.Counter or None, default None
        Counts text token frequencies in the text data. Its keys will be indexed
        according to frequency thresholds such as `most_freq_count` and
        `min_freq`. Keys of `counter`, `unknown_token`, and  values of
        `reserved_tokens` must be the same type with __hash__() and __cmp__().
        Examples: str, int, and typle.
    embeds : an instance or a list of instances of
        :func:`~mxnet.text.embedding.TextEmbed`
        One or multiple pre-trained text embeddings to load. If it is a list of
        multiple embeddings, these embedding vectors will be concatenated for
        each token.
    most_freq_count : None or int, default None
        The maximum possible number of the most frequent tokens in the keys of
        `counter` that can be indexed. Note that this argument does not count
        any token from `reserved_tokens`. If this argument is None or larger
        than its largest possible value restricted by `counter` and
        `reserved_tokens`, this argument becomes positive infinity.
    min_freq : int, default 1
        The minimum frequency required for a token in the keys of `counter` to
        be indexed.
    unknown_token : type with __hash__() and __cmp__(), default '<unk>'
        The representation for any unknown token. In other words, any unknown
        token will be indexed as the same representation. Keys of `counter`,
        `unknown_token`, and  values of `reserved_tokens` must be the same type
        with __hash__() and __cmp__(). Examples: str, int, and typle.
    reserved_tokens : list of types with __hash__() and __cmp__() or None,
        default None
        A list of reserved tokens that will always be indexed, such as special
        symbols representing padding, beginning of sentence, and end of
        sentence. It cannot contain `unknown_token`, or duplicate reserved
        tokens. Keys of `counter`, `unknown_token`, and values of
        `reserved_tokens` must be the same type with __hash__() and __cmp__().
        Examples: str, int, and typle.

    """
    def __init__(self, counter, token_embeddings, most_freq_count=None,
                 min_freq=1, unknown_token='<unk>', reserved_tokens=None):

        if not isinstance(token_embeddings, list):
            token_embeddings = [token_embeddings]

        # Sanity checks.
        for embed in token_embeddings:
            assert isinstance(embed, TokenEmbedding), \
                'The parameter `token_embeddings` must be an instance or a ' \
                'list of instances of `mxnet.text.embedding.TextEmbed` ' \
                'whose embedding vectors will be loaded or ' \
                'concatenated-then-loaded to map to the indexed tokens.'

        # Index tokens from keys of `counter` and reserved tokens.
        super(Glossary, self).__init__(counter=counter,
                                       most_freq_count=most_freq_count,
                                       min_freq=min_freq,
                                       unknown_token=unknown_token,
                                       reserved_tokens=reserved_tokens)

        # Set _idx_to_vec so that indices of tokens from keys of `counter` are
        # associated with text embedding vectors from `token_embeddings`.
        self._set_idx_to_vec_by_embeds(token_embeddings)

    def _set_idx_to_vec_by_embeds(self, token_embeddings):
        """Sets the mapping between token indices and token embedding vectors.


        Parameters
        ----------
        token_embeddings : an instance or a list of instances of
            :func:`~mxnet.text.embedding.TextEmbed`
            One or multiple pre-trained text embeddings to load. If it is a list
            of multiple embeddings, these embedding vectors will be concatenated
            for each token.
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
