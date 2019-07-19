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
# pylint: disable=consider-iterating-dictionary

"""Text token indexer."""
from __future__ import absolute_import
from __future__ import print_function

import collections

from . import _constants as C


class Vocabulary(object):
    """Indexing for text tokens.


    Build indices for the unknown token, reserved tokens, and input counter keys. Indexed tokens can
    be used by token embeddings.


    Parameters
    ----------
    counter : collections.Counter or None, default None
        Counts text token frequencies in the text data. Its keys will be indexed according to
        frequency thresholds such as `most_freq_count` and `min_freq`. Keys of `counter`,
        `unknown_token`, and values of `reserved_tokens` must be of the same hashable type.
        Examples: str, int, and tuple.
    most_freq_count : None or int, default None
        The maximum possible number of the most frequent tokens in the keys of `counter` that can be
        indexed. Note that this argument does not count any token from `reserved_tokens`. Suppose
        that there are different keys of `counter` whose frequency are the same, if indexing all of
        them will exceed this argument value, such keys will be indexed one by one according to
        their __cmp__() order until the frequency threshold is met. If this argument is None or
        larger than its largest possible value restricted by `counter` and `reserved_tokens`, this
        argument has no effect.
    min_freq : int, default 1
        The minimum frequency required for a token in the keys of `counter` to be indexed.
    unknown_token : hashable object, default '&lt;unk&gt;'
        The representation for any unknown token. In other words, any unknown token will be indexed
        as the same representation. Keys of `counter`, `unknown_token`, and values of
        `reserved_tokens` must be of the same hashable type. Examples: str, int, and tuple.
    reserved_tokens : list of hashable objects or None, default None
        A list of reserved tokens that will always be indexed, such as special symbols representing
        padding, beginning of sentence, and end of sentence. It cannot contain `unknown_token`, or
        duplicate reserved tokens. Keys of `counter`, `unknown_token`, and values of
        `reserved_tokens` must be of the same hashable type. Examples: str, int, and tuple.


    Attributes
    ----------
    unknown_token : hashable object
        The representation for any unknown token. In other words, any unknown token will be indexed
        as the same representation.
    reserved_tokens : list of strs or None
        A list of reserved tokens that will always be indexed.
    """

    def __init__(self, counter=None, most_freq_count=None, min_freq=1, unknown_token='<unk>',
                 reserved_tokens=None):

        # Sanity checks.
        assert min_freq > 0, '`min_freq` must be set to a positive value.'

        if reserved_tokens is not None:
            reserved_token_set = set(reserved_tokens)
            assert unknown_token not in reserved_token_set, \
                '`reserved_token` cannot contain `unknown_token`.'
            assert len(reserved_token_set) == len(reserved_tokens), \
                '`reserved_tokens` cannot contain duplicate reserved tokens.'

        self._index_unknown_and_reserved_tokens(unknown_token, reserved_tokens)

        if counter is not None:
            self._index_counter_keys(counter, unknown_token, reserved_tokens, most_freq_count,
                                     min_freq)

    def _index_unknown_and_reserved_tokens(self, unknown_token, reserved_tokens):
        """Indexes unknown and reserved tokens."""

        self._unknown_token = unknown_token
        # Thus, constants.UNKNOWN_IDX must be 0.
        self._idx_to_token = [unknown_token]

        if reserved_tokens is None:
            self._reserved_tokens = None
        else:
            self._reserved_tokens = reserved_tokens[:]
            self._idx_to_token.extend(reserved_tokens)

        self._token_to_idx = {token: idx for idx, token in enumerate(self._idx_to_token)}

    def _index_counter_keys(self, counter, unknown_token, reserved_tokens, most_freq_count,
                            min_freq):
        """Indexes keys of `counter`.


        Indexes keys of `counter` according to frequency thresholds such as `most_freq_count` and
        `min_freq`.
        """

        assert isinstance(counter, collections.Counter), \
            '`counter` must be an instance of collections.Counter.'

        unknown_and_reserved_tokens = set(reserved_tokens) if reserved_tokens is not None else set()
        unknown_and_reserved_tokens.add(unknown_token)

        token_freqs = sorted(counter.items(), key=lambda x: x[0])
        token_freqs.sort(key=lambda x: x[1], reverse=True)

        token_cap = len(unknown_and_reserved_tokens) + (
            len(counter) if most_freq_count is None else most_freq_count)

        for token, freq in token_freqs:
            if freq < min_freq or len(self._idx_to_token) == token_cap:
                break
            if token not in unknown_and_reserved_tokens:
                self._idx_to_token.append(token)
                self._token_to_idx[token] = len(self._idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    @property
    def token_to_idx(self):
        """
        dict mapping str to int: A dict mapping each token to its index integer.
        """
        return self._token_to_idx

    @property
    def idx_to_token(self):
        """
        list of strs:  A list of indexed tokens where the list indices and the token indices are aligned.
        """
        return self._idx_to_token

    @property
    def unknown_token(self):
        return self._unknown_token

    @property
    def reserved_tokens(self):
        return self._reserved_tokens

    def to_indices(self, tokens):
        """Converts tokens to indices according to the vocabulary.


        Parameters
        ----------
        tokens : str or list of strs
            A source token or tokens to be converted.


        Returns
        -------
        int or list of ints
            A token index or a list of token indices according to the vocabulary.
        """

        to_reduce = False
        if not isinstance(tokens, list):
            tokens = [tokens]
            to_reduce = True

        indices = [self.token_to_idx[token] if token in self.token_to_idx
                   else C.UNKNOWN_IDX for token in tokens]

        return indices[0] if to_reduce else indices

    def to_tokens(self, indices):
        """Converts token indices to tokens according to the vocabulary.


        Parameters
        ----------
        indices : int or list of ints
            A source token index or token indices to be converted.


        Returns
        -------
        str or list of strs
            A token or a list of tokens according to the vocabulary.
        """

        to_reduce = False
        if not isinstance(indices, list):
            indices = [indices]
            to_reduce = True

        max_idx = len(self.idx_to_token) - 1

        tokens = []
        for idx in indices:
            if not isinstance(idx, int) or idx > max_idx: # pylint: disable=no-else-raise
                raise ValueError('Token index %d in the provided `indices` is invalid.' % idx)
            else:
                tokens.append(self.idx_to_token[idx])

        return tokens[0] if to_reduce else tokens
