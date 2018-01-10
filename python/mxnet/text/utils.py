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

"""Provide utilities for text data processing."""
from __future__ import absolute_import
from __future__ import print_function

from collections import Counter
import re

from . import constants as C
from .embedding import TokenIndexer


def count_tokens_from_str(source_str, token_delim=' ', seq_delim='\n',
                          to_lower=False, counter_to_update=None):
    """Counts tokens in the specified string.

    For token_delim='<td>' and seq_delim='<sd>', a specified string of two
    sequences of tokens may look like::

    <td>token1<td>token2<td>token3<td><sd><td>token4<td>token5<td><sd>


    Parameters
    ----------
    source_str : str
        A source string of tokens.
    token_delim : str, default ' '
        A token delimiter.
    seq_delim : str, default '\\\\n'
        A sequence delimiter.
    to_lower : bool, default False
        Whether to convert the source source_str to the lower case.
    counter_to_update : collections.Counter or None, default None
        The collections.Counter instance to be updated with the token counts
        of `source_str`. If None, return a new collections.Counter instance
        counting tokens from `source_str`.


    Returns
    -------
    collections.Counter
        The `counter_to_update` collections.Counter instance after being updated
        with the token counts of `source_str`. If `counter_to_update` is None,
        return a new collections.Counter instance counting tokens from
        `source_str`.


    Examples
    --------
    >>> source_str = ' Life is great ! \\n life is good . \\n'
    >>> count_tokens_from_str(token_line, ' ', '\\n', True)
    Counter({'!': 1, '.': 1, 'good': 1, 'great': 1, 'is': 2, 'life': 2})
    """

    source_str = filter(None,
                        re.split(token_delim + '|' + seq_delim, source_str))
    if to_lower:
        source_str = [t.lower() for t in source_str]

    if counter_to_update is None:
        return Counter(source_str)
    else:
        counter_to_update.update(source_str)
        return counter_to_update


def tokens_to_indices(tokens, indexer):
    """Converts tokens to indices according to the text indexer.


    Parameters
    ----------
    tokens : str or list of strs
        A source token or tokens to be converted.
    indexer : :class:`~mxnet.text.embeddings.TokenIndexer`
        A text indexer.


    Returns
    -------
    int or list of ints
        A token index or a list of token indices according to the text indexer.
    """

    assert isinstance(indexer, TokenIndexer), \
        '`indexer` must be an instance of `mxnet.text.embeddings.TokenIndexer`.'

    to_reduce = False
    if not isinstance(tokens, list):
        tokens = [tokens]
        to_reduce = True

    indices = [indexer.token_to_idx[token] if token in indexer.token_to_idx
               else C.UNKNOWN_IDX for token in tokens]

    return indices[0] if to_reduce else indices


def indices_to_tokens(indices, indexer):
    """Converts token indices to tokens according to the text indexer.


    Parameters
    ----------
    indices : int or list of ints
        A source token index or token indices to be converted.
    indexer : :class:`~mxnet.text.embeddings.TokenIndexer`
        A text indexer.


    Returns
    -------
    str or list of strs
        A token or a list of tokens according to the text indexer.
    """

    assert isinstance(indexer, TokenIndexer), \
        '`indexer` must be an instance of `mxnet.text.embeddings.TokenIndexer`.'

    to_reduce = False
    if not isinstance(indices, list):
        indices = [indices]
        to_reduce = True

    max_idx = len(indexer.idx_to_token) - 1

    tokens = []
    for idx in indices:
        if not isinstance(idx, int) or idx > max_idx:
            raise ValueError('Token index %d in the provided `indices` is '
                             'invalid.' % idx)
        else:
            tokens.append(indexer.idx_to_token[idx])

    return tokens[0] if to_reduce else tokens
