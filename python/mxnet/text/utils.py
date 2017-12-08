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

"""Provide text utilities."""
from __future__ import absolute_import
from __future__ import print_function

from .glossary import Glossary

from collections import Counter
import os
import re


def count_tokens_from_str(tokens, token_delim=' ', seq_delim='\n',
                          to_lower=False, counter_to_add=Counter()):
    """Counts tokens, such as words and punctuations, in the specified string.

    For token_delim="<td>" and seq_delim="<sd>", a specified string of two
    sequences of tokens may look like::

    <td>token1<td>token2<td>token3<td><sd><td>token4<td>token5<td><sd>


    Parameters
    ----------
    tokens : str
        A source string of tokens.
    token_delim : str, default ' '
        A token delimiter.
    seq_delim : str, default '\n'
        A sequence delimiter.
    to_lower : bool, default False
        Whether to convert the source tokens to the lower case.
    counter_to_add : collections.Counter, default collections.Counter()
        A collections.Counter instance to update the output collections.Counter
        object.


    Returns
    -------
    collections.Counter
        A collections.Counter instance containing the frequency for each token
        in the source string, after it is updated with `counter_to_add`.


    Examples
    --------
    >>> tokens = ' Life is great ! \n life is good . \n'
    >>> count_tokens_from_str(token_line, ' ', '\n', True)
    Counter({'!': 1, '.': 1, 'good': 1, 'great': 1, 'is': 2, 'life': 2})
    """

    tokens = filter(None,
                    re.split(token_delim + '|' + seq_delim, tokens))
    if to_lower:
        tokens = [t.lower() for t in tokens]
    counter = Counter(tokens)
    counter.update(counter_to_add)
    return counter


def count_tokens_from_path(path, token_delim=' ', seq_delim='\n',
                           to_lower=False, counter_to_add=Counter()):
    """Counts tokens, such as words and punctuations, in the specified path.

    For token_delim="<td>" and seq_delim="<sd>", any file of two sequences of
    tokens in the specified path may look like::

    <td>token1<td>token2<td>token3<td><sd><td>token4<td>token5<td><sd>


    Parameters
    ----------
    path : str
        A string path to a file or multiple files of sequences of tokens.
    token_delim : str, default ' '
        A token delimiter.
    seq_delim : str, default '\n'
        A sequence delimiter.
    to_lower : bool, default False
        Whether to convert the source tokens to the lower case.
    counter_to_add : collections.Counter, default collections.Counter()
        A collections.Counter instance to update the output collections.Counter
        instance.


    Returns
    -------
    collections.Counter
        A collections.Counter instance containing the counts for each token in
        the source path, after it is updated with `counter_to_add`.


    Examples
    --------
    >>> See `count_tokens_from_str`.
    """

    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path)
                 if os.path.isfile(os.path.join(path, f))]
    elif os.path.isfile(path):
        files = [path]
    else:
        raise IOError('%s is not a valid path to a directory of files or a '
                      'single file.', path)

    file_strs = []

    for f in files:
        try:
            with open(f) as fin:
                file_strs.append(fin.read())
        except IOError:
            raise IOError('%s contains or is a file that cannot be read.', path)

    return count_tokens_from_str(''.join(file_strs), token_delim, seq_delim,
                                 to_lower, counter_to_add)


def tokens_to_indices(tokens, glossary):
    """Converts tokens to indices according to the glossary indexing.


    Parameters
    ----------
    tokens : str or list of strs
        A source token or tokens to be converted.
    glossary : mxnet.text.Glossary
        A glossary instance.


    Returns
    -------
    int or list of ints
        A token index or a list of token indices according to the glossary
        indexing.
    """

    assert isinstance(glossary, Glossary), \
        '`glossary` must be an instance of mxnet.text.Glossary.'

    to_reduce = False
    if not isinstance(tokens, list):
        tokens = [tokens]
        to_reduce = True

    indices = [glossary.token_to_idx[token] if token in glossary.token_to_idx
               else glossary.unk_idx() for token in tokens]

    return indices[0] if to_reduce else indices


def indices_to_tokens(indices, glossary):
    """Converts token indices to tokens according to the glossary indexing.


    Parameters
    ----------
    indices : int or list of ints
        A source token index or token indices to be converted.
    glossary : mxnet.text.Glossary
        A glossary instance.


    Returns
    -------
    str or list of strs
        A token or a list of tokens according to the glossary indexing.
    """

    assert isinstance(glossary, Glossary), \
        '`glossary` must be an instance of mxnet.text.Glossary.'

    to_reduce = False
    if not isinstance(indices, list):
        indices = [indices]
        to_reduce = True

    max_idx = len(glossary.idx_to_token) - 1

    tokens = []
    for idx in indices:
        if not isinstance(idx, int) or idx > max_idx:
            raise ValueError('Token index %d in the provided `indices` is '
                             'invalid.' % idx)
        else:
            tokens.append(glossary.idx_to_token[idx])

    return tokens[0] if to_reduce else tokens
