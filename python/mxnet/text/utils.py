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

from collections import Counter

import os
import re


def count_tokens_from_str(tokens, token_delim=" ", seq_delim="\n",
                          to_lower=False):
    """Counts tokens, such as words and punctuations, in the specified string.

    For token_delim="<td>" and seq_delim="<sd>", a specified string of two
    sequences of tokens may look like::

    <td>token1<td>token2<td>token3<td><sd><td>token4<td>token5<td><sd>


    Parameters
    ----------
    tokens : str
        The source string of tokens.
    token_delim : str
        The token delimiter.
    seq_delim : str
        The sequence delimiter.
    to_lower : bool
        Whether to convert the source tokens to the lower case.


    Returns
    -------
    Counter
        A `Counter` containing the frequency of each token in the source string.

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
    return counter


def count_tokens_from_path(path, token_delim=' ', seq_delim='\n',
                           to_lower=False):
    """Counts tokens, such as words and punctuations, in the specified path.

     For token_delim="<td>" and seq_delim="<sd>", any file of two sequences of
     tokens in the specified path may look like::

     <td>token1<td>token2<td>token3<td><sd><td>token4<td>token5<td><sd>


     Parameters
     ----------
     path : str
         The string path to a file or multiple files of sequences of tokens.
     token_delim : str
         The token delimiter.
     seq_delim : str
         The sequence delimiter.
     to_lower : bool
         Whether to convert the source tokens to the lower case.


     Returns
     -------
     Counter
         A `Counter` containing the frequency of each token in the source path.

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
                                 to_lower)
