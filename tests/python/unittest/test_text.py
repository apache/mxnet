# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# 'License'); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

from collections import Counter
import os

from mxnet.test_utils import *
from mxnet.text import utils as tu
from mxnet.text.glossary import Glossary
from mxnet.text.glossary import TextEmbed


def _get_test_str_of_tokens(token_delim, seq_delim):
    seq1 = token_delim + token_delim.join(['Life', 'is', 'great', '!']) \
           + token_delim + seq_delim
    seq2 = token_delim + token_delim.join(['life', 'is', 'good', '.']) \
           + token_delim + seq_delim
    seq3 = token_delim + token_delim.join(['life', "isn't", 'bad', '.']) \
           + token_delim + seq_delim
    seqs = seq1 + seq2 + seq3
    return seqs


def _mk_dir_of_files(path, token_delim, seq_delim):
    if not os.path.exists(path):
        os.makedirs(path)
    seqs = _get_test_str_of_tokens(token_delim, seq_delim)

    with open(os.path.join(path, '1.txt'), 'w') as fout:
        fout.write(seqs)
    with open(os.path.join(path, '2.txt'), 'w') as fout:
        for _ in range(2):
            fout.write(seqs)


def _test_count_tokens_from_str_with_delims(token_delim, seq_delim):
    str_of_tokens = _get_test_str_of_tokens(token_delim, seq_delim)

    cnt1 = tu.count_tokens_from_str(str_of_tokens, token_delim, seq_delim,
                                    to_lower=False)
    assert cnt1 == Counter(
        {'is': 2, 'life': 2, '.': 2, 'Life': 1, 'great': 1, '!': 1, 'good': 1,
         "isn't": 1, 'bad': 1})

    cnt2 = tu.count_tokens_from_str(str_of_tokens, token_delim, seq_delim,
                                    to_lower=True)
    assert cnt2 == Counter(
        {'life': 3, 'is': 2, '.': 2, 'great': 1, '!': 1, 'good': 1,
         "isn't": 1, 'bad': 1})

    counter_to_add = Counter({'life': 2})

    cnt3 = tu.count_tokens_from_str(str_of_tokens, token_delim, seq_delim,
                                    to_lower=False,
                                    counter_to_add=counter_to_add)
    assert cnt3 == Counter(
        {'is': 2, 'life': 4, '.': 2, 'Life': 1, 'great': 1, '!': 1, 'good': 1,
         "isn't": 1, 'bad': 1})

    cnt4 = tu.count_tokens_from_str(str_of_tokens, token_delim, seq_delim,
                                    to_lower=True,
                                    counter_to_add=counter_to_add)
    assert cnt4 == Counter(
        {'life': 5, 'is': 2, '.': 2, 'great': 1, '!': 1, 'good': 1,
         "isn't": 1, 'bad': 1})


def test_count_tokens_from_str():
    _test_count_tokens_from_str_with_delims(' ', '\n')
    _test_count_tokens_from_str_with_delims('IS', 'LIFE')


def _test_count_tokens_from_path_with_delims(path, token_delim, seq_delim):
    _mk_dir_of_files(path, token_delim, seq_delim)
    file1 = os.path.join(path, '1.txt')

    cnt1 = tu.count_tokens_from_path(path, token_delim=token_delim,
                                     seq_delim=seq_delim, to_lower=False)
    assert cnt1 == Counter(
        {'is': 6, 'life': 6, '.': 6, 'Life': 3, 'great': 3, '!': 3, 'good': 3,
         "isn't": 3, 'bad': 3})

    cnt2 = tu.count_tokens_from_path(path, token_delim=token_delim,
                                     seq_delim=seq_delim, to_lower=True)
    assert cnt2 == Counter(
        {'life': 9, 'is': 6, '.': 6, 'great': 3, '!': 3, 'good': 3,
         "isn't": 3, 'bad': 3})

    cnt3 = tu.count_tokens_from_path(file1, token_delim=token_delim,
                                     seq_delim=seq_delim, to_lower=False)
    assert cnt3 == Counter(
        {'is': 2, 'life': 2, '.': 2, 'Life': 1, 'great': 1, '!': 1, 'good': 1,
         "isn't": 1, 'bad': 1})

    counter_to_add = Counter({'life': 1, 'Life': 1})

    cnt4 = tu.count_tokens_from_path(path, token_delim=token_delim,
                                     seq_delim=seq_delim, to_lower=False,
                                     counter_to_add=counter_to_add)
    assert cnt4 == Counter(
        {'is': 6, 'life': 7, '.': 6, 'Life': 4, 'great': 3, '!': 3, 'good': 3,
         "isn't": 3, 'bad': 3})

    cnt5 = tu.count_tokens_from_path(path, token_delim=token_delim,
                                     seq_delim=seq_delim, to_lower=True,
                                     counter_to_add=counter_to_add)
    assert cnt5 == Counter(
        {'life': 10, 'is': 6, '.': 6, 'great': 3, '!': 3, 'good': 3,
         "isn't": 3, 'bad': 3, 'Life': 1})

    cnt6 = tu.count_tokens_from_path(file1, token_delim=token_delim,
                                     seq_delim=seq_delim, to_lower=False,
                                     counter_to_add=counter_to_add)
    assert cnt6 == Counter(
        {'is': 2, 'life': 3, '.': 2, 'Life': 2, 'great': 1, '!': 1, 'good': 1,
         "isn't": 1, 'bad': 1})


def test_count_tokens_from_path():
    path = os.path.join('./data', 'test_texts')
    _test_count_tokens_from_path_with_delims(path, ' ', '\n')
    _test_count_tokens_from_path_with_delims(path, 'IS', 'LIFE')


def test_check_pretrain_files():
    for embed_name, embed_cls in TextEmbed.embed_registry.items():
        for pretrain_file in embed_cls.pretrain_file_sha1.keys():
            TextEmbed.check_pretrain_files(pretrain_file, embed_name)


def test_text_embed():
    fasttext_simple = TextEmbed.create_text_embed(
        'fasttext', pretrain_file='wiki.simple.vec')
    assert len(fasttext_simple) == 111051




if __name__ == '__main__':
    import nose
    nose.runmodule()
