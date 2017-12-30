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
import unittest

from common import assertRaises
from mxnet import ndarray as nd
from mxnet.test_utils import *
from mxnet.text import utils as tu
from mxnet.text.glossary import Glossary
from mxnet.text.embedding import TextIndexer, TextEmbed, CustomEmbed


def _get_test_str_of_tokens(token_delim, seq_delim):
    seq1 = token_delim + token_delim.join(['Life', 'is', 'great', '!']) \
           + token_delim + seq_delim
    seq2 = token_delim + token_delim.join(['life', 'is', 'good', '.']) \
           + token_delim + seq_delim
    seq3 = token_delim + token_delim.join(['life', "isn't", 'bad', '.']) \
           + token_delim + seq_delim
    seqs = seq1 + seq2 + seq3
    return seqs


def _test_count_tokens_from_str_with_delims(token_delim, seq_delim):
    source_str = _get_test_str_of_tokens(token_delim, seq_delim)

    cnt1 = tu.count_tokens_from_str(source_str, token_delim, seq_delim,
                                    to_lower=False)
    assert cnt1 == Counter(
        {'is': 2, 'life': 2, '.': 2, 'Life': 1, 'great': 1, '!': 1, 'good': 1,
         "isn't": 1, 'bad': 1})

    cnt2 = tu.count_tokens_from_str(source_str, token_delim, seq_delim,
                                    to_lower=True)
    assert cnt2 == Counter(
        {'life': 3, 'is': 2, '.': 2, 'great': 1, '!': 1, 'good': 1,
         "isn't": 1, 'bad': 1})

    counter_to_update = Counter({'life': 2})

    cnt3 = tu.count_tokens_from_str(source_str, token_delim, seq_delim,
                                    to_lower=False,
                                    counter_to_update=counter_to_update.copy())
    assert cnt3 == Counter(
        {'is': 2, 'life': 4, '.': 2, 'Life': 1, 'great': 1, '!': 1, 'good': 1,
         "isn't": 1, 'bad': 1})

    cnt4 = tu.count_tokens_from_str(source_str, token_delim, seq_delim,
                                    to_lower=True,
                                    counter_to_update=counter_to_update.copy())
    assert cnt4 == Counter(
        {'life': 5, 'is': 2, '.': 2, 'great': 1, '!': 1, 'good': 1,
         "isn't": 1, 'bad': 1})


def test_count_tokens_from_str():
    _test_count_tokens_from_str_with_delims(' ', '\n')
    _test_count_tokens_from_str_with_delims('IS', 'LIFE')


def test_tokens_to_indices():
    counter = Counter(['a', 'b', 'b', 'c', 'c', 'c', 'some_word$'])

    indexer = TextIndexer(counter, most_freq_count=None, min_freq=1,
                          unknown_token='<unk>', reserved_tokens=None)

    i1 = tu.tokens_to_indices('c', indexer)
    assert i1 == 1

    i2 = tu.tokens_to_indices(['c'], indexer)
    assert i2 == [1]

    i3 = tu.tokens_to_indices(['<unk>', 'non-exist'], indexer)
    assert i3 == [0, 0]

    i4 = tu.tokens_to_indices(['a', 'non-exist', 'a', 'b'], indexer)
    assert i4 == [3, 0, 3, 2]


def test_indices_to_tokens():
    counter = Counter(['a', 'b', 'b', 'c', 'c', 'c', 'some_word$'])

    indexer = TextIndexer(counter, most_freq_count=None, min_freq=1,
                          unknown_token='<unknown>', reserved_tokens=None)

    i1 = tu.indices_to_tokens(1, indexer)
    assert i1 == 'c'

    i2 = tu.indices_to_tokens([1], indexer)
    assert i2 == ['c']

    i3 = tu.indices_to_tokens([0, 0], indexer)
    assert i3 == ['<unknown>', '<unknown>']

    i4 = tu.indices_to_tokens([3, 0, 3, 2], indexer)
    assert i4 == ['a', '<unknown>', 'a', 'b']

    assertRaises(ValueError, tu.indices_to_tokens, 100, indexer)


def test_check_pretrain_files():
    for embed_name, embed_cls in TextEmbed.embed_registry.items():
        for pretrain_file in embed_cls.pretrain_file_sha1.keys():
            TextEmbed.check_pretrain_files(pretrain_file, embed_name)


def test_glove():
    glove_6b_50d = TextEmbed.create('glove', pretrain_file='glove.6B.50d.txt')

    assert len(glove_6b_50d) == 400001
    assert glove_6b_50d.vec_len == 50
    assert glove_6b_50d.token_to_idx['hi'] == 11084
    assert glove_6b_50d.idx_to_token[11084] == 'hi'

    first_vec_sum = glove_6b_50d.idx_to_vec[0].sum().asnumpy()[0]
    assert_almost_equal(first_vec_sum, 0)

    unk_vec_sum = glove_6b_50d['<unk$unk@unk>'].sum().asnumpy()[0]
    assert_almost_equal(unk_vec_sum, 0)

    unk_vecs_sum = glove_6b_50d[['<unk$unk@unk>',
                                 '<unk$unk@unk>']].sum().asnumpy()[0]
    assert_almost_equal(unk_vecs_sum, 0)


def test_fasttext():
    fasttext_simple = TextEmbed.create('fasttext',
                                       pretrain_file='wiki.simple.vec',
                                       unknown_vec=nd.ones)

    assert len(fasttext_simple) == 111052
    assert fasttext_simple.vec_len == 300
    assert fasttext_simple.token_to_idx['hi'] == 3241
    assert fasttext_simple.idx_to_token[3241] == 'hi'

    first_vec_sum = fasttext_simple.idx_to_vec[0].sum().asnumpy()[0]
    assert_almost_equal(first_vec_sum, fasttext_simple.vec_len)

    unk_vec_sum = fasttext_simple['<unk$unk@unk>'].sum().asnumpy()[0]
    assert_almost_equal(unk_vec_sum, fasttext_simple.vec_len)

    unk_vecs_sum = fasttext_simple[['<unk$unk@unk>',
                                    '<unk$unk@unk>']].sum().asnumpy()[0]
    assert_almost_equal(unk_vecs_sum, fasttext_simple.vec_len * 2)


def test_all_embeds():
    for embed_name, embed_cls in TextEmbed.embed_registry.items():
        print('embed_name: %s' % embed_name)
        for pretrain_file in embed_cls.pretrain_file_sha1.keys():

            print('pretrain_file: %s' % pretrain_file)
            te = TextEmbed.create(embed_name,
                                  pretrain_file=pretrain_file)
            print(len(te))


def _mk_my_pretrain_file(path, token_delim, pretrain_file):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.makedirs(path)
    seq1 = token_delim.join(['a', '0.1', '0.2', '0.3', '0.4', '0.5']) + '\n'
    seq2 = token_delim.join(['b', '0.6', '0.7', '0.8', '0.9', '1.0']) + '\n'
    seqs = seq1 + seq2
    with open(os.path.join(path, pretrain_file), 'w') as fout:
        fout.write(seqs)


def _mk_my_pretrain_file2(path, token_delim, pretrain_file):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.makedirs(path)
    seq1 = token_delim.join(['a', '0.01', '0.02', '0.03', '0.04', '0.05']) \
           + '\n'
    seq2 = token_delim.join(['c', '0.06', '0.07', '0.08', '0.09', '0.1']) \
           + '\n'
    seqs = seq1 + seq2
    with open(os.path.join(path, pretrain_file), 'w') as fout:
        fout.write(seqs)


def test_custom_embed():
    embed_root = '~/.mxnet/embeddings/'
    embed_name = 'my_embed'
    elem_delim = '/t'
    pretrain_file = 'my_pretrain_file.txt'

    _mk_my_pretrain_file(os.path.join(embed_root, embed_name), elem_delim,
                         pretrain_file)

    pretrain_file_path = os.path.join(embed_root, embed_name, pretrain_file)

    my_embed = CustomEmbed(pretrain_file_path, elem_delim)

    assert len(my_embed) == 3
    assert my_embed.vec_len == 5
    assert my_embed.token_to_idx['a'] == 1
    assert my_embed.idx_to_token[1] == 'a'

    first_vec_sum = my_embed.idx_to_vec[0].sum().asnumpy()[0]
    assert_almost_equal(first_vec_sum, 0)

    unk_vec_sum = my_embed['<unk$unk@unk>'].sum().asnumpy()[0]
    assert_almost_equal(unk_vec_sum, 0)

    unk_vecs_sum = my_embed[['<unk$unk@unk>',
                             '<unk$unk@unk>']].sum().asnumpy()[0]
    assert_almost_equal(unk_vecs_sum, 0)


def test_text_indexer():
    counter = Counter(['a', 'b', 'b', 'c', 'c', 'c', 'some_word$'])

    g1 = TextIndexer(counter, most_freq_count=None, min_freq=1,
                     unknown_token='<unk>', reserved_tokens=None)
    assert len(g1) == 5
    assert g1.token_to_idx == {'<unk>': 0, 'c': 1, 'b': 2, 'a': 3,
                               'some_word$': 4}
    assert g1.idx_to_token[1] == 'c'
    assert g1.unknown_token == '<unk>'
    assert g1.reserved_tokens is None
    assert g1.unknown_idx == 0

    g2 = TextIndexer(counter, most_freq_count=None, min_freq=2,
                     unknown_token='<unk>', reserved_tokens=None)
    assert len(g2) == 3
    assert g2.token_to_idx == {'<unk>': 0, 'c': 1, 'b': 2}
    assert g2.idx_to_token[1] == 'c'
    assert g2.unknown_token == '<unk>'
    assert g2.reserved_tokens is None
    assert g2.unknown_idx == 0

    g3 = TextIndexer(counter, most_freq_count=None, min_freq=100,
                     unknown_token='<unk>', reserved_tokens=None)
    assert len(g3) == 1
    assert g3.token_to_idx == {'<unk>': 0}
    assert g3.idx_to_token[0] == '<unk>'
    assert g3.unknown_token == '<unk>'
    assert g3.reserved_tokens is None
    assert g3.unknown_idx == 0

    g4 = TextIndexer(counter, most_freq_count=2, min_freq=1,
                     unknown_token='<unk>', reserved_tokens=None)
    assert len(g4) == 3
    assert g4.token_to_idx == {'<unk>': 0, 'c': 1, 'b': 2}
    assert g4.idx_to_token[1] == 'c'
    assert g4.unknown_token == '<unk>'
    assert g4.reserved_tokens is None
    assert g4.unknown_idx == 0

    g5 = TextIndexer(counter, most_freq_count=3, min_freq=1,
                     unknown_token='<unk>', reserved_tokens=None)
    assert len(g5) == 4
    assert g5.token_to_idx == {'<unk>': 0, 'c': 1, 'b': 2, 'a': 3}
    assert g5.idx_to_token[1] == 'c'
    assert g5.unknown_token == '<unk>'
    assert g5.reserved_tokens is None
    assert g5.unknown_idx == 0

    g6 = TextIndexer(counter, most_freq_count=100, min_freq=1,
                     unknown_token='<unk>', reserved_tokens=None)
    assert len(g6) == 5
    assert g6.token_to_idx == {'<unk>': 0, 'c': 1, 'b': 2, 'a': 3,
                               'some_word$': 4}
    assert g6.idx_to_token[1] == 'c'
    assert g6.unknown_token == '<unk>'
    assert g6.reserved_tokens is None
    assert g6.unknown_idx == 0

    g7 = TextIndexer(counter, most_freq_count=1, min_freq=2,
                     unknown_token='<unk>', reserved_tokens=None)
    assert len(g7) == 2
    assert g7.token_to_idx == {'<unk>': 0, 'c': 1}
    assert g7.idx_to_token[1] == 'c'
    assert g7.unknown_token == '<unk>'
    assert g7.reserved_tokens is None
    assert g7.unknown_idx == 0

    assertRaises(AssertionError, TextIndexer, counter, most_freq_count=None,
                 min_freq=0, unknown_token='<unknown>',
                 reserved_tokens=['b'])

    assertRaises(AssertionError, TextIndexer, counter, most_freq_count=None,
                 min_freq=1, unknown_token='<unknown>',
                 reserved_tokens=['b', 'b'])

    assertRaises(AssertionError, TextIndexer, counter, most_freq_count=None,
                 min_freq=1, unknown_token='<unknown>',
                 reserved_tokens=['b', '<unknown>'])

    assertRaises(AssertionError, TextIndexer, counter, most_freq_count=None,
                 min_freq=1, unknown_token='a', reserved_tokens=None)

    g8 = TextIndexer(counter, most_freq_count=None, min_freq=1,
                     unknown_token='<unknown>', reserved_tokens=['b'])
    assert len(g8) == 5
    assert g8.token_to_idx == {'<unknown>': 0, 'b': 1, 'c': 2, 'a': 3,
                               'some_word$': 4}
    assert g8.idx_to_token[1] == 'b'
    assert g8.unknown_token == '<unknown>'
    assert g8.reserved_tokens == ['b']
    assert g8.unknown_idx == 0

    g9 = TextIndexer(counter, most_freq_count=None, min_freq=2,
                     unknown_token='<unk>', reserved_tokens=['b', 'a'])
    assert len(g9) == 4
    assert g9.token_to_idx == {'<unk>': 0, 'b': 1, 'a': 2, 'c': 3}
    assert g9.idx_to_token[1] == 'b'
    assert g9.unknown_token == '<unk>'
    assert g9.reserved_tokens == ['b', 'a']
    assert g9.unknown_idx == 0

    g10 = TextIndexer(counter, most_freq_count=None, min_freq=100,
                      unknown_token='<unk>', reserved_tokens=['b', 'c'])
    assert len(g10) == 3
    assert g10.token_to_idx == {'<unk>': 0, 'b': 1, 'c': 2}
    assert g10.idx_to_token[1] == 'b'
    assert g10.unknown_token == '<unk>'
    assert g10.reserved_tokens == ['b', 'c']
    assert g10.unknown_idx == 0

    g11 = TextIndexer(counter, most_freq_count=1, min_freq=2,
                      unknown_token='<unk>', reserved_tokens=['<pad>', 'b'])
    assert len(g11) == 4
    assert g11.token_to_idx == {'<unk>': 0, '<pad>': 1, 'b': 2, 'c': 3}
    assert g11.idx_to_token[1] == '<pad>'
    assert g11.unknown_token == '<unk>'
    assert g11.reserved_tokens == ['<pad>', 'b']
    assert g11.unknown_idx == 0


def test_glossary_with_one_embed():
    embed_root = '~/.mxnet/embeddings/'
    embed_name = 'my_embed'
    elem_delim = '/t'
    pretrain_file = 'my_pretrain_file1.txt'

    _mk_my_pretrain_file(os.path.join(embed_root, embed_name), elem_delim,
                         pretrain_file)

    pretrain_file_path = os.path.join(embed_root, embed_name, pretrain_file)

    my_embed = CustomEmbed(pretrain_file_path, elem_delim, unknown_vec=nd.ones)

    counter = Counter(['a', 'b', 'b', 'c', 'c', 'c', 'some_word$'])

    g1 = Glossary(counter, my_embed, most_freq_count=None, min_freq=1,
                  unknown_token='<unk>', reserved_tokens=['<pad>'])

    assert g1.token_to_idx == {'<unk>': 0, '<pad>': 1, 'c': 2, 'b': 3, 'a': 4,
                               'some_word$': 5}
    assert g1.idx_to_token == ['<unk>', '<pad>', 'c', 'b', 'a', 'some_word$']

    assert_almost_equal(g1.idx_to_vec.asnumpy(),
                        np.array([[1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [0.6, 0.7, 0.8, 0.9, 1],
                                  [0.1, 0.2, 0.3, 0.4, 0.5],
                                  [1, 1, 1, 1, 1]])
                        )

    assert g1.vec_len == 5
    assert g1.reserved_tokens == ['<pad>']

    assert_almost_equal(g1['c'].asnumpy(),
                        np.array([1, 1, 1, 1, 1])
                        )

    assert_almost_equal(g1[['c']].asnumpy(),
                        np.array([[1, 1, 1, 1, 1]])
                        )

    assert_almost_equal(g1[['a', 'not_exist']].asnumpy(),
                        np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                                  [1, 1, 1, 1, 1]])
                        )

    g1.update_token_vectors(['a', 'b'],
                            nd.array([[2, 2, 2, 2, 2],
                                      [3, 3, 3, 3, 3]])
                            )
    assert_almost_equal(g1.idx_to_vec.asnumpy(),
                        np.array([[1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [3, 3, 3, 3, 3],
                                  [2, 2, 2, 2, 2],
                                  [1, 1, 1, 1, 1]])
                        )

    assertRaises(ValueError, g1.update_token_vectors, 'unknown$$$',
                 nd.array([0, 0, 0, 0, 0]))

    g1.update_token_vectors(['<unk>'],
                            nd.array([0, 0, 0, 0, 0])
                            )
    assert_almost_equal(g1.idx_to_vec.asnumpy(),
                        np.array([[0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [3, 3, 3, 3, 3],
                                  [2, 2, 2, 2, 2],
                                  [1, 1, 1, 1, 1]])
                        )
    g1.update_token_vectors(['<unk>'],
                            nd.array([[10, 10, 10, 10, 10]])
                            )
    assert_almost_equal(g1.idx_to_vec.asnumpy(),
                        np.array([[10, 10, 10, 10, 10],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [3, 3, 3, 3, 3],
                                  [2, 2, 2, 2, 2],
                                  [1, 1, 1, 1, 1]])
                        )
    g1.update_token_vectors('<unk>',
                            nd.array([0, 0, 0, 0, 0])
                            )
    assert_almost_equal(g1.idx_to_vec.asnumpy(),
                        np.array([[0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [3, 3, 3, 3, 3],
                                  [2, 2, 2, 2, 2],
                                  [1, 1, 1, 1, 1]])
                        )
    g1.update_token_vectors('<unk>',
                            nd.array([[10, 10, 10, 10, 10]])
                            )
    assert_almost_equal(g1.idx_to_vec.asnumpy(),
                        np.array([[10, 10, 10, 10, 10],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [3, 3, 3, 3, 3],
                                  [2, 2, 2, 2, 2],
                                  [1, 1, 1, 1, 1]])
                        )


def test_glossary_with_two_embeds():
    embed_root = '.'
    embed_name = 'my_embed'
    elem_delim = '/t'
    pretrain_file1 = 'my_pretrain_file1.txt'
    pretrain_file2 = 'my_pretrain_file2.txt'

    _mk_my_pretrain_file(os.path.join(embed_root, embed_name), elem_delim,
                         pretrain_file1)
    _mk_my_pretrain_file2(os.path.join(embed_root, embed_name), elem_delim,
                          pretrain_file2)

    pretrain_file_path1 = os.path.join(embed_root, embed_name, pretrain_file1)
    pretrain_file_path2 = os.path.join(embed_root, embed_name, pretrain_file2)

    my_embed1 = CustomEmbed(pretrain_file_path1, elem_delim,
                            unknown_vec=nd.ones)
    my_embed2 = CustomEmbed(pretrain_file_path2, elem_delim)

    counter = Counter(['a', 'b', 'b', 'c', 'c', 'c', 'some_word$'])

    g1 = Glossary(counter, [my_embed1, my_embed2], most_freq_count=None,
                  min_freq=1, unknown_token='<unk>', reserved_tokens=None)

    assert g1.token_to_idx == {'<unk>': 0, 'c': 1, 'b': 2, 'a': 3,
                               'some_word$': 4}
    assert g1.idx_to_token == ['<unk>', 'c', 'b', 'a', 'some_word$']

    assert_almost_equal(g1.idx_to_vec.asnumpy(),
                        np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 0.06, 0.07, 0.08, 0.09, 0.1],
                                  [0.6, 0.7, 0.8, 0.9, 1, 0, 0, 0, 0, 0],
                                  [0.1, 0.2, 0.3, 0.4, 0.5,
                                   0.01, 0.02, 0.03, 0.04, 0.05],
                                  [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
                        )

    assert g1.vec_len == 10
    assert g1.reserved_tokens is None
    assert_almost_equal(g1['c'].asnumpy(),
                        np.array([1, 1, 1, 1, 1, 0.06, 0.07, 0.08, 0.09, 0.1])
                        )

    assert_almost_equal(g1[['b', 'not_exist']].asnumpy(),
                        np.array([[0.6, 0.7, 0.8, 0.9, 1, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
                        )

    g1.update_token_vectors(['a', 'b'],
                            nd.array([[2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                                      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]])
                            )
    assert_almost_equal(g1.idx_to_vec.asnumpy(),
                        np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 0.06, 0.07, 0.08, 0.09, 0.1],
                                  [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                                  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                                  [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
                        )

    my_embed3 = CustomEmbed(pretrain_file_path1, elem_delim,
                            unknown_token='<different_unk>')

    assertRaises(AssertionError, Glossary, counter, my_embed3,
                 most_freq_count=None, min_freq=1, unknown_token='<unk>',
                 reserved_tokens=None)

    my_embed4 = CustomEmbed(pretrain_file_path2, elem_delim,
                            unknown_token='<different_unk>')

    assertRaises(AssertionError, Glossary, counter, [my_embed3, my_embed4],
                 most_freq_count=None, min_freq=1, unknown_token='<unk>',
                 reserved_tokens=None)

    assertRaises(AssertionError, Glossary, counter, [my_embed1, my_embed3],
                 most_freq_count=None, min_freq=1, unknown_token='<unk>',
                 reserved_tokens=None)


if __name__ == '__main__':
    import nose
    nose.runmodule()
