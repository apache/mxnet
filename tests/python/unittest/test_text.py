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

from mxnet import ndarray as nd
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

    counter_to_update = Counter({'life': 2})

    cnt3 = tu.count_tokens_from_str(str_of_tokens, token_delim, seq_delim,
                                    to_lower=False,
                                    counter_to_update=counter_to_update.copy())
    assert cnt3 == Counter(
        {'is': 2, 'life': 4, '.': 2, 'Life': 1, 'great': 1, '!': 1, 'good': 1,
         "isn't": 1, 'bad': 1})

    cnt4 = tu.count_tokens_from_str(str_of_tokens, token_delim, seq_delim,
                                    to_lower=True,
                                    counter_to_update=counter_to_update.copy())
    assert cnt4 == Counter(
        {'life': 5, 'is': 2, '.': 2, 'great': 1, '!': 1, 'good': 1,
         "isn't": 1, 'bad': 1})


def test_count_tokens_from_str():
    _test_count_tokens_from_str_with_delims(' ', '\n')
    _test_count_tokens_from_str_with_delims('IS', 'LIFE')


def test_check_pretrain_files():
    for embed_name, embed_cls in TextEmbed.embed_registry.items():
        for pretrain_file in embed_cls.pretrain_file_sha1.keys():
            TextEmbed.check_pretrain_files(pretrain_file, embed_name)


@unittest.skip('')
def test_glove():
    glove_6b_50d = TextEmbed.create_text_embed('glove',
                                               pretrain_file='glove.6B.50d.txt')

    assert len(glove_6b_50d) == 400000
    assert glove_6b_50d.vec_len == 50
    assert glove_6b_50d.token_to_idx['hi'] == 11083
    assert glove_6b_50d.idx_to_token[11083] == 'hi'

    last_vec_sum = glove_6b_50d.idx_to_vec[400000].sum().asnumpy()[0]
    assert_almost_equal(last_vec_sum, 0)

    unk_vec_sum = glove_6b_50d['<unk$unk@unk>'].sum().asnumpy()[0]
    assert_almost_equal(unk_vec_sum, 0)

    unk_vecs_sum = glove_6b_50d[['<unk$unk@unk>',
                                 '<unk$unk@unk>']].sum().asnumpy()[0]
    assert_almost_equal(unk_vecs_sum, 0)


@unittest.skip('')
def test_fasttext():
    fasttext_simple = TextEmbed.create_text_embed(
        'fasttext', pretrain_file='wiki.simple.vec')

    assert len(fasttext_simple) == 111051
    assert fasttext_simple.vec_len == 300
    assert fasttext_simple.token_to_idx['hi'] == 3240
    assert fasttext_simple.idx_to_token[3240] == 'hi'

    last_vec_sum = fasttext_simple.idx_to_vec[111051].sum().asnumpy()[0]
    assert_almost_equal(last_vec_sum, 0)

    unk_vec_sum = fasttext_simple['<unk$unk@unk>'].sum().asnumpy()[0]
    assert_almost_equal(unk_vec_sum, 0)

    unk_vecs_sum = fasttext_simple[['<unk$unk@unk>',
                                    '<unk$unk@unk>']].sum().asnumpy()[0]
    assert_almost_equal(unk_vecs_sum, 0)


def _mk_my_pretrain_file(path, token_delim, pretrain_file):
    if not os.path.exists(path):
        os.makedirs(path)
    seq1 = token_delim.join(['a', '0.1', '0.2', '0.3', '0.4', '0.5']) + '\n'
    seq2 = token_delim.join(['b', '0.1', '0.2', '0.3', '0.4', '0.5']) + '\n'
    seqs = seq1 + seq2
    with open(os.path.join(path, pretrain_file), 'w') as fout:
        fout.write(seqs)


def _mk_my_pretrain_file2(path, token_delim, pretrain_file):
    if not os.path.exists(path):
        os.makedirs(path)
    seq1 = token_delim.join(['a', '0.01', '0.02', '0.03', '0.04', '0.05']) \
           + '\n'
    seq2 = token_delim.join(['c', '0.01', '0.02', '0.03', '0.04', '0.05']) \
           + '\n'
    seqs = seq1 + seq2
    with open(os.path.join(path, pretrain_file), 'w') as fout:
        fout.write(seqs)


def test_text_embed():
    embed_root = os.path.expanduser('~/.mxnet/embeddings/')
    embed_name = 'my_embed'
    token_delim = '/t'
    pretrain_file = os.path.expanduser('my_pretrain_file.txt')

    _mk_my_pretrain_file(os.path.join(embed_root, embed_name), token_delim,
                         pretrain_file)

    my_embed = TextEmbed(os.path.join(embed_root, embed_name, pretrain_file),
                         url=None, embed_name=embed_name, embed_root=embed_root,
                         reserved_init_vec=nd.zeros, token_delim=token_delim)

    assert len(my_embed) == 2
    assert my_embed.vec_len == 5
    assert my_embed.token_to_idx['a'] == 0
    assert my_embed.idx_to_token[0] == 'a'

    last_vec_sum = my_embed.idx_to_vec[2].sum().asnumpy()[0]
    assert_almost_equal(last_vec_sum, 0)

    unk_vec_sum = my_embed['<unk$unk@unk>'].sum().asnumpy()[0]
    assert_almost_equal(unk_vec_sum, 0)

    unk_vecs_sum = my_embed[['<unk$unk@unk>',
                             '<unk$unk@unk>']].sum().asnumpy()[0]
    assert_almost_equal(unk_vecs_sum, 0)

@unittest.skip('')
def test_all_embeds():
    for embed_name, embed_cls in TextEmbed.embed_registry.items():
        print('embed_name: %s' % embed_name)
        for pretrain_file in embed_cls.pretrain_file_sha1.keys():


            print('pretrain_file: %s' % pretrain_file)
            te = TextEmbed.create_text_embed(embed_name,
                                             pretrain_file=pretrain_file)
            print(len(te))


def test_glossary_frequency_thresholds():
    counter = Counter(['a', 'b', 'b', 'c', 'c', 'c', 'some_word$'])

    g1 = Glossary(counter, top_k_freq=None, min_freq=1, unknown='<unk>',
                  other_reserveds=[], embeds=None)
    assert len(g1) == 5
    assert g1.token_to_idx == {'<unk>': 0, 'c': 1, 'b': 2, 'a': 3,
                               'some_word$': 4}
    assert g1.idx_to_token[1] == 'c'
    assert g1.unknown == '<unk>'
    assert g1.other_reserveds == []
    assert g1.idx_to_vec is None
    assert g1.vec_len == 0

    g2 = Glossary(counter, top_k_freq=None, min_freq=2, unknown='<unk>',
                  other_reserveds=[], embeds=None)
    assert len(g2) == 3
    assert g2.token_to_idx == {'<unk>': 0, 'c': 1, 'b': 2}
    assert g2.idx_to_token[1] == 'c'
    assert g2.unknown == '<unk>'
    assert g2.other_reserveds == []
    assert g2.idx_to_vec is None
    assert g2.vec_len == 0

    g3 = Glossary(counter, top_k_freq=None, min_freq=100, unknown='<unk>',
                  other_reserveds=[], embeds=None)
    assert len(g3) == 1
    assert g3.token_to_idx == {'<unk>': 0}
    assert g3.idx_to_token[0] == '<unk>'
    assert g3.unknown == '<unk>'
    assert g3.other_reserveds == []
    assert g3.idx_to_vec is None
    assert g3.vec_len == 0

    g4 = Glossary(counter, top_k_freq=2, min_freq=1, unknown='<unk>',
                  other_reserveds=[], embeds=None)
    assert len(g4) == 3
    assert g4.token_to_idx == {'<unk>': 0, 'c': 1, 'b': 2}
    assert g4.idx_to_token[1] == 'c'
    assert g4.unknown == '<unk>'
    assert g4.other_reserveds == []
    assert g4.idx_to_vec is None
    assert g4.vec_len == 0

    g5 = Glossary(counter, top_k_freq=3, min_freq=1, unknown='<unk>',
                  other_reserveds=[], embeds=None)
    assert len(g5) == 4
    assert g5.token_to_idx == {'<unk>': 0, 'c': 1, 'b': 2, 'a': 3}
    assert g5.idx_to_token[1] == 'c'
    assert g5.unknown == '<unk>'
    assert g5.other_reserveds == []
    assert g5.idx_to_vec is None
    assert g5.vec_len == 0

    g6 = Glossary(counter, top_k_freq=100, min_freq=1, unknown='<unk>',
                  other_reserveds=[], embeds=None)
    assert len(g6) == 5
    assert g6.token_to_idx == {'<unk>': 0, 'c': 1, 'b': 2, 'a': 3,
                               'some_word$': 4}
    assert g6.idx_to_token[1] == 'c'
    assert g6.unknown == '<unk>'
    assert g6.other_reserveds == []
    assert g6.idx_to_vec is None
    assert g6.vec_len == 0

    g7 = Glossary(counter, top_k_freq=1, min_freq=2, unknown='<unk>',
                  other_reserveds=[], embeds=None)
    assert len(g7) == 2
    assert g7.token_to_idx == {'<unk>': 0, 'c': 1}
    assert g7.idx_to_token[1] == 'c'
    assert g7.unknown == '<unk>'
    assert g7.other_reserveds == []
    assert g7.idx_to_vec is None
    assert g7.vec_len == 0


def test_glossary_with_one_embed():
    embed_root = os.path.expanduser('~/.mxnet/embeddings/')
    embed_name = 'my_embed'
    token_delim = '/t'
    pretrain_file = os.path.expanduser('my_pretrain_file1.txt')

    _mk_my_pretrain_file(os.path.join(embed_root, embed_name), token_delim,
                         pretrain_file)

    my_embed1 = TextEmbed(os.path.join(embed_root, embed_name, pretrain_file),
                          url=None, embed_name=embed_name,
                          embed_root=embed_root, reserved_init_vec=nd.zeros,
                          token_delim=token_delim)

    counter = Counter(['a', 'b', 'b', 'c', 'c', 'c', 'some_word$'])

    g1 = Glossary(counter, top_k_freq=None, min_freq=1, unknown='<unk>',
                  other_reserveds=['<pad>'], embeds=my_embed1)

    print(g1.token_to_idx)


def test_glossary_with_two_embeds():
    embed_root = os.path.expanduser('.')
    embed_name = 'my_embed'
    token_delim = '/t'
    pretrain_file1 = os.path.expanduser('my_pretrain_file1.txt')
    pretrain_file2 = os.path.expanduser('my_pretrain_file2.txt')

    _mk_my_pretrain_file(os.path.join(embed_root, embed_name), token_delim,
                         pretrain_file1)
    _mk_my_pretrain_file2(os.path.join(embed_root, embed_name), token_delim,
                          pretrain_file2)

    my_embed1 = TextEmbed(os.path.join(embed_root, embed_name, pretrain_file1),
                          url=None, embed_name=embed_name,
                          embed_root=embed_root, reserved_init_vec=nd.zeros,
                          token_delim=token_delim)
    my_embed2 = TextEmbed(os.path.join(embed_root, embed_name, pretrain_file2),
                          url=None, embed_name=embed_name,
                          embed_root=embed_root, reserved_init_vec=nd.zeros,
                          token_delim=token_delim)

    counter = Counter(['a', 'b', 'b', 'c', 'c', 'c', 'some_word$'])


if __name__ == '__main__':
    import nose
    nose.runmodule()
