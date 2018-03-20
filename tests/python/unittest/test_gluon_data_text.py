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

from __future__ import print_function
import collections
import mxnet as mx
from mxnet.gluon import text, contrib, nn
from mxnet.gluon import data as d
from common import setup_module, with_seed

def get_frequencies(dataset):
    return collections.Counter(x for tup in dataset for x in tup[0]+tup[1][-1:])


def test_wikitext2():
    train = d.text.lm.WikiText2(root='data/wikitext-2', segment='train')
    val = d.text.lm.WikiText2(root='data/wikitext-2', segment='val')
    test = d.text.lm.WikiText2(root='data/wikitext-2', segment='test')
    train_freq, val_freq, test_freq = [get_frequencies(x) for x in [train, val, test]]
    assert len(train) == 59305, len(train)
    assert len(train_freq) == 33278, len(train_freq)
    assert len(val) == 6182, len(val)
    assert len(val_freq) == 13778, len(val_freq)
    assert len(test) == 6974, len(test)
    assert len(test_freq) == 14143, len(test_freq)
    assert test_freq['English'] == 33, test_freq['English']
    assert len(train[0][0]) == 35, len(train[0][0])
    test_no_pad = d.text.lm.WikiText2(root='data/wikitext-2', segment='test', pad=None)
    assert len(test_no_pad) == 6974, len(test_no_pad)

    train_paragraphs = d.text.lm.WikiText2(root='data/wikitext-2', segment='train', seq_len=None)
    assert len(train_paragraphs) == 23767, len(train_paragraphs)
    assert len(train_paragraphs[0][0]) != 35, len(train_paragraphs[0][0])


    vocab = text.vocab.Vocabulary(get_frequencies(train))
    def index_tokens(data, label):
        return vocab[data], vocab[label]
    nbatch_train = len(train) // 80
    train_data = d.DataLoader(train.transform(index_tokens),
                              batch_size=80,
                              sampler=contrib.data.IntervalSampler(len(train),
                                                                   nbatch_train),
                              last_batch='discard')
    sampler = contrib.data.IntervalSampler(len(train), nbatch_train)

    for i, (data, target) in enumerate(train_data):
        pass



if __name__ == '__main__':
    import nose
    nose.runmodule()
