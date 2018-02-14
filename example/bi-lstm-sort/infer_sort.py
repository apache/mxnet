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

# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
import os
import argparse
import numpy as np
import mxnet as mx

from sort_io import BucketSentenceIter, default_build_vocab
from rnn_model import BiLSTMInferenceModel

TRAIN_FILE = "sort.train.txt"
TEST_FILE = "sort.test.txt"
VALID_FILE = "sort.valid.txt"
DATA_DIR = os.path.join(os.getcwd(), "data")
SEQ_LEN = 5

def MakeInput(char, vocab, arr):
    idx = vocab[char]
    tmp = np.zeros((1,))
    tmp[0] = idx
    arr[:] = tmp

def main():
    tks = sys.argv[1:]
    assert len(tks) >= 5, "Please provide 5 numbers for sorting as sequence length is 5"
    batch_size = 1
    buckets = []
    num_hidden = 300
    num_embed = 512
    num_lstm_layer = 2

    num_epoch = 1
    learning_rate = 0.1
    momentum = 0.9

    contexts = [mx.context.cpu(i) for i in range(1)]

    vocab = default_build_vocab(os.path.join(DATA_DIR, TRAIN_FILE))
    rvocab = {}
    for k, v in vocab.items():
        rvocab[v] = k

    _, arg_params, __ = mx.model.load_checkpoint("sort", 1)
    for tk in tks:
        assert (tk in vocab), "{} not in range of numbers  that  the model trained for.".format(tk)

    model = BiLSTMInferenceModel(SEQ_LEN, len(vocab),
                                 num_hidden=num_hidden, num_embed=num_embed,
                                 num_label=len(vocab), arg_params=arg_params, ctx=contexts, dropout=0.0)

    data = np.zeros((1, len(tks)))
    for k in range(len(tks)):
        data[0][k] = vocab[tks[k]]

    data = mx.nd.array(data)
    prob = model.forward(data)
    for k in range(len(tks)):
        print(rvocab[np.argmax(prob, axis = 1)[k]])


if __name__ == '__main__':
    sys.exit(main())
