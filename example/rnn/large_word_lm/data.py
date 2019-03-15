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

import mxnet as mx
import numpy as np
import codecs, glob, random, logging, collections

class Vocabulary(object):
    """ A dictionary for words.
        Adapeted from @rafaljozefowicz's implementation.
    """
    def __init__(self):
        self._token_to_id = {}
        self._token_to_count = collections.Counter()
        self._id_to_token = []
        self._num_tokens = 0
        self._total_count = 0
        self._s_id = None
        self._unk_id = None

    @property
    def num_tokens(self):
        return self._num_tokens

    @property
    def unk(self):
        return "<UNK>"

    @property
    def unk_id(self):
        return self._unk_id

    @property
    def s(self):
        return "<S>"

    @property
    def s_id(self):
        return self._s_id

    def add(self, token, count):
        self._token_to_id[token] = self._num_tokens
        self._token_to_count[token] = count
        self._id_to_token.append(token)
        self._num_tokens += 1
        self._total_count += count

    def finalize(self):
        self._s_id = self.get_id(self.s)
        self._unk_id = self.get_id(self.unk)

    def get_id(self, token):
        # Unseen token are mapped to UNK
        return self._token_to_id.get(token, self.unk_id)

    def get_token(self, id_):
        return self._id_to_token[id_]

    @staticmethod
    def from_file(filename):
        vocab = Vocabulary()
        with codecs.open(filename, "r", "utf-8") as f:
            for line in f:
                word, count = line.strip().split()
                vocab.add(word, int(count))
        vocab.finalize()
        return vocab

class Dataset(object):
    """ A dataset for truncated bptt with multiple sentences.
        Adapeted from @rafaljozefowicz's implementation.
     """
    def __init__(self, vocab, file_pattern, shuffle=False):
        self._vocab = vocab
        self._file_pattern = file_pattern
        self._shuffle = shuffle

    def _parse_sentence(self, line):
        s_id = self._vocab.s_id
        return [s_id] + [self._vocab.get_id(word) for word in line.strip().split()] + [s_id]

    def _parse_file(self, file_name):
        logging.debug("Processing file: %s" % file_name)
        with codecs.open(file_name, "r", "utf-8") as f:
            lines = [line.strip() for line in f]
            if not self._shuffle:
                random.shuffle(lines)
            logging.debug("Finished processing!")
            for line in lines:
                yield self._parse_sentence(line)

    def _sentence_stream(self, file_stream):
        for file_name in file_stream:
            for sentence in self._parse_file(file_name):
                yield sentence

    def _iterate(self, sentences, batch_size, num_steps):
        streams = [None] * batch_size
        x = np.zeros([batch_size, num_steps], np.int32)
        y = np.zeros([batch_size, num_steps], np.int32)
        w = np.zeros([batch_size, num_steps], np.uint8)
        while True:
            x[:] = 0
            y[:] = 0
            w[:] = 0
            for i in range(batch_size):
                tokens_filled = 0
                try:
                    while tokens_filled < num_steps:
                        if streams[i] is None or len(streams[i]) <= 1:
                            streams[i] = next(sentences)
                        num_tokens = min(len(streams[i]) - 1, num_steps - tokens_filled)
                        x[i, tokens_filled:tokens_filled+num_tokens] = streams[i][:num_tokens]
                        y[i, tokens_filled:tokens_filled + num_tokens] = streams[i][1:num_tokens+1]
                        w[i, tokens_filled:tokens_filled + num_tokens] = 1
                        streams[i] = streams[i][num_tokens:]
                        tokens_filled += num_tokens
                except StopIteration:
                    pass
            if not np.any(w):
                return

            yield x, y, w

    def iterate_once(self, batch_size, num_steps):
        def file_stream():
            file_patterns = glob.glob(self._file_pattern)
            if not self._shuffle:
                random.shuffle(file_patterns)
            for file_name in file_patterns:
                yield file_name
        for value in self._iterate(self._sentence_stream(file_stream()), batch_size, num_steps):
            yield value

    def iterate_forever(self, batch_size, num_steps):
        def file_stream():
            while True:
                file_patterns = glob.glob(self._file_pattern)
                if not self._shuffle:
                    random.shuffle(file_patterns)
                for file_name in file_patterns:
                    yield file_name
        for value in self._iterate(self._sentence_stream(file_stream()), batch_size, num_steps):
            yield value

class MultiSentenceIter(mx.io.DataIter):
    """ An MXNet iterator that returns the a batch of sequence data and label each time.
        It also returns a mask which indicates padded/missing data at the end of the dataset.
        The iterator re-shuffles the data when reset is called.
    """
    def __init__(self, data_file, vocab, batch_size, bptt):
        super(MultiSentenceIter, self).__init__()
        self.batch_size = batch_size
        self.bptt = bptt
        self.provide_data = [('data', (batch_size, bptt), np.int32), ('mask', (batch_size, bptt))]
        self.provide_label = [('label', (batch_size, bptt))]
        self.vocab = vocab
        self.data_file = data_file
        self._dataset = Dataset(self.vocab, data_file, shuffle=True)
        self._iter = self._dataset.iterate_once(batch_size, bptt)

    def iter_next(self):
        data = next(self._iter)
        if data is None:
            return False
        self._next_data = mx.nd.array(data[0], dtype=np.int32)
        self._next_label = mx.nd.array(data[1])
        self._next_mask = mx.nd.array(data[2])
        return True

    def next(self):
        if self.iter_next():
            return mx.io.DataBatch(data=self.getdata(), label=self.getlabel())
        else:
            raise StopIteration

    def reset(self):
        self._dataset = Dataset(self.vocab, self.data_file, shuffle=False)
        self._iter = self._dataset.iterate_once(self.batch_size, self.bptt)
        self._next_data = None
        self._next_label = None
        self._next_mask = None

    def getdata(self):
        return [self._next_data, self._next_mask]

    def getlabel(self):
        return [self._next_label]
