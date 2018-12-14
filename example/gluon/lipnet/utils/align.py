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

"""
Module: align
This is used when the data is genrated by LipsDataset
"""

import numpy as np
from .common import word_to_vector


class Align(object):
    """
    Preprocess for Align
    """
    skip_list = ['sil', 'sp']

    def __init__(self, align_path):
        self.build(align_path)

    def build(self, align_path):
        """
        Build the align array
        """
        file = open(align_path, 'r')
        lines = file.readlines()
        file.close()
        # words: list([op, ed, word])
        words = []
        for line in lines:
            _op, _ed, word = line.strip().split(' ')
            if word not in Align.skip_list:
                words.append((int(_op), int(_ed), word))
        self.words = words
        self.n_words = len(words)
        self.sentence_str = " ".join([w[2] for w in self.words])
        self.sentence_length = len(self.sentence_str)

    def sentence(self, padding=75):
        """
        Get sentence
        """
        vec = word_to_vector(self.sentence_str)
        vec += [-1] * (padding - self.sentence_length)
        return np.array(vec, dtype=np.int32)

    def word(self, _id, padding=75):
        """
        Get words
        """
        word = self.words[_id][2]
        vec = word_to_vector(word)
        vec += [-1] * (padding - len(vec))
        return np.array(vec, dtype=np.int32)

    def word_length(self, _id):
        """
        Get the length of words
        """
        return len(self.words[_id][2])

    def word_frame_pos(self, _id):
        """
        Get the position of words
        """
        left = int(self.words[_id][0]/1000)
        right = max(left+1, int(self.words[_id][1]/1000))
        return (left, right)
