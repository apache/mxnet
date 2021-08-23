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

import logging
import os
import random

import numpy as np

import mxnet as mx
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


class Vocab(object):
    # constants for special tokens: padding, unknown, and beginning/end of sentence.
    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3
    PAD_WORD = '<blank>'
    UNK_WORD = '<unk>'
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'

    def __init__(self, filepaths=[], embedpath=None, include_unseen=False, lower=False):
        self.idx2tok = []
        self.tok2idx = {}
        self.lower = lower
        self.include_unseen = include_unseen

        self.add(Vocab.PAD_WORD)
        self.add(Vocab.UNK_WORD)
        self.add(Vocab.BOS_WORD)
        self.add(Vocab.EOS_WORD)

        self.embed = None

        for filename in filepaths:
            logging.info('loading %s'%filename)
            with open(filename, 'r') as f:
                self.load_file(f)
        if embedpath is not None:
            logging.info('loading %s'%embedpath)
            with open(embedpath, 'r') as f:
                self.load_embedding(f, reset=set([Vocab.PAD_WORD, Vocab.UNK_WORD, Vocab.BOS_WORD,
                                                  Vocab.EOS_WORD]))

    @property
    def size(self):
        return len(self.idx2tok)

    def get_index(self, key):
        return self.tok2idx.get(key.lower() if self.lower else key,
                                Vocab.UNK)

    def get_token(self, idx):
        if idx < self.size:
            return self.idx2tok[idx]
        else:
            return Vocab.UNK_WORD

    def add(self, token):
        token = token.lower() if self.lower else token
        if token in self.tok2idx:
            idx = self.tok2idx[token]
        else:
            idx = len(self.idx2tok)
            self.idx2tok.append(token)
            self.tok2idx[token] = idx
        return idx

    def to_indices(self, tokens, add_bos=False, add_eos=False):
        vec = [Vocab.BOS] if add_bos else []
        vec += [self.get_index(token) for token in tokens]
        if add_eos:
            vec.append(Vocab.EOS)
        return vec

    def to_tokens(self, indices, stop):
        tokens = []
        for i in indices:
            tokens += [self.get_token(i)]
            if i == stop:
                break
        return tokens

    def load_file(self, f):
        for line in f:
            tokens = line.rstrip('\n').split()
            for token in tokens:
                self.add(token)

    def load_embedding(self, f, reset=[]):
        vectors = {}
        for line in tqdm(f.readlines(), desc='Loading embeddings'):
            tokens = line.rstrip('\n').split(' ')
            word = tokens[0].lower() if self.lower else tokens[0]
            if self.include_unseen:
                self.add(word)
            if word in self.tok2idx:
                vectors[word] = [float(x) for x in tokens[1:]]
        dim = len(list(vectors.values())[0])
        def to_vector(tok):
            if tok in vectors and tok not in reset:
                return vectors[tok]
            elif tok not in vectors:
                return np.random.normal(-0.05, 0.05, size=dim)
            else:
                return [0.0]*dim
        self.embed = mx.nd.array([vectors[tok] if tok in vectors and tok not in reset
                                  else [0.0]*dim for tok in self.idx2tok])

class Tree(object):
    def __init__(self, idx):
        self.children = []
        self.idx = idx

    def __repr__(self):
        if self.children:
            return '{0}: {1}'.format(self.idx, str(self.children))
        else:
            return str(self.idx)

# Dataset class for SICK dataset
class SICKDataIter(object):
    def __init__(self, path, vocab, num_classes, shuffle=True):
        super(SICKDataIter, self).__init__()
        self.vocab = vocab
        self.num_classes = num_classes
        self.l_sentences = self.read_sentences(os.path.join(path,'a.toks'))
        self.r_sentences = self.read_sentences(os.path.join(path,'b.toks'))
        self.l_trees = self.read_trees(os.path.join(path,'a.parents'))
        self.r_trees = self.read_trees(os.path.join(path,'b.parents'))
        self.labels = self.read_labels(os.path.join(path,'sim.txt'))
        self.size = len(self.labels)
        self.shuffle = shuffle
        self.reset()

    def reset(self):
        if self.shuffle:
            mask = list(range(self.size))
            random.shuffle(mask)
            self.l_sentences = [self.l_sentences[i] for i in mask]
            self.r_sentences = [self.r_sentences[i] for i in mask]
            self.l_trees = [self.l_trees[i] for i in mask]
            self.r_trees = [self.r_trees[i] for i in mask]
            self.labels = [self.labels[i] for i in mask]
        self.index = 0

    def next(self):
        out = self[self.index]
        self.index += 1
        return out

    def set_context(self, context):
        self.l_sentences = [a.as_in_context(context) for a in self.l_sentences]
        self.r_sentences = [a.as_in_context(context) for a in self.r_sentences]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        l_tree = self.l_trees[index]
        r_tree = self.r_trees[index]
        l_sent = self.l_sentences[index]
        r_sent = self.r_sentences[index]
        label = self.labels[index]
        return (l_tree,l_sent,r_tree,r_sent,label)

    def read_sentence(self, line):
        indices = self.vocab.to_indices(line.split())
        return mx.nd.array(indices)

    def read_sentences(self, filename):
        with open(filename,'r') as f:
            sentences = [self.read_sentence(line) for line in f.readlines()]
        return sentences

    def read_tree(self, line):
        parents = [int(x) for x in line.split()]
        nodes = {}
        root = None
        for i in range(1,len(parents)+1):
            if i-1 not in nodes and parents[i-1]!=-1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx-1]
                    if parent == -1:
                        break
                    tree = Tree(idx)
                    if prev is not None:
                        tree.children.append(prev)
                    nodes[idx-1] = tree
                    tree.idx = idx-1
                    if parent-1 in nodes:
                        nodes[parent-1].children.append(tree)
                        break
                    elif parent==0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        return root

    def read_trees(self, filename):
        with open(filename,'r') as f:
            trees = [self.read_tree(line) for line in tqdm(f.readlines(), 'Parsing trees')]
        return trees

    def read_labels(self, filename):
        with open(filename,'r') as f:
            labels = [float(x) for x in f.readlines()]
        return labels
