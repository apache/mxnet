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
Help functions to support for implementing CNN + Highway Network for Text Classification in MXNet
"""

import itertools
import os
import re
from collections import Counter

import numpy as np

import word2vec
# from gensim.models import word2vec


def clean_str(string):
    """Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", r" \( ", string)
    string = re.sub(r"\)", r" \) ", string)
    string = re.sub(r"\?", r" \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    # The dataset is from https://github.com/dennybritz/cnn-text-classification-tf/tree/master/data/rt-polaritydata
    # The dataset is copyright to Denny Britz and licensed under Apache License 2.0.
    # For full text of the license, see https://github.com/dennybritz/cnn-text-classification-tf/blob/master/LICENSE
    pos_path = "./data/rt-polaritydata/rt-polarity.pos"
    neg_path = "./data/rt-polaritydata/rt-polarity.neg"
    if not os.path.exists(pos_path):
        os.system("git clone https://github.com/dennybritz/cnn-text-classification-tf.git")
        os.system('mv cnn-text-classification-tf/data .')
        os.system('rm -rf cnn-text-classification-tf')
    positive_examples = list(open(pos_path).readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(neg_path).readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def pad_sentences(sentences, padding_word="</s>"):
    """Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i, sentence in enumerate(sentences):
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """Maps sentencs and labels to vectors based on a vocabulary."""
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def build_input_data_with_word2vec(sentences, labels, word2vec_list):
    """
    Map sentences and labels to vectors based on a pretrained word2vec
    """
    x_vec = []
    for sent in sentences:
        vec = []
        for word in sent:
            if word in word2vec_list:
                vec.append(word2vec_list[word])
            else:
                vec.append(word2vec_list['</s>'])
        x_vec.append(vec)
    x_vec = np.array(x_vec)
    y_vec = np.array(labels)
    return [x_vec, y_vec]


def load_data_with_word2vec(word2vec_list):
    """Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    # vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    return build_input_data_with_word2vec(sentences_padded, labels, word2vec_list)


def load_data():
    """Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    """Generates a batch iterator for a dataset."""
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def load_pretrained_word2vec(infile):
    """Load the pre-trained word2vec from file."""
    if isinstance(infile, str):
        infile = open(infile)

    word2vec_list = {}
    for idx, line in enumerate(infile):
        if idx == 0:
            vocab_size, dim = line.strip().split()
        else:
            tks = line.strip().split()
            word2vec_list[tks[0]] = map(float, tks[1:])

    return word2vec_list


def load_google_word2vec(path):
    model = word2vec.Word2Vec.load_word2vec_format(path, binary=True)
    return model
