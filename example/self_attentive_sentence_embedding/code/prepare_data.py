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

# This module is used to parse the raw data and process the training data needed for the model.
# author: kenjewu

import mxnet as mx
import numpy as np
import gluonnlp as nlp

import os
import re
import json
import pickle
import collections
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split


UNK = '<unk>'
PAD = '<pad>'


def clean_str(string):
    """
    Tokenization/string cleaning.
    Original from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
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
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()


def pad_sequences(sequences, max_len, pad_value):
    '''
    Fill the sequence to the specified length, long truncation
    Args:
        sequences: A list of all sentences, a list of list
        max_len: Specified maximum length
        pad_value: Specified fill value
    Returns:
        pades_seqs: A numpy array
    '''

    # max_len = max(map(lambda x: len(x), sequences))

    paded_seqs = np.zeros((len(sequences), max_len))
    for idx, seq in enumerate(sequences):
        paded = None
        if len(seq) < max_len:
            paded = np.array((seq + [pad_value] * (max_len - len(seq))))
        else:
            paded = np.array(seq[0:max_len])
        paded_seqs[idx] = paded

    return paded_seqs


def get_vocab(sentences, wv_name):
    '''
    Get the vocab that is a instance of nlp.Vocab
    Args:
        sentences: all sentences, a list of str.
        wv_name: one of {'glove', 'w2v', 'fasttext', 'random'}.The way the representative word is embedded.
    Returns:
        my_vocab: a instance of nlp.Vocab
    '''
    tokens = []
    for sent in sentences:
        tokens.extend(clean_str(sent).split())

    token_counter = nlp.data.count_tokens(tokens)
    my_vocab = nlp.Vocab(token_counter)

    if wv_name == 'glove':
        my_embedding = nlp.embedding.GloVe(source='glove.6B.50d', embedding_root='..data/embedding')
    elif wv_name == 'w2v':
        my_embedding = nlp.embedding.Word2Vec(
            source='GoogleNews-vectors-negative300', embedding_root='..data/embedding')
    elif wv_name == 'fasttext':
        my_embedding = nlp.embedding.FastText(source='wiki.simple', embedding_root='..data/embedding')
    else:
        my_embedding = None

    if my_embedding is not None:
        my_vocab.set_embedding(my_embedding)

    return my_vocab


def sentences2idx(sentences, my_vocab):
    '''
    Convert all words of sentences their corresponding index in the vocabulary.
    Args:
        sentences: all sentences, a list of str.
        my_vocab: a instance of nlp.Vocab
    Retruns:
        sentences_idx: all index of all words, a list of list.
    '''
    sentences_indices = []
    for sent in sentences:
        sentences_indices.append(my_vocab.to_indices(clean_str(sent).split()))
    return sentences_indices


def get_data(data_json_path, wv_name, formated_data_path):
    '''
    Process raw data and obtain standard data that can be used for model training.
    Args:
        data_json_path: the path of raw data. This is a json file.
        wv_name: one of {'glove', 'w2v', 'fasttext', 'random'}.The way the representative word is embedded.
        formated_data_path: The path to save the processed standard data.
    Returns:
        formated_data: A dict.
    Returns
    '''

    if os.path.exists(formated_data_path):
        with open(formated_data_path, 'rb') as f:
            formated_data = pickle.load(f)
    else:
        with open(data_json_path, 'r', encoding='utf-8') as fr:
            data = json.load(fr)
        sentences, labels = data['texts'], data['labels']

        my_vocab = get_vocab(sentences, wv_name)
        pad_num_value = my_vocab.to_indices(PAD)

        # 将输入数据转为整数索引
        input_idx = sentences2idx(sentences, my_vocab)

        # 准备训练和验证数据迭代器
        max_seq_len = 100
        input_paded = pad_sequences(input_idx, max_seq_len, pad_num_value)
        labels = np.array(labels).reshape((-1, 1)) - 1

        formated_data = {'x': input_paded, 'y': labels, 'vocab': my_vocab}
        with open(formated_data_path, 'wb') as fw:
            pickle.dump(formated_data, fw)

    return formated_data
