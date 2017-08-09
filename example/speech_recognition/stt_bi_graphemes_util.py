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

import csv
from collections import Counter


def split_every(n, label):
    index = 0
    if index <= len(label) - 1 <= index + n - 1:
        yield label[index:len(label)]
        index = index + n
    while index+n-1 <= len(label)-1:
        yield label[index:index+n]
        index = index + n
        if index <= len(label)-1 <= index+n-1:
            yield label[index:len(label)]
            index=index+n

def generate_bi_graphemes_label(label):
    label_bi_graphemes = []
    label = label.split(' ')
    last_index = len(label) - 1
    for label_index, item in enumerate(label):
        for pair in split_every(2, item):
            label_bi_graphemes.append(pair)
        if label_index != last_index:
            label_bi_graphemes.append(" ")
    return label_bi_graphemes

def generate_bi_graphemes_dictionary(label_list):
    freqs = Counter()
    for label in label_list:
        label = label.split(' ')
        for i in label:
            for pair in split_every(2, i):
                if len(pair) == 2:
                    freqs[pair] += 1


    with open('resources/unicodemap_en_baidu_bi_graphemes.csv', 'w') as bigram_label:
        bigramwriter = csv.writer(bigram_label, delimiter = ',')
        baidu_labels = list('\' abcdefghijklmnopqrstuvwxyz')
        for index, key in enumerate(baidu_labels):
            bigramwriter.writerow((key, index+1))
        for index, key in enumerate(freqs.keys()):
            bigramwriter.writerow((key, index+len(baidu_labels)+1))
