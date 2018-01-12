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

# coding: utf-8

"""Read text files and load embeddings."""
from __future__ import absolute_import
from __future__ import print_function

UNKNOWN_IDX = 0

APACHE_REPO_URL = 'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/'

GLOVE_PRETRAINED_FILE_SHA1 = \
    {'glove.42B.300d.zip': 'f8e722b39578f776927465b71b231bae2ae8776a',
     'glove.6B.zip': 'b64e54f1877d2f735bdd000c1d7d771e25c7dfdc',
     'glove.840B.300d.zip': '8084fbacc2dee3b1fd1ca4cc534cbfff3519ed0d',
     'glove.twitter.27B.zip': 'dce69c404025a8312c323197347695e81fd529fc'}

GLOVE_PRETRAINED_ARCHIVE_SHA1 = \
    {'glove.42B.300d.txt': '876767977d6bd4d947c0f84d44510677bc94612a',
     'glove.6B.50d.txt': '21bf566a9d27f84d253e0cd4d4be9dcc07976a6d',
     'glove.6B.100d.txt': '16b1dbfaf35476790bd9df40c83e2dfbd05312f1',
     'glove.6B.200d.txt': '17d0355ddaa253e298ede39877d1be70f99d9148',
     'glove.6B.300d.txt': '646443dd885090927f8215ecf7a677e9f703858d',
     'glove.840B.300d.txt': '294b9f37fa64cce31f9ebb409c266fc379527708',
     'glove.twitter.27B.25d.txt':
         '767d80889d8c8a22ae7cd25e09d0650a6ff0a502',
     'glove.twitter.27B.50d.txt':
         '9585f4be97e286339bf0112d0d3aa7c15a3e864d',
     'glove.twitter.27B.100d.txt':
         '1bbeab8323c72332bd46ada0fc3c99f2faaa8ca8',
     'glove.twitter.27B.200d.txt':
         '7921c77a53aa5977b1d9ce3a7c4430cbd9d1207a'}

FAST_TEXT_FILE_SHA1 = \
    {'wiki.en.vec': 'c1e418f144ceb332b4328d27addf508731fa87df',
     'wiki.simple.vec': '55267c50fbdf4e4ae0fbbda5c73830a379d68795',
     'wiki.zh.vec': '117ab34faa80e381641fbabf3a24bc8cfba44050'}
