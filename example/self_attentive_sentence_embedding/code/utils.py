# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Some general function modules
# author: kenjewu


import argparse


def get_args():
    '''
    Parsing to get command line arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=300,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers in BiLSTM')
    parser.add_argument('--attention-unit', type=int, default=350,
                        help='number of attention unit')
    parser.add_argument('--attention-hops', type=int, default=1,
                        help='number of attention hops, for multi-hop attention model')
    parser.add_argument('--drop-prob', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='clip to prevent the too large grad in LSTM')
    parser.add_argument('--nfc', type=int, default=512,
                        help='hidden (fully connected) layer size for classifier MLP')
    parser.add_argument('--lr', type=float, default=.001,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                        help='upper epoch limit')
    parser.add_argument('--loss-name', type=str, default='sce', help='loss function name')
    parser.add_argument('--seed', type=int, default=2018,
                        help='random seed')

    parser.add_argument('--pool-way', type=str, default='flatten', help='pool att output way')
    parser.add_argument('--prune-p', type=int, default=None, help='prune p size')
    parser.add_argument('--prune-q', type=int, default=None, help='prune q size')

    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for training')
    parser.add_argument('--class-number', type=int, default=5,
                        help='number of classes')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='type of optimizer')
    parser.add_argument('--penalization-coeff', type=float, default=0.1,
                        help='the penalization coefficient')

    parser.add_argument('--save', type=str, default='../models', help='path to save the final model')
    parser.add_argument('--wv-name', type=str, choices={'glove', 'w2v', 'fasttext', 'random'},
                        default='random', help='word embedding way')
    parser.add_argument('--data-json-path', type=str, default='../data/sub_review_labels.json', help='raw data path')
    parser.add_argument('--formated-data-path', type=str,
                        default='../data/formated_data.pkl', help='formated data path')

    return parser.parse_args()
