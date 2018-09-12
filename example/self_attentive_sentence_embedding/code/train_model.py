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

# This is the training script
# author: kenjewu

import os
import warnings
warnings.filterwarnings('ignore')
import mxnet as mx
import gluonnlp as nlp
from mxnet import nd, gluon, init
from mxnet.gluon.data import ArrayDataset, DataLoader

import train_helper as th
from utils import get_args
from prepare_data import get_data
from models import SelfAttentiveBiLSTM
from weighted_softmaxCE import WeightedSoftmaxCE


def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()."""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx


if __name__ == '__main__':
    # 解析参数 (Parsing command line arguments)
    args = get_args()
    emsize = args.emsize
    nhide = args.nhid
    nlayers = args.nlayers
    att_unit = args.attention_unit
    att_hops = args.attention_hops
    nfc = args.nfc
    nclass = args.class_number
    drop_prob = args.drop_prob
    pool_way = args.pool_way
    prune_p = args.prune_p
    prune_q = args.prune_q

    penal_coeff = args.penalization_coeff
    optim = args.optimizer
    lr = args.lr
    num_epochs = args.epochs
    batch_size = args.batch_size
    loss_name = args.loss_name
    clip = args.clip

    # 设置 mxnet 随机数种子 (Set mxnet random number seed)
    mx.random.seed(args.seed)

    # 设置 gpu 或者 cpu (set the useful of gpu or cpu)
    ctx = try_gpu()

    # 获取训练数据与验证数据集 (Get training data and validation data set)
    print('Getting the data...')
    data = get_data(args.data_json_path, args.wv_name, args.formated_data_path)
    x, y, my_vocab = data['x'], data['y'], data['vocab']

    if any([args.wv_name == 'glove', args.wv_name == 'fasttext', args.wv_name == 'w2v']):
        embedding_weights = my_vocab.embedding.idx_to_vec
    else:
        embedding_weights = None

    data_set = ArrayDataset(nd.array(x, ctx=ctx), nd.array(y, ctx=ctx))
    train_data_set, valid_data_set = nlp.data.train_valid_split(data_set, 0.01)
    data_iter_train = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, last_batch='rollover')
    data_iter_valid = DataLoader(valid_data_set, batch_size=batch_size, shuffle=False)

    # 配置模型 (Configuration model)
    vocab_len = len(my_vocab)
    model = SelfAttentiveBiLSTM(vocab_len, emsize, nhide, nlayers, att_unit, att_hops, nfc, nclass,
                                drop_prob, pool_way, prune_p, prune_q)
    model.initialize(init=init.Xavier(), ctx=ctx)
    model.hybridize()
    if embedding_weights is not None:
        model.embedding_layer.weight.set_data(embedding_weights)
        model.embedding_layer.collect_params().setattr('grad_req', 'null')

    trainer = gluon.Trainer(model.collect_params(), optim, {'learning_rate': lr})

    class_weight = None
    if loss_name == 'sce':
        loss = gluon.loss.SoftmaxCrossEntropyLoss()
    elif loss_name == 'wsce':
        loss = WeightedSoftmaxCE()
        class_weight = nd.array([3.0, 5.3, 4.0, 2.0, 1.0], ctx=ctx)

    # 训练 (Train)
    th.train(data_iter_train, data_iter_valid, model, loss, trainer, ctx,
             num_epochs, penal_coeff=penal_coeff, clip=clip, class_weight=class_weight, loss_name=loss_name)

    # 保存模型 (Save the structure and parameters of the model)
    model_dir = args.save
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'self_att_bilstm_model')
    model.export(model_path)
    print('模型训练完毕，训练好的模型已经保存于：', model_path)
