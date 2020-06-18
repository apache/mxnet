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

# pylint: skip-file
import sys
import os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import *
import mxnet as mx
from numpy.random import randint
from mxnet.test_utils import assert_almost_equal


def prepare_bucketing_data(buckets, len_vocab, batch_size, invalid_label, num_sentence):
    train_sent = []
    val_sent = []

    for _ in range(num_sentence):
        len_sentence = randint(6, max(buckets)-1) # leave out the two last buckets empty
        train_sentence = []
        val_sentence = []
        for _ in range(len_sentence):
            train_sentence.append(randint(1, len_vocab))
            val_sentence.append(randint(1, len_vocab))
        train_sent.append(train_sentence)
        val_sent.append(val_sentence)

    data_train = mx.rnn.BucketSentenceIter(train_sent, batch_size, buckets=buckets,
                                   invalid_label=invalid_label)
    data_val =  mx.rnn.BucketSentenceIter(val_sent, batch_size, buckets=buckets,
                                 invalid_label=invalid_label)

    return (data_train, data_val)

def init_logging():
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)

def train_model(context=mx.cpu(), logging_info=True,
                eval_end_callback=None, batch_end_callback=None):
    if logging_info:
        init_logging()

    batch_size = 128
    num_epochs = 5
    num_hidden = 25
    num_embed = 25
    num_layers = 2
    len_vocab = 50
    buckets = [5, 10, 20, 30, 40]

    invalid_label = -1
    num_sentence = 1000

    data_train, data_val = prepare_bucketing_data(buckets, len_vocab, batch_size, invalid_label, num_sentence)

    stack = mx.rnn.SequentialRNNCell()
    for i in range(num_layers):
        stack.add(mx.rnn.LSTMCell(num_hidden=num_hidden, prefix='lstm_l%d_' % i))

    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('softmax_label')
        embed = mx.sym.Embedding(data=data, input_dim=len_vocab,
                                 output_dim=num_embed, name='embed')

        stack.reset()
        outputs, states = stack.unroll(seq_len, inputs=embed, merge_outputs=True)

        pred = mx.sym.Reshape(outputs, shape=(-1, num_hidden))
        pred = mx.sym.FullyConnected(data=pred, num_hidden=len_vocab, name='pred')

        label = mx.sym.Reshape(label, shape=(-1,))
        loss = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

        return loss, ('data',), ('softmax_label',)

    model = mx.mod.BucketingModule(
        sym_gen=sym_gen,
        default_bucket_key=data_train.default_bucket_key,
        context=context)

    if logging_info:
        logging.info('Begin fit...')

    if batch_end_callback is None:
        batch_end_callback = mx.callback.Speedometer(batch_size, 50)

    model.fit(
        train_data=data_train,
        eval_data=data_val,
        eval_metric=mx.metric.Perplexity(invalid_label), # Use Perplexity for multiclass classification.
        kvstore='device',
        optimizer='sgd',
        optimizer_params={'learning_rate': 0.01,
                          'momentum': 0,
                          'wd': 0.00001},
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        num_epoch=num_epochs,
        eval_end_callback=eval_end_callback,
        batch_end_callback=batch_end_callback)

    if logging_info:
        logging.info('Finished fit...')
    return model


def test_bucket_module():
    # For training the model in any context with the same random numbers,
    # we have to launch train_model (and not test_bucket_module()) with the same seed
    env_seed_str = os.getenv('MXNET_TEST_SEED')
    seed = int(env_seed_str) if env_seed_str else np.random.randint(0, np.iinfo(np.int32).max)
    batch_lst = []
    eval_lst = []

    @with_seed(seed)
    def train_bucket_model(context):
        def batch_end_callback(params):
            for _, val in params.eval_metric.get_global_name_value():
                batch_lst.append(val)

        def eval_end_callback(params):
            for _, val in params.eval_metric.get_name_value():
                eval_lst.append(val)

        return train_model(context, False, eval_end_callback, batch_end_callback)

    init_logging()
    manager_type = "Pooled"
    pool_type = 'Naive'
    context_lst = [mx.cpu(),  mx.gpu(0), mx.cpu_pinned()]
    # First path:  for default Pooled Storage Managers with 'Naive' strategy for pool type
    # Second path: for default Pooled Storage Managers with 'Round' strategy for pool type
    # Third path:  for default Pooled Storage Managers with 'Unpooled' strategy for pool type
    # Fourth path: for Naive storage Managers
    for j in range(4):
        for i, context in enumerate(context_lst):
            batch_lst = []
            eval_lst = []
            pool_startegy = ', pooling strategy: {}'.format(pool_type) if j < 3 else ''
            test_descr = '%s using %s Storage Manager%s' % (context, manager_type, pool_startegy)
            logging.info('\nBegin training for %s...' % test_descr)
            train_bucket_model(context)
            logging.info('Finished training for %s...' % test_descr)
            if j + i > 0:
                # Compare with the data of the first
                assert_almost_equal(cmp_array_batch, np.array(batch_lst))
                assert_almost_equal(cmp_array_eval, np.array(eval_lst))
            else:
                # Save data of the first path for comparison
                cmp_array_batch = np.array(batch_lst)
                cmp_array_eval = np.array(eval_lst)

        if j <= 1:
            # Preparing for the second and third paths
            pool_type = 'Round' if j == 0 else 'Unpooled'
            os.environ['MXNET_CPU_MEM_POOL_TYPE'] = pool_type
            os.environ['MXNET_GPU_MEM_POOL_TYPE'] = pool_type
            os.environ['MXNET_CPU_PINNED_MEM_POOL_TYPE'] = pool_type
        else:
            # Preparing for the fourth path
            context_lst.append(mx.Context('cpu_shared', 0))
            manager_type = 'Naive'
            os.environ['MXNET_USE_NAIVE_STORAGE_MANAGER'] = "1"

    # Removing environment variables
    del os.environ['MXNET_USE_NAIVE_STORAGE_MANAGER']
    del os.environ['MXNET_CPU_MEM_POOL_TYPE']
    del os.environ['MXNET_GPU_MEM_POOL_TYPE']
    del os.environ['MXNET_CPU_PINNED_MEM_POOL_TYPE']

if __name__ == "__main__":
    test_bucket_module()
