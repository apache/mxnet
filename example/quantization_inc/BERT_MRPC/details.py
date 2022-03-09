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

import time
import logging
import random
import itertools
import collections
import numpy as np
import numpy.ma as ma
import gluonnlp as nlp
import mxnet as mx

from mxnet.contrib.quantization import quantize_net_v2
from gluonnlp.model import BERTClassifier as BERTModel
from gluonnlp.data import BERTTokenizer
from gluonnlp.data import GlueMRPC
from functools import partial

nlp.utils.check_version('0.9', warning_only=True)
logging.basicConfig()
logging = logging.getLogger()

CTX = mx.cpu()

TASK_NAME = 'MRPC'
MODEL_NAME = 'bert_12_768_12'
DATASET_NAME = 'book_corpus_wiki_en_uncased'
BACKBONE, VOCAB = nlp.model.get_model(name=MODEL_NAME,
                                      dataset_name=DATASET_NAME,
                                      pretrained=True,
                                      ctx=CTX,
                                      use_decoder=False,
                                      use_classifier=False)
TOKENIZER = BERTTokenizer(VOCAB, lower=('uncased' in DATASET_NAME))
MAX_LEN = int(512)

LABEL_DTYPE = 'int32'
CLASS_LABELS = ['0', '1']
NUM_CLASSES = len(CLASS_LABELS)
LABEL_MAP = {l: i for (i, l) in enumerate(CLASS_LABELS)}

BATCH_SIZE = int(32)
LR = 3e-5
EPSILON = 1e-6
LOSS_FUNCTION = mx.gluon.loss.SoftmaxCELoss()
EPOCH_NUMBER = int(4)
TRAINING_STEPS = None  # if specified, epochs will be ignored
ACCUMULATE = int(1)  # >= 1
WARMUP_RATIO = 0.1
EARLY_STOP = None
TRAINING_LOG_INTERVAL = 10*ACCUMULATE

METRIC = mx.metric.Accuracy


class FixedDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        input_ids, segment_ids, valid_length, label = self.dataset[idx]
        return input_ids, segment_ids, np.float32(valid_length), label

    def __len__(self):
        return len(self.dataset)


def truncate_seqs_equal(seqs, max_len):
    assert isinstance(seqs, list)
    lens = list(map(len, seqs))
    if sum(lens) <= max_len:
        return seqs

    lens = ma.masked_array(lens, mask=[0] * len(lens))
    while True:
        argmin = lens.argmin()
        minval = lens[argmin]
        quotient, remainder = divmod(max_len, len(lens) - sum(lens.mask))
        if minval <= quotient:  # Ignore values that don't need truncation
            lens.mask[argmin] = 1
            max_len -= minval
        else:  # Truncate all
            lens.data[~lens.mask] = [
                quotient + 1 if i < remainder else quotient for i in range(lens.count())
            ]
            break
    seqs = [seq[:length] for (seq, length) in zip(seqs, lens.data.tolist())]
    return seqs


def concat_sequences(seqs, separators, seq_mask=0, separator_mask=1):
    assert isinstance(seqs, collections.abc.Iterable) and len(seqs) > 0
    assert isinstance(seq_mask, (list, int))
    assert isinstance(separator_mask, (list, int))
    concat = sum((seq + sep for sep, seq in itertools.zip_longest(separators, seqs, fillvalue=[])),
                 [])
    segment_ids = sum(
        ([i] * (len(seq) + len(sep))
         for i, (sep, seq) in enumerate(itertools.zip_longest(separators, seqs, fillvalue=[]))),
        [])
    if isinstance(seq_mask, int):
        seq_mask = [[seq_mask] * len(seq) for seq in seqs]
    if isinstance(separator_mask, int):
        separator_mask = [[separator_mask] * len(sep) for sep in separators]

    p_mask = sum((s_mask + mask for sep, seq, s_mask, mask in itertools.zip_longest(
        separators, seqs, seq_mask, separator_mask, fillvalue=[])), [])
    return concat, segment_ids, p_mask


def convert_examples_to_features(example, is_test):
    truncate_length = MAX_LEN if is_test else MAX_LEN - 3
    if not is_test:
        example, label = example[:-1], example[-1]
        label = np.array([LABEL_MAP[label]], dtype=LABEL_DTYPE)

    tokens_raw = [TOKENIZER(l) for l in example]
    tokens_trun = truncate_seqs_equal(tokens_raw, truncate_length)
    tokens_trun[0] = [VOCAB.cls_token] + tokens_trun[0]
    tokens, segment_ids, _ = concat_sequences(tokens_trun, [[VOCAB.sep_token]] * len(tokens_trun))
    input_ids = VOCAB[tokens]
    valid_length = len(input_ids)
    if not is_test:
        return input_ids, segment_ids, valid_length, label
    else:
        return input_ids, segment_ids, valid_length


def preprocess_data():
    def preprocess_dataset(segment):
        is_calib = segment == 'calib'
        is_test = segment == 'test'
        segment = 'train' if is_calib else segment
        trans = partial(convert_examples_to_features, is_test=is_test)
        batchify = [nlp.data.batchify.Pad(axis=0, pad_val=VOCAB[VOCAB.padding_token]),  # 0. input
                    nlp.data.batchify.Pad(axis=0, pad_val=0),                           # 1. segment
                    nlp.data.batchify.Stack()]                                          # 2. length
        batchify += [] if is_test else [nlp.data.batchify.Stack(LABEL_DTYPE)]           # 3. label
        batchify_fn = nlp.data.batchify.Tuple(*batchify)

        dataset = list(map(trans, GlueMRPC(segment)))
        random.shuffle(dataset)
        dataset = mx.gluon.data.SimpleDataset(dataset)

        batch_arg = {}
        if segment == 'train' and not is_calib:
            seq_len = dataset.transform(lambda *args: args[2], lazy=False)
            sampler = nlp.data.sampler.FixedBucketSampler(seq_len, BATCH_SIZE, num_buckets=10,
                                                          ratio=0, shuffle=True)
            batch_arg['batch_sampler'] = sampler
        else:
            batch_arg['batch_size'] = BATCH_SIZE

        dataset = FixedDataset(dataset)
        return mx.gluon.data.DataLoader(dataset, num_workers=0, shuffle=False,
                                        batchify_fn=batchify_fn, **batch_arg)

    return (preprocess_dataset(seg) for seg in ['train', 'dev', 'calib'])


def log_train(batch_id, batch_num, metric, step_loss, epoch_id, learning_rate):
    """Generate and print out the log message for training. """
    metric_nm, metric_val = metric.get()
    if not isinstance(metric_nm, list):
        metric_nm, metric_val = [metric_nm], [metric_val]

    train_str = '[Epoch %d Batch %d/%d] loss=%.4f, lr=%.7f, metrics:' + \
                ','.join([i + ':%.4f' for i in metric_nm])
    logging.info(train_str, epoch_id, batch_id, batch_num, step_loss / TRAINING_LOG_INTERVAL,
                 learning_rate, *metric_val)


def finetune(model, train_dataloader, dev_dataloader, output_dir_path):
    model.classifier.initialize(init=mx.init.Normal(0.02), ctx=CTX)

    all_model_params = model.collect_params()
    optimizer_params = {'learning_rate': LR, 'epsilon': EPSILON, 'wd': 0.01}
    trainer = mx.gluon.Trainer(all_model_params, 'bertadam', optimizer_params,
                               update_on_kvstore=False)
    epochs = 9999 if TRAINING_STEPS else EPOCH_NUMBER
    batches_in_epoch = TRAINING_STEPS if TRAINING_STEPS else int(len(train_dataloader) / ACCUMULATE)
    num_train_steps = batches_in_epoch * epochs

    logging.info('training steps=%d', num_train_steps)
    num_warmup_steps = int(num_train_steps * WARMUP_RATIO)

    # Do not apply weight decay on LayerNorm and bias terms
    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0

    # Collect differentiable parameters
    params = [p for p in all_model_params.values() if p.grad_req != 'null']

    # Set grad_req if gradient accumulation is required
    if ACCUMULATE > 1:
        for p in params:
            p.grad_req = 'add'

    # track best eval score
    metric = METRIC()
    metric_history = []
    best_metric = None
    patience = EARLY_STOP

    step_num = 0
    epoch_id = 0
    finish_flag = False
    while epoch_id < epochs and not finish_flag and (not EARLY_STOP or patience > 0):
        epoch_id += 1
        metric.reset()
        step_loss = 0
        tic = time.time()
        all_model_params.zero_grad()

        for batch_id, batch in enumerate(train_dataloader):
            batch_id += 1
            # learning rate schedule
            if step_num < num_warmup_steps:
                new_lr = LR * step_num / num_warmup_steps
            else:
                non_warmup_steps = step_num - num_warmup_steps
                offset = non_warmup_steps / (num_train_steps - num_warmup_steps)
                new_lr = LR - offset * LR
            trainer.set_learning_rate(new_lr)

            # forward and backward
            with mx.autograd.record():
                input_ids, segment_ids, valid_length, label = batch
                input_ids = input_ids.as_in_context(CTX)
                valid_length = valid_length.as_in_context(CTX).astype('float32')
                label = label.as_in_context(CTX)
                out = model(input_ids, segment_ids.as_in_context(CTX), valid_length)
                ls = LOSS_FUNCTION(out, label).mean()
                ls.backward()

            # update
            if ACCUMULATE <= 1 or batch_id % ACCUMULATE == 0:
                trainer.allreduce_grads()
                nlp.utils.clip_grad_global_norm(params, 1)
                trainer.update(ACCUMULATE)
                step_num += 1
                if ACCUMULATE > 1:
                    # set grad to zero for gradient accumulation
                    all_model_params.zero_grad()

            step_loss += ls.asscalar()
            label = label.reshape((-1))
            metric.update([label], [out])
            if batch_id % TRAINING_LOG_INTERVAL == 0:
                log_train(batch_id, batches_in_epoch, metric, step_loss, epoch_id,
                          trainer.learning_rate)
                step_loss = 0
            if step_num >= num_train_steps:
                logging.info('Finish training step: %d', step_num)
                finish_flag = True
                break
        mx.nd.waitall()

        # inference on dev data
        metric_val = evaluate(model, dev_dataloader)
        if best_metric is None or metric_val >= best_metric:
            best_metric = metric_val
            patience = EARLY_STOP
        else:
            if EARLY_STOP is not None:
                patience -= 1
        metric_history.append((epoch_id, METRIC().name, metric_val))
        print('Results of evaluation on dev dataset: {}:{}'.format(METRIC().name, metric_val))

        # save params
        ckpt_name = 'model_bert_{}_{}.params'.format(TASK_NAME, epoch_id)
        params_path = (output_dir_path / ckpt_name)

        model.save_parameters(str(params_path))
        logging.info('params saved in: %s', str(params_path))
        toc = time.time()
        logging.info('Time cost=%.2fs', toc - tic)

    # we choose the best model assuming higher score stands for better model quality
    metric_history.sort(key=lambda x: x[2], reverse=True)
    best_epoch = metric_history[0]
    ckpt_name = 'model_bert_{}_{}.params'.format(TASK_NAME, best_epoch[0])
    metric_str = 'Best model at epoch {}. Validation metrics: {}:{}'.format(*best_epoch)
    logging.info(metric_str)

    model.load_parameters(str(output_dir_path / ckpt_name), ctx=CTX, cast_dtype=True)
    return model


def evaluate(model, dataloader):
    metric = METRIC()
    for batch in dataloader:
        input_ids, segment_ids, valid_length, label = batch
        input_ids = input_ids.as_in_context(CTX)
        segment_ids = segment_ids.as_in_context(CTX)
        valid_length = valid_length.as_in_context(CTX)
        label = label.as_in_context(CTX).reshape((-1))

        out = model(input_ids, segment_ids, valid_length)
        metric.update([label], [out])

    metric_name, metric_val = metric.get()
    return metric_val


def native_quantization(model, calib_dataloader, dev_dataloader):
    quantized_model = quantize_net_v2(model,
                                      quantize_mode='smart',
                                      calib_data=calib_dataloader,
                                      calib_mode='naive',
                                      num_calib_examples=BATCH_SIZE*10)
    print('Native quantization results: {}'.format(evaluate(quantized_model, dev_dataloader)))
    return quantized_model
