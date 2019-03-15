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

import json
import os
import sys
from collections import namedtuple
from datetime import datetime
from config_util import parse_args, parse_contexts, generate_file_path
from train import do_training
import mxnet as mx
from stt_io_iter import STTIter
from label_util import LabelUtil
from log_util import LogUtil
import numpy as np
from stt_datagenerator import DataGenerator
from stt_metric import STTMetric
from stt_bi_graphemes_util import generate_bi_graphemes_dictionary
from stt_bucketing_module import STTBucketingModule
from stt_io_bucketingiter import BucketSTTIter
sys.path.insert(0, "../../python")

# os.environ['MXNET_ENGINE_TYPE'] = "NaiveEngine"
os.environ['MXNET_ENGINE_TYPE'] = "ThreadedEnginePerDevice"
os.environ['MXNET_ENABLE_GPU_P2P'] = "0"

logUtil = LogUtil.getInstance()

class WHCS:
    width = 0
    height = 0
    channel = 0
    stride = 0

class ConfigLogger(object):
    def __init__(self, log):
        self.__log = log

    def __call__(self, config):
        self.__log.info("Config:")
        config.write(self)

    def write(self, data):
        # stripping the data makes the output nicer and avoids empty lines
        line = data.strip()
        self.__log.info(line)

def load_labelutil(labelUtil, is_bi_graphemes, language="en"):
    if language == "en":
        if is_bi_graphemes:
            try:
                labelUtil.load_unicode_set("resources/unicodemap_en_baidu_bi_graphemes.csv")
            except:
                raise Exception("There is no resources/unicodemap_en_baidu_bi_graphemes.csv." +
                                " Please set overwrite_bi_graphemes_dictionary True at train section")
        else:
            labelUtil.load_unicode_set("resources/unicodemap_en_baidu.csv")
    else:
        raise Exception("Error: Language Type: %s" % language)



def load_data(args):
    mode = args.config.get('common', 'mode')
    if mode not in ['train', 'predict', 'load']:
        raise Exception('mode must be the one of the followings - train,predict,load')
    batch_size = args.config.getint('common', 'batch_size')

    whcs = WHCS()
    whcs.width = args.config.getint('data', 'width')
    whcs.height = args.config.getint('data', 'height')
    whcs.channel = args.config.getint('data', 'channel')
    whcs.stride = args.config.getint('data', 'stride')
    save_dir = 'checkpoints'
    model_name = args.config.get('common', 'prefix')
    is_bi_graphemes = args.config.getboolean('common', 'is_bi_graphemes')
    overwrite_meta_files = args.config.getboolean('train', 'overwrite_meta_files')
    overwrite_bi_graphemes_dictionary = args.config.getboolean('train', 'overwrite_bi_graphemes_dictionary')
    max_duration = args.config.getfloat('data', 'max_duration')
    language = args.config.get('data', 'language')

    log = logUtil.getlogger()
    labelUtil = LabelUtil.getInstance()
    if mode == "train" or mode == "load":
        data_json = args.config.get('data', 'train_json')
        val_json = args.config.get('data', 'val_json')
        datagen = DataGenerator(save_dir=save_dir, model_name=model_name)
        datagen.load_train_data(data_json, max_duration=max_duration)
        datagen.load_validation_data(val_json, max_duration=max_duration)
        if is_bi_graphemes:
            if not os.path.isfile("resources/unicodemap_en_baidu_bi_graphemes.csv") or overwrite_bi_graphemes_dictionary:
                load_labelutil(labelUtil=labelUtil, is_bi_graphemes=False, language=language)
                generate_bi_graphemes_dictionary(datagen.train_texts+datagen.val_texts)
        load_labelutil(labelUtil=labelUtil, is_bi_graphemes=is_bi_graphemes, language=language)
        args.config.set('arch', 'n_classes', str(labelUtil.get_count()))

        if mode == "train":
            if overwrite_meta_files:
                log.info("Generate mean and std from samples")
                normalize_target_k = args.config.getint('train', 'normalize_target_k')
                datagen.sample_normalize(normalize_target_k, True)
            else:
                log.info("Read mean and std from meta files")
                datagen.get_meta_from_file(
                    np.loadtxt(generate_file_path(save_dir, model_name, 'feats_mean')),
                    np.loadtxt(generate_file_path(save_dir, model_name, 'feats_std')))
        elif mode == "load":
            # get feat_mean and feat_std to normalize dataset
            datagen.get_meta_from_file(
                np.loadtxt(generate_file_path(save_dir, model_name, 'feats_mean')),
                np.loadtxt(generate_file_path(save_dir, model_name, 'feats_std')))

    elif mode == 'predict':
        test_json = args.config.get('data', 'test_json')
        datagen = DataGenerator(save_dir=save_dir, model_name=model_name)
        datagen.load_train_data(test_json, max_duration=max_duration)
        labelutil = load_labelutil(labelUtil, is_bi_graphemes, language="en")
        args.config.set('arch', 'n_classes', str(labelUtil.get_count()))
        datagen.get_meta_from_file(
            np.loadtxt(generate_file_path(save_dir, model_name, 'feats_mean')),
            np.loadtxt(generate_file_path(save_dir, model_name, 'feats_std')))

    is_batchnorm = args.config.getboolean('arch', 'is_batchnorm')
    if batch_size == 1 and is_batchnorm and (mode == 'train' or mode == 'load'):
        raise Warning('batch size 1 is too small for is_batchnorm')

    # sort file paths by its duration in ascending order to implement sortaGrad
    if mode == "train" or mode == "load":
        max_t_count = datagen.get_max_seq_length(partition="train")
        max_label_length = \
            datagen.get_max_label_length(partition="train", is_bi_graphemes=is_bi_graphemes)
    elif mode == "predict":
        max_t_count = datagen.get_max_seq_length(partition="test")
        max_label_length = \
            datagen.get_max_label_length(partition="test", is_bi_graphemes=is_bi_graphemes)

    args.config.set('arch', 'max_t_count', str(max_t_count))
    args.config.set('arch', 'max_label_length', str(max_label_length))
    from importlib import import_module
    prepare_data_template = import_module(args.config.get('arch', 'arch_file'))
    init_states = prepare_data_template.prepare_data(args)
    sort_by_duration = (mode == "train")
    is_bucketing = args.config.getboolean('arch', 'is_bucketing')
    save_feature_as_csvfile = args.config.getboolean('train', 'save_feature_as_csvfile')
    if is_bucketing:
        buckets = json.loads(args.config.get('arch', 'buckets'))
        data_loaded = BucketSTTIter(partition="train",
                                    count=datagen.count,
                                    datagen=datagen,
                                    batch_size=batch_size,
                                    num_label=max_label_length,
                                    init_states=init_states,
                                    seq_length=max_t_count,
                                    width=whcs.width,
                                    height=whcs.height,
                                    sort_by_duration=sort_by_duration,
                                    is_bi_graphemes=is_bi_graphemes,
                                    buckets=buckets,
                                    save_feature_as_csvfile=save_feature_as_csvfile)
    else:
        data_loaded = STTIter(partition="train",
                              count=datagen.count,
                              datagen=datagen,
                              batch_size=batch_size,
                              num_label=max_label_length,
                              init_states=init_states,
                              seq_length=max_t_count,
                              width=whcs.width,
                              height=whcs.height,
                              sort_by_duration=sort_by_duration,
                              is_bi_graphemes=is_bi_graphemes,
                              save_feature_as_csvfile=save_feature_as_csvfile)

    if mode == 'train' or mode == 'load':
        if is_bucketing:
            validation_loaded = BucketSTTIter(partition="validation",
                                              count=datagen.val_count,
                                              datagen=datagen,
                                              batch_size=batch_size,
                                              num_label=max_label_length,
                                              init_states=init_states,
                                              seq_length=max_t_count,
                                              width=whcs.width,
                                              height=whcs.height,
                                              sort_by_duration=False,
                                              is_bi_graphemes=is_bi_graphemes,
                                              buckets=buckets,
                                              save_feature_as_csvfile=save_feature_as_csvfile)
        else:
            validation_loaded = STTIter(partition="validation",
                                        count=datagen.val_count,
                                        datagen=datagen,
                                        batch_size=batch_size,
                                        num_label=max_label_length,
                                        init_states=init_states,
                                        seq_length=max_t_count,
                                        width=whcs.width,
                                        height=whcs.height,
                                        sort_by_duration=False,
                                        is_bi_graphemes=is_bi_graphemes,
                                        save_feature_as_csvfile=save_feature_as_csvfile)
        return data_loaded, validation_loaded, args
    elif mode == 'predict':
        return data_loaded, args


def load_model(args, contexts, data_train):
    # load model from model_name prefix and epoch of model_num_epoch with gpu contexts of contexts
    mode = args.config.get('common', 'mode')
    load_optimizer_states = args.config.getboolean('load', 'load_optimizer_states')
    is_start_from_batch = args.config.getboolean('load', 'is_start_from_batch')

    from importlib import import_module
    symbol_template = import_module(args.config.get('arch', 'arch_file'))
    is_bucketing = args.config.getboolean('arch', 'is_bucketing')

    if mode == 'train':
        if is_bucketing:
            bucketing_arch = symbol_template.BucketingArch(args)
            model_loaded = bucketing_arch.get_sym_gen()
        else:
            model_loaded = symbol_template.arch(args)
        model_num_epoch = None
    elif mode == 'load' or mode == 'predict':
        model_file = args.config.get('common', 'model_file')
        model_name = os.path.splitext(model_file)[0]
        model_num_epoch = int(model_name[-4:])
        if is_bucketing:
            bucketing_arch = symbol_template.BucketingArch(args)
            model_loaded = bucketing_arch.get_sym_gen()
        else:
            model_path = 'checkpoints/' + str(model_name[:-5])

            data_names = [x[0] for x in data_train.provide_data]
            label_names = [x[0] for x in data_train.provide_label]

            model_loaded = mx.module.Module.load(
                prefix=model_path, epoch=model_num_epoch, context=contexts,
                data_names=data_names, label_names=label_names,
                load_optimizer_states=load_optimizer_states)
        if is_start_from_batch:
            import re
            model_num_epoch = int(re.findall('\d+', model_file)[0])

    return model_loaded, model_num_epoch


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise Exception('cfg file path must be provided. ' +
                        'ex)python main.py --configfile examplecfg.cfg')
    args = parse_args(sys.argv[1])
    # set parameters from cfg file
    # give random seed
    random_seed = args.config.getint('common', 'random_seed')
    mx_random_seed = args.config.getint('common', 'mx_random_seed')
    # random seed for shuffling data list
    if random_seed != -1:
        np.random.seed(random_seed)
    # set mx.random.seed to give seed for parameter initialization
    if mx_random_seed != -1:
        mx.random.seed(mx_random_seed)
    else:
        mx.random.seed(hash(datetime.now()))
    # set log file name
    log_filename = args.config.get('common', 'log_filename')
    log = logUtil.getlogger(filename=log_filename)

    # set parameters from data section(common)
    mode = args.config.get('common', 'mode')
    if mode not in ['train', 'predict', 'load']:
        raise Exception(
            'Define mode in the cfg file first. ' +
            'train or predict or load can be the candidate for the mode.')

    # get meta file where character to number conversions are defined

    contexts = parse_contexts(args)
    num_gpu = len(contexts)
    batch_size = args.config.getint('common', 'batch_size')
    # check the number of gpus is positive divisor of the batch size for data parallel
    if batch_size % num_gpu != 0:
        raise Exception('num_gpu should be positive divisor of batch_size')
    if mode == "train" or mode == "load":
        data_train, data_val, args = load_data(args)
    elif mode == "predict":
        data_train, args = load_data(args)
    is_batchnorm = args.config.getboolean('arch', 'is_batchnorm')
    is_bucketing = args.config.getboolean('arch', 'is_bucketing')

    # log current config
    config_logger = ConfigLogger(log)
    config_logger(args.config)

    # load model
    model_loaded, model_num_epoch = load_model(args, contexts, data_train)
    # if mode is 'train', it trains the model
    if mode == 'train':
        if is_bucketing:
            module = STTBucketingModule(
                sym_gen=model_loaded,
                default_bucket_key=data_train.default_bucket_key,
                context=contexts
                )
        else:
            data_names = [x[0] for x in data_train.provide_data]
            label_names = [x[0] for x in data_train.provide_label]
            module = mx.mod.Module(model_loaded, context=contexts,
                                   data_names=data_names, label_names=label_names)
        do_training(args=args, module=module, data_train=data_train, data_val=data_val)
    # if mode is 'load', it loads model from the checkpoint and continues the training.
    elif mode == 'load':
        do_training(args=args, module=model_loaded, data_train=data_train, data_val=data_val,
                    begin_epoch=model_num_epoch + 1)
    # if mode is 'predict', it predict label from the input by the input model
    elif mode == 'predict':
        # predict through data
        if is_bucketing:
            max_t_count = args.config.getint('arch', 'max_t_count')
            load_optimizer_states = args.config.getboolean('load', 'load_optimizer_states')
            model_file = args.config.get('common', 'model_file')
            model_name = os.path.splitext(model_file)[0]
            model_num_epoch = int(model_name[-4:])

            model_path = 'checkpoints/' + str(model_name[:-5])
            model = STTBucketingModule(
                sym_gen=model_loaded,
                default_bucket_key=data_train.default_bucket_key,
                context=contexts
                )

            model.bind(data_shapes=data_train.provide_data,
                       label_shapes=data_train.provide_label,
                       for_training=True)
            _, arg_params, aux_params = mx.model.load_checkpoint(model_path, model_num_epoch)
            model.set_params(arg_params, aux_params)
            model_loaded = model
        else:
            model_loaded.bind(for_training=False, data_shapes=data_train.provide_data,
                              label_shapes=data_train.provide_label)
        max_t_count = args.config.getint('arch', 'max_t_count')
        eval_metric = STTMetric(batch_size=batch_size, num_gpu=num_gpu)
        if is_batchnorm:
            for nbatch, data_batch in enumerate(data_train):
                model_loaded.forward(data_batch, is_train=False)
                model_loaded.update_metric(eval_metric, data_batch.label)
        else:
            #model_loaded.score(eval_data=data_train, num_batch=None,
            #                   eval_metric=eval_metric, reset=True)
            for nbatch, data_batch in enumerate(data_train):
                model_loaded.forward(data_batch, is_train=False)
                model_loaded.update_metric(eval_metric, data_batch.label)
    else:
        raise Exception(
            'Define mode in the cfg file first. ' +
            'train or predict or load can be the candidate for the mode')
