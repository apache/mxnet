import sys

sys.path.insert(0, "../../python")
from config_util import parse_args, parse_contexts, generate_file_path
from train import do_training
import mxnet as mx
from stt_io_iter import STTIter
from label_util import LabelUtil
from log_util import LogUtil

import numpy as np
from stt_datagenerator import DataGenerator
from stt_metric import STTMetric
from datetime import datetime
########################################
########## FOR JUPYTER NOTEBOOK
import os

# os.environ['MXNET_ENGINE_TYPE'] = "NaiveEngine"
os.environ['MXNET_ENGINE_TYPE'] = "ThreadedEnginePerDevice"


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


def load_data(args):
    mode = args.config.get('common', 'mode')
    batch_size = args.config.getint('common', 'batch_size')

    whcs = WHCS()
    whcs.width = args.config.getint('data', 'width')
    whcs.height = args.config.getint('data', 'height')
    whcs.channel = args.config.getint('data', 'channel')
    whcs.stride = args.config.getint('data', 'stride')
    save_dir = 'checkpoints'
    model_name = args.config.get('common', 'prefix')

    if mode == 'predict':
        test_json = args.config.get('data', 'test_json')
        datagen = DataGenerator(save_dir=save_dir, model_name=model_name)
        datagen.load_train_data(test_json)
        datagen.get_meta_from_file(np.loadtxt(generate_file_path(save_dir, model_name, 'feats_mean')),
                                   np.loadtxt(generate_file_path(save_dir, model_name, 'feats_std')))
    else:
        data_json = args.config.get('data', 'train_json')
        val_json = args.config.get('data', 'val_json')
        datagen = DataGenerator(save_dir=save_dir, model_name=model_name)
        datagen.load_train_data(data_json)
        datagen.load_validation_data(val_json)

        if mode == "train":
            normalize_target_k = args.config.getint('train', 'normalize_target_k')
            datagen.sample_normalize(normalize_target_k, True)
        elif mode == "load":
            # get feat_mean and feat_std to normalize dataset
            datagen.get_meta_from_file(np.loadtxt(generate_file_path(save_dir, model_name, 'feats_mean')),
                                       np.loadtxt(generate_file_path(save_dir, model_name, 'feats_std')))

    is_batchnorm = args.config.getboolean('arch', 'is_batchnorm')
    if batch_size == 1 and is_batchnorm:
        raise Warning('batch size 1 is too small for is_batchnorm')

    # sort file paths by its duration in ascending order to implement sortaGrad

    if mode == "train" or mode == "load":
        max_t_count = datagen.get_max_seq_length(partition="train")
        max_label_length = datagen.get_max_label_length(partition="train")
    elif mode == "predict":
        max_t_count = datagen.get_max_seq_length(partition="test")
        max_label_length = datagen.get_max_label_length(partition="test")
    else:
        raise Exception(
            'Define mode in the cfg file first. train or predict or load can be the candidate for the mode.')

    args.config.set('arch', 'max_t_count', str(max_t_count))
    args.config.set('arch', 'max_label_length', str(max_label_length))
    from importlib import import_module
    prepare_data_template = import_module(args.config.get('arch', 'arch_file'))
    init_states = prepare_data_template.prepare_data(args)
    if mode == "train":
        sort_by_duration=True
        shuffle=False
    else:
        sort_by_duration=False
	shuffle=True

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
                          shuffle=shuffle)

    if mode == 'predict':
        return data_loaded, args
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
                                    sort_by_duration=True,
                                    shuffle=False)
        return data_loaded, validation_loaded, args


def load_model(args, contexts, data_train):
    # load model from model_name prefix and epoch of model_num_epoch with gpu contexts of contexts
    mode = args.config.get('common', 'mode')
    load_optimizer_states = args.config.getboolean('load', 'load_optimizer_states')

    from importlib import import_module
    symbol_template = import_module(args.config.get('arch', 'arch_file'))
    model_loaded = symbol_template.arch(args)

    if mode == 'train':
        model_num_epoch = None
    else:
        model_file = args.config.get('common', 'model_file')
        model_name = os.path.splitext(model_file)[0]
        model_num_epoch = int(model_name[-4:])

        model_path = 'checkpoints/' + str(model_name[:-5])

        data_names = [x[0] for x in data_train.provide_data]
        label_names = [x[0] for x in data_train.provide_label]

        if load_optimizer_states is True:
            model_loaded = mx.module.Module.load(prefix=model_path, epoch=model_num_epoch, context=contexts,
                                                 data_names=data_names, label_names=label_names,
                                                 load_optimizer_states=True)
        else:
            model_loaded = mx.module.Module.load(prefix=model_path, epoch=model_num_epoch, context=contexts,
                                                 data_names=data_names, label_names=label_names,
                                                 load_optimizer_states=False)

    return model_loaded, model_num_epoch


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise Exception('cfg file path must be provided. ex)python main.py --configfile examplecfg.cfg')
    mx.random.seed(hash(datetime.now()))
    # set parameters from cfg file
    args = parse_args(sys.argv[1])

    log_filename = args.config.get('common', 'log_filename')
    log = LogUtil(filename=log_filename).getlogger()

    # set parameters from data section(common)
    mode = args.config.get('common', 'mode')
    if mode not in ['train', 'predict', 'load']:
        raise Exception(
            'Define mode in the cfg file first. train or predict or load can be the candidate for the mode.')

    # get meta file where character to number conversions are defined
    language = args.config.get('data', 'language')
    labelUtil = LabelUtil.getInstance()
    if language == "en":
        labelUtil.load_unicode_set("resources/unicodemap_en_baidu.csv")
    else:
        raise Exception("Error: Language Type: %s" % language)
    args.config.set('arch', 'n_classes', str(labelUtil.get_count()))

    contexts = parse_contexts(args)
    num_gpu = len(contexts)
    batch_size = args.config.getint('common', 'batch_size')

    # check the number of gpus is positive divisor of the batch size
    if batch_size % num_gpu != 0:
        raise Exception('num_gpu should be positive divisor of batch_size')

    if mode == "predict":
        data_train, args = load_data(args)
    elif mode == "train" or mode == "load":
        data_train, data_val, args = load_data(args)

    # log current config
    config_logger = ConfigLogger(log)
    config_logger(args.config)

    model_loaded, model_num_epoch = load_model(args, contexts, data_train)

    # if mode is 'train', it trains the model
    if mode == 'train':
        data_names = [x[0] for x in data_train.provide_data]
        label_names = [x[0] for x in data_train.provide_label]
        module = mx.mod.Module(model_loaded, context=contexts, data_names=data_names, label_names=label_names)
        do_training(args=args, module=module, data_train=data_train, data_val=data_val)
    # if mode is 'load', it loads model from the checkpoint and continues the training.
    elif mode == 'load':
        do_training(args=args, module=model_loaded, data_train=data_train, data_val=data_val, begin_epoch=model_num_epoch+1)
    # if mode is 'predict', it predict label from the input by the input model
    elif mode == 'predict':
        # predict through data
        model_loaded.bind(for_training=False, data_shapes=data_train.provide_data,
                          label_shapes=data_train.provide_label)
        max_t_count = args.config.getint('arch', 'max_t_count')
        eval_metric = STTMetric(batch_size=batch_size, num_gpu=num_gpu, seq_length=max_t_count)
        model_loaded.score(eval_data=data_train, num_batch=None, eval_metric=eval_metric, reset=True)
