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
from stt_bi_graphemes_util import generate_bi_graphemes_dictionary
########################################
########## FOR JUPYTER NOTEBOOK
import os

# os.environ['MXNET_ENGINE_TYPE'] = "NaiveEngine"
os.environ['MXNET_ENGINE_TYPE'] = "ThreadedEnginePerDevice"
os.environ['MXNET_ENABLE_GPU_P2P'] = "0"


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
    is_bi_graphemes = args.config.getboolean('common', 'is_bi_graphemes')
    overwrite_meta_files = args.config.getboolean('train', 'overwrite_meta_files')
    language = args.config.get('data', 'language')
    is_bi_graphemes = args.config.getboolean('common', 'is_bi_graphemes')

    labelUtil = LabelUtil.getInstance()
    if language == "en":
        if is_bi_graphemes:
            try:
                labelUtil.load_unicode_set("resources/unicodemap_en_baidu_bi_graphemes.csv")
            except:
                raise Exception("There is no resources/unicodemap_en_baidu_bi_graphemes.csv. Please set overwrite_meta_files at train section True")
        else:
            labelUtil.load_unicode_set("resources/unicodemap_en_baidu.csv")
    else:
        raise Exception("Error: Language Type: %s" % language)
    args.config.set('arch', 'n_classes', str(labelUtil.get_count()))

    if mode == 'predict':
        test_json = args.config.get('data', 'test_json')
        datagen = DataGenerator(save_dir=save_dir, model_name=model_name)
        datagen.load_train_data(test_json)
        datagen.get_meta_from_file(np.loadtxt(generate_file_path(save_dir, model_name, 'feats_mean')),
                                   np.loadtxt(generate_file_path(save_dir, model_name, 'feats_std')))
    elif mode =="train" or mode == "load":
        data_json = args.config.get('data', 'train_json')
        val_json = args.config.get('data', 'val_json')
        datagen = DataGenerator(save_dir=save_dir, model_name=model_name)
        datagen.load_train_data(data_json)
        #test bigramphems

        if overwrite_meta_files and is_bi_graphemes:
            generate_bi_graphemes_dictionary(datagen.train_texts)

        args.config.set('arch', 'n_classes', str(labelUtil.get_count()))

        if mode == "train":
            if overwrite_meta_files:
                normalize_target_k = args.config.getint('train', 'normalize_target_k')
                datagen.sample_normalize(normalize_target_k, True)
            else:
                datagen.get_meta_from_file(np.loadtxt(generate_file_path(save_dir, model_name, 'feats_mean')),
                                           np.loadtxt(generate_file_path(save_dir, model_name, 'feats_std')))
            datagen.load_validation_data(val_json)

        elif mode == "load":
            # get feat_mean and feat_std to normalize dataset
            datagen.get_meta_from_file(np.loadtxt(generate_file_path(save_dir, model_name, 'feats_mean')),
                                       np.loadtxt(generate_file_path(save_dir, model_name, 'feats_std')))
            datagen.load_validation_data(val_json)
    else:
        raise Exception(
            'Define mode in the cfg file first. train or predict or load can be the candidate for the mode.')

    is_batchnorm = args.config.getboolean('arch', 'is_batchnorm')
    if batch_size == 1 and is_batchnorm:
        raise Warning('batch size 1 is too small for is_batchnorm')

    # sort file paths by its duration in ascending order to implement sortaGrad

    if mode == "train" or mode == "load":
        max_t_count = datagen.get_max_seq_length(partition="train")
        max_label_length = datagen.get_max_label_length(partition="train",is_bi_graphemes=is_bi_graphemes)
    elif mode == "predict":
        max_t_count = datagen.get_max_seq_length(partition="test")
        max_label_length = datagen.get_max_label_length(partition="test",is_bi_graphemes=is_bi_graphemes)
    else:
        raise Exception(
            'Define mode in the cfg file first. train or predict or load can be the candidate for the mode.')

    args.config.set('arch', 'max_t_count', str(max_t_count))
    args.config.set('arch', 'max_label_length', str(max_label_length))
    from importlib import import_module
    prepare_data_template = import_module(args.config.get('arch', 'arch_file'))
    init_states = prepare_data_template.prepare_data(args)
    if mode == "train":
        sort_by_duration = True
    else:
        sort_by_duration = False

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
                          is_bi_graphemes=is_bi_graphemes)

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
                                    sort_by_duration=False,
                                    is_bi_graphemes=is_bi_graphemes)
        return data_loaded, validation_loaded, args


def load_model(args, contexts, data_train):
    # load model from model_name prefix and epoch of model_num_epoch with gpu contexts of contexts
    mode = args.config.get('common', 'mode')
    load_optimizer_states = args.config.getboolean('load', 'load_optimizer_states')
    is_start_from_batch = args.config.getboolean('load','is_start_from_batch')

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

        model_loaded = mx.module.Module.load(prefix=model_path, epoch=model_num_epoch, context=contexts,
                                             data_names=data_names, label_names=label_names,
                                             load_optimizer_states=load_optimizer_states)
        if is_start_from_batch:
            import re
            model_num_epoch = int(re.findall('\d+', model_file)[0])

    return model_loaded, model_num_epoch


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise Exception('cfg file path must be provided. ex)python main.py --configfile examplecfg.cfg')
    args = parse_args(sys.argv[1])
    # set parameters from cfg file
    # give random seed
    random_seed = args.config.getint('common', 'random_seed')
    mx_random_seed = args.config.getint('common', 'mx_random_seed')
    # random seed for shuffling data list
    if random_seed != -1:
        random.seed(random_seed)
    # set mx.random.seed to give seed for parameter initialization
    if mx_random_seed !=-1:
        mx.random.seed(mx_random_seed)
    else:
        mx.random.seed(hash(datetime.now()))
    # set log file name
    log_filename = args.config.get('common', 'log_filename')
    log = LogUtil(filename=log_filename).getlogger()

    # set parameters from data section(common)
    mode = args.config.get('common', 'mode')
    if mode not in ['train', 'predict', 'load']:
        raise Exception(
            'Define mode in the cfg file first. train or predict or load can be the candidate for the mode.')

    # get meta file where character to number conversions are defined

    contexts = parse_contexts(args)
    num_gpu = len(contexts)
    batch_size = args.config.getint('common', 'batch_size')

    # check the number of gpus is positive divisor of the batch size for data parallel
    if batch_size % num_gpu != 0:
        raise Exception('num_gpu should be positive divisor of batch_size')

    if mode == "predict":
        data_train, args = load_data(args)
    elif mode == "train" or mode == "load":
        data_train, data_val, args = load_data(args)

    # log current config
    config_logger = ConfigLogger(log)
    config_logger(args.config)

    # load model
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
        is_batchnorm = args.config.getboolean('arch', 'is_batchnorm')
        if is_batchnorm :
            for nbatch, data_batch in enumerate(data_train):
                # when is_train = False it leads to high cer when batch_norm
                model_loaded.forward(data_batch, is_train=True)
                model_loaded.update_metric(eval_metric, data_batch.label)
        else :
            model_loaded.score(eval_data=data_train, num_batch=None, eval_metric=eval_metric, reset=True)
