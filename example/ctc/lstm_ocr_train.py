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
""" An example of using WarpCTC loss for an OCR problem using LSTM and CAPTCHA image data"""

from __future__ import print_function

import argparse
import logging
import os

from captcha_generator import MPDigitCaptcha
from hyperparams import Hyperparams
from ctc_metrics import CtcMetrics
import lstm
import mxnet as mx
from ocr_iter import OCRIter


def get_fonts(path):
    fonts = list()
    if os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.endswith('.ttf'):
                fonts.append(os.path.join(path, filename))
    else:
        fonts.append(path)
    return fonts


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("font_path", help="Path to ttf font file or directory containing ttf files")
    parser.add_argument("--loss", help="'ctc' or 'warpctc' loss [Default 'ctc']", default='ctc')
    parser.add_argument("--cpu",
                        help="Number of CPUs for training [Default 8]. Ignored if --gpu is specified.",
                        type=int, default=8)
    parser.add_argument("--gpu", help="Number of GPUs for training [Default 0]", type=int)
    parser.add_argument("--num_proc", help="Number CAPTCHA generating processes [Default 4]", type=int, default=4)
    parser.add_argument("--prefix", help="Checkpoint prefix [Default 'ocr']", default='ocr')
    return parser.parse_args()


def main():
    args = parse_args()
    if not any(args.loss == s for s in ['ctc', 'warpctc']):
        raise ValueError("Invalid loss '{}' (must be 'ctc' or 'warpctc')".format(args.loss))

    hp = Hyperparams()

    # Start a multiprocessor captcha image generator
    mp_captcha = MPDigitCaptcha(
        font_paths=get_fonts(args.font_path), h=hp.seq_length, w=30,
        num_digit_min=3, num_digit_max=4, num_processes=args.num_proc, max_queue_size=hp.batch_size * 2)
    try:
        # Must call start() before any call to mxnet module (https://github.com/apache/incubator-mxnet/issues/9213)
        mp_captcha.start()

        if args.gpu:
            contexts = [mx.context.gpu(i) for i in range(args.gpu)]
        else:
            contexts = [mx.context.cpu(i) for i in range(args.cpu)]

        init_states = lstm.init_states(hp.batch_size, hp.num_lstm_layer, hp.num_hidden)

        data_train = OCRIter(
            hp.train_epoch_size // hp.batch_size, hp.batch_size, init_states, captcha=mp_captcha, name='train')
        data_val = OCRIter(
            hp.eval_epoch_size // hp.batch_size, hp.batch_size, init_states, captcha=mp_captcha, name='val')

        symbol = lstm.lstm_unroll(
            num_lstm_layer=hp.num_lstm_layer,
            seq_len=hp.seq_length,
            num_hidden=hp.num_hidden,
            num_label=hp.num_label,
            loss_type=args.loss)

        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=head)

        module = mx.mod.Module(
            symbol,
            data_names=['data', 'l0_init_c', 'l0_init_h', 'l1_init_c', 'l1_init_h'],
            label_names=['label'],
            context=contexts)

        metrics = CtcMetrics(hp.seq_length)
        module.fit(train_data=data_train,
                   eval_data=data_val,
                   # use metrics.accuracy or metrics.accuracy_lcs
                   eval_metric=mx.metric.np(metrics.accuracy, allow_extra_outputs=True),
                   optimizer='sgd',
                   optimizer_params={'learning_rate': hp.learning_rate,
                                     'momentum': hp.momentum,
                                     'wd': 0.00001,
                                     },
                   initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
                   num_epoch=hp.num_epoch,
                   batch_end_callback=mx.callback.Speedometer(hp.batch_size, 50),
                   epoch_end_callback=mx.callback.do_checkpoint(args.prefix),
                   )
    except KeyboardInterrupt:
        print("W: interrupt received, stopping...")
    finally:
        # Reset multiprocessing captcha generator to stop processes
        mp_captcha.reset()


if __name__ == '__main__':
    main()

