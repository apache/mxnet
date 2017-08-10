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

from __future__ import print_function
import find_mxnet
import submission_dsb
import mxnet as mx
import logging
import argparse
import time

parser = argparse.ArgumentParser(description='generate predictions an image classifer on Kaggle Data Science Bowl 1')
parser.add_argument('--batch-size', type=int, default=100,
                    help='the batch size')
parser.add_argument('--data-dir', type=str, default="data48/",
                    help='the input data directory')
parser.add_argument('--gpus', type=str,
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--model-prefix', type=str,default= "./models/sample_net-0",
                    help='the prefix of the model to load')
parser.add_argument('--num-round', type=int,default= 50,
                    help='the round/epoch to use')
args = parser.parse_args()



# device used
devs = mx.cpu() if args.gpus is None else [
    mx.gpu(int(i)) for i in args.gpus.split(',')]


# Load the pre-trained model
model = mx.model.FeedForward.load(args.model_prefix, args.num_round, ctx=devs, numpy_batch_size=args.batch_size)


# test set data iterator
data_shape = (3, 36, 36)
test = mx.io.ImageRecordIter(
    path_imgrec = args.data_dir + "test.rec",
    mean_r      = 128,
    mean_b      = 128,
    mean_g      = 128,
    scale       = 0.0078125,
    rand_crop   = False,
    rand_mirror = False,
    data_shape  = data_shape,
    batch_size  = args.batch_size)

# generate matrix of prediction prob
tic=time.time()
predictions = model.predict(test)
print("Time required for prediction", time.time()-tic)


# create submission csv file to submit to kaggle
submission_dsb.gen_sub(predictions,test_lst_path="data/test.lst",submission_path="submission.csv")
