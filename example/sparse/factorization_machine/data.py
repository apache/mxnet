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

import os, gzip, argparse, sys
import mxnet as mx
import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.INFO, format=head)

class DummyIter(mx.io.DataIter):
    "A dummy iterator that always return the same batch, used for speed testing"
    def __init__(self, real_iter):
        super(DummyIter, self).__init__()
        self.real_iter = real_iter
        self.provide_data = real_iter.provide_data
        self.provide_label = real_iter.provide_label
        self.batch_size = real_iter.batch_size

        for batch in real_iter:
            self.the_batch = batch
            break

    def __iter__(self):
        return self

    def next(self):
        return self.the_batch


def get_criteo_data(data_dir):
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    try:
        logging.info("Downloading dataset criteo to " + data_dir + " now ...")
        os.system("aws s3 cp --recursive --no-sign-request s3://sparse-dataset/criteo " + data_dir)
    except Exception as e:
        logging.error(e)

parser = argparse.ArgumentParser(description="Download criteo dataset",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir', type=str, default='./data/',
                    help='destination directory to store criteo LibSVM dataset.')

if __name__ == '__main__':
    # arg parser
    args = parser.parse_args()
    logging.info(args)
    get_criteo_data(args.dir)
